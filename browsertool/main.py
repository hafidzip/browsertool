"""
BrowserControl Tool
======================
Controls Tauri child webviews via Python Playwright CDP.
The CDP port is read from the OPENCHAD_CDP_PORT environment variable (default 9222).

The tool accepts a single `code` parameter: a Python code string that is
executed directly. Use print() to produce output. The following globals
are pre-injected:

    page      - Playwright Page object for the target webview
    asyncio   - stdlib asyncio
    base64    - stdlib base64
    json      - stdlib json
    re        - stdlib re
    _snapshot(data) - converts raw accessibility snapshot to {snapshot, refs}

"""

import shutil
import sys
import os
from openchadpy.main import get_tauri_product_name
from openchadpy.main import get_tauri_identifier
from openchadpy.event_emitter import event_emitter
from uuid import uuid4
from openchadpy.context import cdp_ports
import asyncio
import base64
import io
import json
import logging
import re
import ast
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from openchadpy.tool_base import ToolBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accessibility snapshot helpers
# ---------------------------------------------------------------------------

INTERACTIVE_ROLES = {
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "menuitem", "option", "searchbox", "spinbutton", "slider",
    "switch", "treeitem", "gridcell", "row", "listitem", "menuitemcheckbox",
    "menuitemradio",
}


def _build_snapshot_text(node: dict, counter: list, refs: dict, indent: int = 0) -> List[str]:
    if node is None:
        return []
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")
    checked = node.get("checked", None)
    disabled = node.get("disabled", False)
    expanded = node.get("expanded", None)
    is_interactive = (
        role in INTERACTIVE_ROLES
        or bool(name and role not in {"none", "presentation", "generic", "group", "region"})
    )
    ref = None
    if is_interactive:
        counter[0] += 1
        ref = f"e{counter[0]}"
        refs[ref] = {"role": role, "name": name}
    parts: List[str] = [str(" " * (indent * 2))]
    if ref:
        parts.append(str(f"[{ref}] "))
    parts.append(str(role))
    if name:
        parts.append(str(f' "{name}"'))
    if value:
        parts.append(str(f' value={value!r}'))
    if checked is not None:
        parts.append(str(f' checked={checked}'))
    if expanded is not None:
        parts.append(str(f' expanded={expanded}'))
    if disabled:
        parts.append(str(" (disabled)"))
    lines: List[str] = [str("".join(parts))]
    for child in node.get("children", []):
        lines.extend(_build_snapshot_text(child, counter, refs, indent + 1))
    return lines


def _build_snapshot(data: dict) -> Dict[str, Any]:
    """Convert raw accessibility snapshot dict to standard {snapshot, refs} result."""
    if not data:
        return {"ok": True, "snapshot": "(empty page)", "refs": {}}
    counter = [0]
    refs: dict = {}
    lines = _build_snapshot_text(data, counter, refs)
    return {"ok": True, "snapshot": "\n".join(lines), "refs": refs}


# ---------------------------------------------------------------------------
# Code wrapping helpers
# ---------------------------------------------------------------------------

def _anchor_dedent(code: str) -> str:
    """Dedent ``code`` anchored to its first non-empty line.

    ``textwrap.dedent`` removes only the *common minimum* leading whitespace
    across all non-whitespace-only lines.  When a template literal's closing
    backtick sits at a shallower column than the code body, the minimum is
    determined by that closing line and the actual code lines still carry a
    uniform residual offset.  This helper first applies ``textwrap.dedent``,
    then strips any residual prefix using the first non-empty line as the
    anchor, so the result always starts at column 0.
    """
    text = textwrap.dedent(code).strip("\n")
    lines = text.splitlines()
    # Find the first non-empty line and measure its leading whitespace.
    base = 0
    for line in lines:
        if line.strip():
            base = len(line) - len(line.lstrip())
            break
    if base == 0:
        return text
    # Strip exactly `base` spaces from the start of every line (non-empty ones
    # that start with fewer spaces are left-stripped to avoid an IndexError).
    out = []
    for line in lines:
        if line.strip():
            out.append(line[base:] if line.startswith(" " * base) else line.lstrip())
        else:
            out.append(line)
    return "\n".join(out)


def _wrap_code(code: str, indent: str = "    ") -> str:
    """Wrap user code in an async function so it can be awaited.

    Performs a structural AST normalisation pass on the body before
    wrapping so that non-standard indentation widths (e.g. 8-space try
    bodies, misaligned try/except emitted by some LLMs) are fixed
    *before* the compile step rather than discovered at runtime.
    Falls back to anchor-dedent + re-indent if the body contains real
    syntax errors — those will surface at compile time.
    """
    cleaned = _anchor_dedent(code)
    # Structural normalisation: AST round-trip fixes relative misalignment
    # (e.g. `except` at a different depth than its `try`).
    try:
        cleaned = ast.unparse(ast.parse(cleaned))
    except SyntaxError:
        # Real syntax error — leave as-is; compile will report it properly.
        pass
    indented = textwrap.indent(cleaned, indent)
    return f"async def __browser_code__():\n{indented}\n"


def _heal_indentation(wrapped: str, indent: str = "    ") -> str:
    """Re-indent the body of ``async def __browser_code__():`` so it compiles.

    Mirrors ``heal_indentation`` from ``code_sandbox.py``:

    1. Locate the function header line.
    2. Try an AST parse/unparse round-trip on the body (fixes *relative*
       misalignment between sibling blocks such as try/except).
    3. Fall back to a purely textual uniform-shift when the body itself
       has syntax errors.
    """
    lines = wrapped.splitlines(keepends=True)
    header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("async def __browser_code__", "def __browser_code__")):
            header_idx = i
            break

    if header_idx is None:
        # Fallback: ensure every non-empty line starts with indent.
        out = []
        for line in lines:
            stripped = line.strip()
            if stripped and not line.startswith(indent):
                out.append(indent + stripped + "\n")
            else:
                out.append(line)
        return "".join(out)

    header = "".join(lines[: header_idx + 1])
    body_raw = "".join(lines[header_idx + 1 :])

    # First attempt: structural normalisation via AST round-trip.
    try:
        dedented_body = textwrap.dedent(body_raw).strip("\n")
        normalized = ast.unparse(ast.parse(dedented_body))
        body_fixed = textwrap.indent(normalized, indent) + "\n"
        return header + body_fixed
    except SyntaxError as e:
        logger.warning(
            f"[browser_control] _heal_indentation AST round-trip failed, "
            f"falling back to uniform-shift heuristic: {e}"
        )

    # Fallback: anchor-dedent then uniform shift.
    # Using _anchor_dedent instead of plain textwrap.dedent so that residual
    # base indentation left by template-literal closing backticks is also
    # removed before we re-apply the wrapper indent.
    body_fixed = textwrap.indent(_anchor_dedent(body_raw), indent) + "\n"
    return header + body_fixed


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class BrowserTool(ToolBase):
    """Control any Tauri child webview by running Python / Playwright code."""

    name = "browser"
    description = (
        "Execute Python/Playwright code inside a Tauri webview via CDP. "
        "Use print() to produce output — all printed text is returned as the result. "
        "Pre-injected globals: `page`, `asyncio`, "
        "`base64`, `json`, `re`, and `_snapshot(data)` (accessibility snapshot helper)."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute. Use print() for output. Pre-injected globals:\n"
                    "  page      - Playwright Page for the target webview\n"
                    "  asyncio, base64, json, re - stdlib modules\n"
                    "  _snapshot(data) - builds {snapshot, refs} from accessibility data\n\n"
                    "Examples:\n"
                    "  # Accessibility snapshot\n"
                    "  data = await page.accessibility.snapshot(interesting_only=False)\n"
                    "  print(json.dumps(_snapshot(data)))\n\n"
                    "  # Screenshot (base64)\n"
                    "  raw = await page.screenshot(type='png')\n"
                    "  print(base64.b64encode(raw).decode())\n\n"
                    "  # Navigate\n"
                    "  await page.goto('https://example.com', wait_until='domcontentloaded')\n"
                    "  print(page.url)\n\n"
                    "  # Click and type\n"
                    "  await page.locator('#q').fill('hello')\n"
                    "  await page.keyboard.press('Enter')\n"
                    "  print('done')\n\n"
                    "  # Run JS\n"
                    "  title = await page.evaluate('document.title')\n"
                    "  print(title)\n"
                ),
            },
        },
        "required": ["code"],
    }

    allowed_callers = ["direct", "code_execution", "mcp_client"]

    fields = [
        {
            'name': '(Optional) Browser Profile',
            'value': {
                'type': 'string',
                'placeholder': 'Browser ID...'
            }
        },
        {
            'name': '(Optional) URL',
            'value': {
                'type': 'string',
                'placeholder': 'https://...'
            }
        },
    ]

    # ------------------------------------------------------------------ helpers

    def _get_page(self, context: BrowserContext, target_webview: str | None) -> Page:
        pages = context.pages
        if not pages:
            raise RuntimeError("No pages found in the Tauri CDP context.")
        if not target_webview:
            return pages[0]
        for page in pages:
            if page.type == "page":
                return page
        logger.warning(f"[browser_control] Webview '{target_webview}' not found; using first page")
        return pages[0]

    # ------------------------------------------------------------------ execute

    async def execute(self, **kwargs) -> Dict[str, Any]:
        if sys.platform != 'win32':
            return {"error": "Tool only available for windows"}

        code: str = kwargs.get("code", "").strip()
        if not code:
            return {"error": "'code' parameter is required and must not be empty."}

        profile: str = self.get_field("(Optional) Browser Profile") or "shared"
        url: str = self.get_field("(Optional) URL") or "about:blank"

        logger.info(f"[URL]: {url}")

        request_label = f"webview-agent-{str(uuid4())}"

        if re.search(r"[^\w\-]", profile):
            import hashlib
            profile = f"temp-{request_label}"

        if not re.search(r"^https?\:\/\/", url):
            url = "about:blank"

        await event_emitter.emit("create_browser", {"label": request_label, "storage": profile, "url": "about:blank"})

        label = ""
        # Wait up to 60s for the browser to appear in cdp_ports
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            labels = list(cdp_ports.keys())
            label = next((l for l in labels if l in request_label), "")
            if label:
                break
            await asyncio.sleep(0.5)

        if label == "":
            return {"error": "Timed out waiting for CDP port for target_webview after 60s."}

        cdp_port: int | None = cdp_ports.get(label, None)
        if cdp_port is None:
            return {"error": "No CDP port found for target_webview."}

        cdp_url = f"http://localhost:{cdp_port}"

        try:
            async with async_playwright() as p:
                browser: Browser = await p.chromium.connect_over_cdp(cdp_url)
                contexts = browser.contexts
                if not contexts:
                    return {"error": "No browser contexts found. Is the app running?"}
                context: BrowserContext = contexts[0]

                for ctx in contexts:
                    logger.info(f"Context: {ctx}")
                    for pg in ctx.pages:
                        logger.info(f"  [{pg.url}]")

                page: Page = self._get_page(context, request_label or None)
                
                if url != "about:blank":
                    await page.goto(url)

                # ---------------------------------------------------------
                # Build exec globals
                # ---------------------------------------------------------
                exec_globals: Dict[str, Any] = {
                    "__builtins__": __builtins__,
                    # browser objects
                    "page": page,
                    # stdlib
                    "asyncio": asyncio,
                    "base64": base64,
                    "json": json,
                    "re": re,
                    # convenience helper
                    "_snapshot": _build_snapshot,
                }

                exec_locals: Dict[str, Any] = {}

                # ---------------------------------------------------------
                # Wrap & compile
                # ---------------------------------------------------------
                wrapped = _wrap_code(code)
                logger.info(f"[browser_control] executing:\n{wrapped}")

                try:
                    compiled = compile(wrapped, "<browser_code>", "exec")
                except IndentationError as _ie:
                    logger.warning(
                        f"[browser_control] IndentationError – attempting heal: {_ie}"
                    )
                    wrapped = _heal_indentation(wrapped)
                    logger.info(f"[browser_control] healed wrapped_code:\n{wrapped}")
                    try:
                        compiled = compile(wrapped, "<browser_code>", "exec")
                    except SyntaxError as exc:
                        return {"error": f"SyntaxError in code: {exc}", "code": wrapped}
                except SyntaxError as exc:
                    return {"error": f"SyntaxError in code: {exc}", "code": wrapped}

                # ---------------------------------------------------------
                # Run
                # ---------------------------------------------------------
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                error = None

                try:
                    exec(compiled, exec_globals, exec_locals)
                    user_fn = exec_locals["__browser_code__"]
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        await user_fn()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    logger.exception("[browser_control] Error executing user code")

                return {
                    "output": stdout_capture.getvalue() or None,
                    "error": error or stderr_capture.getvalue() or None,
                    "success": error is None,
                }

        except Exception as exc:
            logger.exception("[browser_control] Error connecting to CDP")
            return {"error": str(exc), "success": False}
        finally:
            await event_emitter.emit("delete_browser", {"label": request_label})
            await asyncio.sleep(1)
            if profile.startswith("temp-"):
                try:
                    if sys.platform == 'win32':
                        local_app_data = os.getenv('LOCALAPPDATA')
                        if local_app_data:
                            identifier = get_tauri_identifier()
                            product_name = get_tauri_product_name()
                            paths = [
                                os.path.join(local_app_data, identifier, "browser-data", profile),
                                os.path.join(local_app_data, product_name, "browser-data", profile)
                            ]
                            for path in paths:
                                if os.path.exists(path):
                                    shutil.rmtree(path)
                                    logger.info(f"Deleted Windows browser data: {path}")                         
                    elif sys.platform == 'darwin':
                        import struct, uuid
                        def get_mac_store_uuid(lbl: str) -> str:
                            FNV_PRIME = 0x00000100000001B3
                            SEED_A = 0xcbf29ce484222325
                            SEED_B = 0x14650fb0739d0383
                            a = SEED_A
                            b = SEED_B
                            MASK = 0xFFFFFFFFFFFFFFFF
                            for byte in lbl.encode('utf-8'):
                                a ^= byte
                                a = (a * FNV_PRIME) & MASK
                                b ^= (byte + 0x5A) & 0xFF
                                b = (b * FNV_PRIME) & MASK
                            a_bytes = struct.pack('<Q', a)
                            b_bytes = struct.pack('<Q', b)
                            id_bytes = a_bytes + b_bytes
                            return str(uuid.UUID(bytes=id_bytes))
                        
                        uuid_str = get_mac_store_uuid(dir_label)
                        home = os.path.expanduser("~")
                        identifier = get_tauri_identifier()
                        product_name = get_tauri_product_name()
                        
                        dirs_to_check = []
                        for base_dir in ["WebKit", "Application Support"]:
                            for app_id in [identifier, product_name]:
                                for u in [uuid_str.upper(), uuid_str.lower()]:
                                    dirs_to_check.append(os.path.join(home, "Library", base_dir, app_id, "WebsiteData", u))
                                    dirs_to_check.append(os.path.join(home, "Library", base_dir, app_id, u))
                                    
                        for path in dirs_to_check:
                            if os.path.exists(path):
                                shutil.rmtree(path)
                                logger.info(f"Deleted macOS browser data: {path}")
                except Exception as e:
                    logger.error(f"Error in delete_browser_data: {e}", exc_info=True)


# Required export
Tool = BrowserTool
