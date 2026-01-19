"""
Async browser automation tools using Steel Browser + Playwright.

Uses Playwright's async API for compatibility with asyncio.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# Global async Playwright instance
_playwright = None
_browser = None
_context = None
_page = None


async def _get_playwright():
    """Get or create async Playwright instance."""
    global _playwright
    if _playwright is None:
        from playwright.async_api import async_playwright
        _playwright = await async_playwright().start()
    return _playwright


async def _get_session():
    """Get or create browser session connected to Steel."""
    global _browser, _context, _page
    
    # Validate existing connection
    if _page is not None:
        try:
            # Quick check if page is still usable
            await _page.title()
            return _page
        except Exception:
            # Connection broken, reset and reconnect
            _browser = None
            _context = None
            _page = None
    
    import http.client
    import json as json_mod
    
    steel_url = os.environ.get("STEEL_BASE_URL", "http://127.0.0.1:3000")
    parsed_host = steel_url.replace("http://", "").replace("https://", "")
    
    # Find existing Steel session or create new one
    session_id = None
    try:
        conn = http.client.HTTPConnection(parsed_host, timeout=5)
        conn.request("GET", "/v1/sessions")
        resp = conn.getresponse()
        data = json_mod.loads(resp.read())
        conn.close()
        
        # First try to find a live session, then fall back to idle
        for status_to_find in ["live", "idle"]:
            for sess in data.get("sessions", []):
                if sess.get("status") == status_to_find:
                    session_id = sess["id"]
                    break
            if session_id:
                break
    except Exception:
        pass
    
    # Create new session if none found
    if not session_id:
        try:
            conn = http.client.HTTPConnection(parsed_host, timeout=10)
            conn.request("POST", "/v1/sessions", json_mod.dumps({}), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status >= 400:
                raise RuntimeError(f"Failed to create session: HTTP {resp.status}")
            data = json_mod.loads(resp.read())
            conn.close()
            session_id = data["id"]
        except Exception as e:
            raise RuntimeError(f"No Steel session found and failed to create one: {e}. Is Steel running at {steel_url}?")
    
    # Connect via CDP
    pw = await _get_playwright()
    ws_url = f"ws://{parsed_host}/v1/cdp/{session_id}"
    _browser = await pw.chromium.connect_over_cdp(ws_url, timeout=15000)
    
    # Get existing context and page
    _context = _browser.contexts[0] if _browser.contexts else await _browser.new_context()
    
    # Find page with content (not about:blank)
    _page = None
    if _context.pages:
        for p in _context.pages:
            if p.url and p.url != "about:blank":
                _page = p
                break
        if _page is None:
            _page = _context.pages[-1]
    else:
        _page = await _context.new_page()
    
    return _page


async def browser_get_context_async(max_elements: int = 100, max_chars: int = 10000) -> str:
    """Get page context via aria snapshot (YAML accessibility tree).
    
    Returns what's actually VISIBLE - buttons, links, inputs, headings, etc.
    
    Args:
        max_elements: Ignored (kept for API compatibility)
        max_chars: Maximum characters for aria snapshot (default 10000)
    
    Returns:
        JSON with URL, title, and aria_snapshot
    """
    page = await _get_session()
    
    aria_snapshot = await page.locator("body").aria_snapshot()
    
    truncated = False
    if len(aria_snapshot) > max_chars:
        aria_snapshot = aria_snapshot[:max_chars]
        truncated = True
    
    result = {
        "url": page.url,
        "title": await page.title(),
        "aria_snapshot": aria_snapshot,
    }
    if truncated:
        result["truncated"] = True
    
    return json.dumps(result, indent=2)


async def browser_get_text_async(max_length: int = 15000) -> str:
    """Get all visible text content via aria snapshot.
    
    Args:
        max_length: Maximum characters to return (default 15000)
    
    Returns:
        JSON with URL, title, and text
    """
    page = await _get_session()
    
    full_text = await page.locator("body").aria_snapshot()
    
    truncated = False
    if len(full_text) > max_length:
        full_text = full_text[:max_length]
        truncated = True
    
    result = {
        "url": page.url,
        "title": await page.title(),
        "text": full_text,
        "length": len(full_text),
    }
    if truncated:
        result["truncated"] = True
    
    return json.dumps(result, indent=2)


async def browser_navigate_async(url: str, wait_until: str = "domcontentloaded") -> str:
    """Navigate to a URL.
    
    Args:
        url: URL to navigate to
        wait_until: Wait condition (domcontentloaded, load, networkidle)
    
    Returns:
        JSON with URL, title, and status
    """
    page = await _get_session()
    
    response = await page.goto(url, wait_until=wait_until)
    
    return json.dumps({
        "url": page.url,
        "title": await page.title(),
        "status": response.status if response else None,
    }, indent=2)


async def browser_click_async(selector: str = "", text: str = "", timeout: int = 10) -> str:
    """Click an element by selector or text.
    
    Args:
        selector: CSS selector
        text: Text content to find
        timeout: Timeout in seconds (default 10)
    
    Returns:
        JSON with success status
    """
    page = await _get_session()
    timeout_ms = timeout * 1000
    
    try:
        if text:
            await page.get_by_text(text).first.click(timeout=timeout_ms)
        elif selector:
            await page.locator(selector).first.click(timeout=timeout_ms)
        else:
            raise ValueError("Must provide selector or text")
        return json.dumps({"success": True, "url": page.url})
    except Exception as e:
        raise RuntimeError(f"browser_click failed: {e}")


async def browser_fill_async(value: str, selector: str = "", label: str = "", timeout: int = 10) -> str:
    """Fill an input field.
    
    Args:
        value: Text to fill
        selector: CSS selector
        label: Label text to find input by
        timeout: Timeout in seconds (default 10)
    
    Returns:
        JSON with success status
    """
    page = await _get_session()
    timeout_ms = timeout * 1000
    
    try:
        if label:
            await page.get_by_label(label).fill(value, timeout=timeout_ms)
        elif selector:
            await page.locator(selector).fill(value, timeout=timeout_ms)
        else:
            raise ValueError("Must provide selector or label")
        return json.dumps({"success": True})
    except Exception as e:
        raise RuntimeError(f"browser_fill failed: {e}")


async def browser_screenshot_async(full_page: bool = False) -> str:
    """Take a screenshot.
    
    Args:
        full_page: Capture full scrollable page
    
    Returns:
        JSON with base64-encoded screenshot
    """
    import base64
    
    page = await _get_session()
    
    screenshot = await page.screenshot(full_page=full_page)
    b64 = base64.b64encode(screenshot).decode()
    
    return json.dumps({
        "url": page.url,
        "title": await page.title(),
        "screenshot_base64": b64,
        "size": len(screenshot),
    }, indent=2)


async def browser_scroll_async(direction: str = "down", amount: int = 500) -> str:
    """Scroll the page.
    
    Args:
        direction: Scroll direction (up, down, left, right)
        amount: Pixels to scroll (default 500)
    
    Returns:
        JSON with success status
    """
    page = await _get_session()
    
    if direction == "down":
        await page.mouse.wheel(0, amount)
    elif direction == "up":
        await page.mouse.wheel(0, -amount)
    elif direction == "right":
        await page.mouse.wheel(amount, 0)
    elif direction == "left":
        await page.mouse.wheel(-amount, 0)
    
    return json.dumps({"success": True, "direction": direction, "amount": amount})


async def browser_wait_for_async(selector: str = "", text: str = "", timeout_seconds: int = 30) -> str:
    """Wait for an element to appear.
    
    Args:
        selector: CSS selector to wait for
        text: Text content to wait for
        timeout_seconds: Maximum wait time (default 30)
    
    Returns:
        JSON with success status
    """
    page = await _get_session()
    
    timeout_ms = timeout_seconds * 1000
    
    if text:
        await page.get_by_text(text).first.wait_for(timeout=timeout_ms)
    elif selector:
        await page.locator(selector).first.wait_for(timeout=timeout_ms)
    else:
        return json.dumps({"error": "Must provide selector or text"})
    
    return json.dumps({"success": True})


async def browser_extract_text_async(selector: str = "") -> str:
    """Extract text from specific element.
    
    Args:
        selector: CSS selector (default: body)
    
    Returns:
        JSON with extracted text
    """
    page = await _get_session()
    
    if selector:
        text = await page.locator(selector).inner_text()
    else:
        text = await page.locator("body").inner_text()
    
    return json.dumps({
        "url": page.url,
        "text": text[:10000],  # Limit length
        "truncated": len(text) > 10000,
    }, indent=2)


async def browser_close_async() -> str:
    """Close browser session."""
    global _browser, _context, _page
    
    if _browser:
        await _browser.close()
    
    _browser = None
    _context = None
    _page = None
    
    return json.dumps({"success": True})
