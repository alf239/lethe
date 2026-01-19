"""
Browser automation tools using Steel Browser + Playwright.

Steel provides the browser infrastructure (session management, anti-bot, proxy).
Playwright provides the browser control via CDP.
Accessibility Tree extraction provides token-efficient context for LLMs.

Design principles:
- Extract what's VISIBLE, not what's in the DOM (accessibility tree)
- Return structured data to minimize tokens
- Persist auth via Steel Profiles (no re-login every session)

Uses Playwright's sync API for compatibility with Lethe's synchronous tool execution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Lazy imports to avoid startup cost when browser tools aren't used
_playwright_sync = None
_playwright_instance = None  # Singleton Playwright instance
_steel = None


def _get_playwright_sync():
    global _playwright_sync
    if _playwright_sync is None:
        from playwright.sync_api import sync_playwright
        _playwright_sync = sync_playwright
    return _playwright_sync


_playwright_lock = None  # Will be initialized on first use


def _get_playwright_instance():
    """Get or create the singleton Playwright instance.
    
    Thread-safe via lock. Creates Playwright instance on first call.
    """
    global _playwright_instance, _playwright_lock
    import threading
    
    # Initialize lock on first call (can't do at module level in all contexts)
    if _playwright_lock is None:
        _playwright_lock = threading.Lock()
    
    with _playwright_lock:
        if _playwright_instance is None:
            sync_playwright = _get_playwright_sync()
            _playwright_instance = sync_playwright().start()
        return _playwright_instance


def _get_steel():
    global _steel
    if _steel is None:
        from steel import Steel
        _steel = Steel
    return _steel


# =============================================================================
# Profile Storage (persists profile IDs between runs)
# =============================================================================

def _get_profiles_path() -> Path:
    """Get path to profiles storage file."""
    data_dir = Path(os.getenv("LETHE_DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "browser_profiles.json"


def _load_profiles() -> dict:
    """Load saved profiles from disk."""
    path = _get_profiles_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_profiles(profiles: dict):
    """Save profiles to disk."""
    path = _get_profiles_path()
    path.write_text(json.dumps(profiles, indent=2))


def _get_profile_id(name: str) -> Optional[str]:
    """Get profile ID by name."""
    profiles = _load_profiles()
    return profiles.get(name, {}).get("profile_id")


def _save_profile(name: str, profile_id: str, description: str = ""):
    """Save a profile ID with a name."""
    profiles = _load_profiles()
    profiles[name] = {
        "profile_id": profile_id,
        "description": description,
    }
    _save_profiles(profiles)


def _delete_profile(name: str) -> bool:
    """Delete a saved profile by name."""
    profiles = _load_profiles()
    if name in profiles:
        del profiles[name]
        _save_profiles(profiles)
        return True
    return False


# =============================================================================
# Browser Session
# =============================================================================

def _get_storage_path(profile_name: str) -> Path:
    """Get path to storage state file for a profile."""
    data_dir = Path(os.getenv("LETHE_DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"browser_storage_{profile_name}.json"


@dataclass
class BrowserSession:
    """Manages a browser session with Playwright control.
    
    Supports two modes:
    - Local mode (headless=False): Visible browser on your machine, persistent storage
    - Steel mode: Headless browser via Steel API (local Docker or Cloud)
    """
    
    session_id: Optional[str] = None
    profile_id: Optional[str] = None
    profile_name: Optional[str] = None
    steel_client: Any = None
    playwright: Any = None
    browser: Any = None
    context: Any = None
    page: Any = None
    _steel_session: Any = None
    
    # Configuration
    steel_api_key: Optional[str] = field(default_factory=lambda: os.getenv("STEEL_API_KEY"))
    steel_base_url: str = field(default_factory=lambda: os.getenv("STEEL_BASE_URL", "http://127.0.0.1:3000"))
    use_proxy: bool = False
    solve_captcha: bool = False
    session_timeout: int = 1800  # 30 minutes default
    persist_profile: bool = False
    local_mode: bool = False  # If True, run visible browser locally (no Steel)
    headless: bool = False  # Only for local mode
    
    def start(
        self,
        profile_name: Optional[str] = None,
        profile_id: Optional[str] = None,
        persist_profile: bool = False,
        local_mode: bool = False,
        headless: bool = False,
    ) -> "BrowserSession":
        """Start a new browser session.
        
        Args:
            profile_name: Named profile for local storage persistence
            profile_id: Steel profile ID (Steel Cloud only)
            persist_profile: Whether to save profile state when session ends
            local_mode: If True, run visible browser locally (no Steel)
            headless: If True and local_mode, run headless (no visible window)
        """
        sync_playwright = _get_playwright_sync()
        
        self.profile_name = profile_name
        self.profile_id = profile_id
        self.persist_profile = persist_profile or bool(profile_name)
        self.local_mode = local_mode
        self.headless = headless
        
        # Get shared Playwright instance (avoids creating new one in thread context)
        self.playwright = _get_playwright_instance()
        
        if local_mode:
            # LOCAL MODE: Run browser directly on machine
            self.browser = self.playwright.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            
            # Load storage state if profile exists
            storage_state = None
            if profile_name:
                storage_path = _get_storage_path(profile_name)
                if storage_path.exists():
                    storage_state = str(storage_path)
            
            # Create context with storage state
            self.context = self.browser.new_context(
                storage_state=storage_state,
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            self.page = self.context.new_page()
            self.session_id = f"local-{id(self.browser)}"
            
        else:
            # STEEL MODE: Use HTTP directly (SDK has SSL issues in threads)
            # Use http.client to avoid urllib's SSL context creation issues
            import http.client
            import json as json_mod
            from urllib.parse import urlparse
            
            parsed = urlparse(self.steel_base_url)
            
            def _http_request(method: str, path: str, body: str = None) -> dict:
                """Make HTTP request to Steel server."""
                conn = http.client.HTTPConnection(parsed.netloc, timeout=10)
                headers = {"Accept": "application/json"}
                if body:
                    headers["Content-Type"] = "application/json"
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read().decode()
                conn.close()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {data[:200]}")
                return json_mod.loads(data) if data else {}
            
            # Try to find an existing session with matching profile
            existing_session_id = None
            try:
                data = _http_request("GET", "/v1/sessions")
                for sess in data.get("sessions", []):
                    sess_profile = sess.get("profileId")
                    sess_status = sess.get("status")
                    # Reuse any live session (or match profile if specified)
                    if sess_status == "live":
                        if not profile_id or sess_profile == profile_id:
                            existing_session_id = sess["id"]
                            if sess_profile:
                                self.profile_id = sess_profile
                            break
            except Exception:
                pass
            
            if existing_session_id:
                # Reuse existing session
                self.session_id = existing_session_id
            else:
                # Create new session via HTTP
                session_data = {
                    "use_proxy": self.use_proxy,
                    "solve_captcha": self.solve_captcha,
                    "timeout": self.session_timeout * 1000,
                }
                if profile_id:
                    session_data["profile_id"] = profile_id
                if persist_profile:
                    session_data["persist_profile"] = True
                
                try:
                    data = _http_request("POST", "/v1/sessions", json_mod.dumps(session_data))
                    self.session_id = data["id"]
                    if data.get("profileId"):
                        self.profile_id = data["profileId"]
                except Exception as e:
                    raise RuntimeError(f"Failed to create Steel session: {e}")
            
            # Build connection URL
            if self.steel_api_key:
                ws_url = f"wss://connect.steel.dev?apiKey={self.steel_api_key}&sessionId={self.session_id}"
            else:
                host = self.steel_base_url.replace("http://", "").replace("https://", "")
                ws_url = f"ws://{host}/v1/cdp/{self.session_id}"
            
            try:
                self.browser = self.playwright.chromium.connect_over_cdp(ws_url, timeout=15000)  # 15s timeout
            except Exception as e:
                raise RuntimeError(f"Failed to connect to browser via CDP ({ws_url}): {e}")
            
            self.context = self.browser.contexts[0] if self.browser.contexts else self.browser.new_context()
            # Reuse existing page - prefer one with content over about:blank
            self.page = None
            if self.context.pages:
                for page in self.context.pages:
                    if page.url and page.url != "about:blank":
                        self.page = page
                        break
                # Fall back to last page if no content pages
                if self.page is None:
                    self.page = self.context.pages[-1]
            else:
                self.page = self.context.new_page()
        
        return self
    
    def stop(self):
        """Stop the browser session and release resources."""
        # Save storage state for local mode
        if self.local_mode and self.persist_profile and self.profile_name and self.context:
            try:
                storage_path = _get_storage_path(self.profile_name)
                self.context.storage_state(path=str(storage_path))
            except Exception:
                pass
        
        if self.context and self.local_mode:
            try:
                self.context.close()
            except Exception:
                pass
            self.context = None
        
        if self.browser:
            try:
                self.browser.close()
            except Exception:
                pass
            self.browser = None
        
        # Don't stop Playwright - it's a shared singleton
        self.playwright = None
        
        if self.steel_client and self.session_id and not self.local_mode:
            try:
                self.steel_client.sessions.release(self.session_id)
            except Exception:
                pass
            self.session_id = None
    
    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> dict:
        """Navigate to a URL."""
        if not self.page:
            raise RuntimeError("Browser session not started")
        
        response = self.page.goto(url, wait_until=wait_until)
        
        return {
            "status": response.status if response else None,
            "url": self.page.url,
            "title": self.page.title(),
        }
    
    def get_context(self, max_chars: int = 10000) -> dict:
        """Get page context via aria snapshot (accessibility tree).
        
        Uses Playwright's locator.aria_snapshot() to get a YAML representation
        of the accessibility tree - roles, names, attributes, and states.
        
        Args:
            max_chars: Maximum characters for aria snapshot (default 10000)
        """
        if not self.page:
            raise RuntimeError("Browser session not started")
        
        # Use the new aria_snapshot() method (replaces deprecated page.accessibility.snapshot())
        aria_snapshot = self.page.locator("body").aria_snapshot()
        
        # Truncate if too large
        truncated = False
        if len(aria_snapshot) > max_chars:
            aria_snapshot = aria_snapshot[:max_chars]
            truncated = True
        
        result = {
            "url": self.page.url,
            "title": self.page.title(),
            "aria_snapshot": aria_snapshot,
        }
        if truncated:
            result["truncated"] = True
        
        return result
    
    def click(self, selector: str = None, text: str = None) -> dict:
        """Click an element by selector or text content."""
        if not self.page:
            raise RuntimeError("Browser session not started")
        
        try:
            if text:
                self.page.get_by_text(text).click()
            elif selector:
                self.page.click(selector)
            else:
                return {"success": False, "error": "Must provide selector or text"}
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def fill(self, selector: str = None, label: str = None, value: str = "") -> dict:
        """Fill a text input field."""
        if not self.page:
            raise RuntimeError("Browser session not started")
        
        try:
            if label:
                self.page.get_by_label(label).fill(value)
            elif selector:
                self.page.fill(selector, value)
            else:
                return {"success": False, "error": "Must provide selector or label"}
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_text(self, selector: str = None) -> dict:
        """Extract text content from the page or a specific element."""
        if not self.page:
            raise RuntimeError("Browser session not started")
        
        try:
            if selector:
                element = self.page.query_selector(selector)
                if element:
                    text = element.inner_text()
                else:
                    return {"success": False, "error": f"Element not found: {selector}"}
            else:
                # Extract main content (body text)
                text = self.page.inner_text("body")
            
            # Truncate if too long
            max_length = 10000
            if len(text) > max_length:
                text = text[:max_length] + f"\n\n[Truncated, {len(text)} total characters]"
            
            return {"success": True, "text": text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def screenshot(self, full_page: bool = False) -> dict:
        """Take a screenshot of the current page."""
        if not self.page:
            raise RuntimeError("Browser session not started")
        
        try:
            import base64
            screenshot_bytes = self.page.screenshot(full_page=full_page)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            
            return {
                "success": True,
                "screenshot_base64": screenshot_b64,
                "format": "png",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def wait_for(self, selector: str = None, text: str = None, timeout: int = 30000) -> dict:
        """Wait for an element to appear."""
        if not self.page:
            raise RuntimeError("Browser session not started")
        
        try:
            if text:
                self.page.get_by_text(text).wait_for(timeout=timeout)
            elif selector:
                self.page.wait_for_selector(selector, timeout=timeout)
            else:
                return {"success": False, "error": "Must provide selector or text"}
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global session for persistent browser state
import threading
_session_lock = threading.Lock()
_global_session: Optional[BrowserSession] = None
_current_profile_name: Optional[str] = None

# Default profile name - always use this for persistence
DEFAULT_PROFILE = os.getenv("LETHE_BROWSER_PROFILE", "lethe")


def _get_or_create_session(
    profile_name: Optional[str] = None,
    persist_profile: bool = True,  # Default to True for persistence
) -> BrowserSession:
    """Get the global browser session, creating one if needed.
    
    Always uses a persistent profile for auth state preservation.
    Thread-safe via _session_lock.
    
    Args:
        profile_name: Named profile to use (defaults to DEFAULT_PROFILE)
        persist_profile: Whether to save profile state when session ends
    """
    global _global_session, _current_profile_name
    
    # Always use default profile if none specified
    if not profile_name:
        profile_name = DEFAULT_PROFILE
    
    with _session_lock:
        # If requesting a different profile, close current session
        if profile_name != _current_profile_name:
            if _global_session:
                _global_session.stop()
                _global_session = None
        
        if _global_session is None or _global_session.page is None:
            # Look up profile ID if profile name provided
            profile_id = _get_profile_id(profile_name)
            _current_profile_name = profile_name
            
            _global_session = BrowserSession()
            _global_session.start(
                profile_name=profile_name,
                profile_id=profile_id,
                persist_profile=True,  # Always persist
            )
            
            # Save the new profile ID if one was created
            if _global_session.profile_id and not profile_id:
                _save_profile(profile_name, _global_session.profile_id)
        
        return _global_session


def _is_tool(func):
    """Decorator to mark a function as a tool."""
    func._is_tool = True
    return func


# =============================================================================
# Tool Functions (exposed to the agent)
# =============================================================================

@_is_tool
def browser_navigate(url: str, wait_until: str = "domcontentloaded") -> str:
    """Navigate the browser to a URL.
    
    Args:
        url: The URL to navigate to (must include protocol, e.g., https://)
        wait_until: When to consider navigation complete - "domcontentloaded" (default, fast), 
                   "load" (all resources), or "networkidle" (no network activity)
    
    Returns:
        JSON with status code, final URL, and page title
    """
    session = _get_or_create_session()
    result = session.navigate(url, wait_until)
    return json.dumps(result, indent=2)


@_is_tool
def browser_get_context(max_chars: int = 10000) -> str:
    """Get page context via aria snapshot (YAML accessibility tree).
    
    Returns what's actually VISIBLE - buttons, links, inputs, headings, etc.
    in a hierarchical YAML format showing roles, names, and states.
    
    Use this to understand what actions are available on the page.
    
    Args:
        max_chars: Maximum characters for aria snapshot (default 10000)
    
    Returns:
        JSON with URL, title, and aria_snapshot (YAML accessibility tree)
    """
    session = _get_or_create_session()
    result = session.get_context(max_chars)
    return json.dumps(result, indent=2)


@_is_tool
def browser_get_text(max_length: int = 15000) -> str:
    """Get all visible text content from the page via aria snapshot.
    
    Returns the aria snapshot (YAML accessibility tree) which contains
    all visible text organized by semantic role.
    
    Args:
        max_length: Maximum characters to return (default 15000)
    
    Returns:
        JSON with URL, title, and aria snapshot text
    """
    session = _get_or_create_session()
    
    if not session.page:
        raise RuntimeError("Browser session not started")
    
    # Get aria snapshot - contains all visible text with semantic structure
    full_text = session.page.locator("body").aria_snapshot()
    
    # Truncate if needed
    truncated = False
    if len(full_text) > max_length:
        full_text = full_text[:max_length]
        truncated = True
    
    result = {
        "url": session.page.url,
        "title": session.page.title(),
        "text": full_text,
        "length": len(full_text),
    }
    if truncated:
        result["truncated"] = True
    
    return json.dumps(result, indent=2)


@_is_tool
def browser_click(selector: str = "", text: str = "") -> str:
    """Click an element on the page.
    
    Args:
        selector: CSS selector (e.g., "button.submit", "#login-btn", "[data-testid='submit']")
        text: Text content to find and click (alternative to selector)
    
    Returns:
        JSON with success status
    """
    session = _get_or_create_session()
    result = session.click(
        selector=selector if selector else None,
        text=text if text else None,
    )
    return json.dumps(result, indent=2)


@_is_tool
def browser_fill(value: str, selector: str = "", label: str = "") -> str:
    """Fill a text input field.
    
    Args:
        value: Text to enter in the field
        selector: CSS selector for the input (e.g., "#email", "input[name='username']")
        label: Label text to find the input (alternative to selector, e.g., "Email address")
    
    Returns:
        JSON with success status
    """
    session = _get_or_create_session()
    result = session.fill(
        selector=selector if selector else None,
        label=label if label else None,
        value=value,
    )
    return json.dumps(result, indent=2)


@_is_tool
def browser_extract_text(selector: str = "") -> str:
    """Extract text content from the page or a specific element.
    
    Args:
        selector: Optional CSS selector. If empty, extracts all visible text from body.
    
    Returns:
        JSON with extracted text (truncated to 10k chars if longer)
    """
    session = _get_or_create_session()
    result = session.extract_text(selector if selector else None)
    return json.dumps(result, indent=2)


@_is_tool
def browser_screenshot(full_page: bool = False) -> str:
    """Take a screenshot of the current page.
    
    Args:
        full_page: If True, capture the entire scrollable page. If False (default), 
                  capture only the visible viewport.
    
    Returns:
        JSON with base64-encoded PNG screenshot
    """
    session = _get_or_create_session()
    result = session.screenshot(full_page=full_page)
    return json.dumps(result, indent=2)


@_is_tool
def browser_wait_for(selector: str = "", text: str = "", timeout_seconds: int = 30) -> str:
    """Wait for an element to appear on the page.
    
    Args:
        selector: CSS selector to wait for
        text: Text content to wait for (alternative to selector)
        timeout_seconds: Maximum time to wait (default 30)
    
    Returns:
        JSON with success status
    """
    session = _get_or_create_session()
    result = session.wait_for(
        selector=selector if selector else None,
        text=text if text else None,
        timeout=timeout_seconds * 1000,
    )
    return json.dumps(result, indent=2)


@_is_tool
def browser_scroll(direction: str = "down", amount: int = 500) -> str:
    """Scroll the page.
    
    Args:
        direction: "down", "up", "top", or "bottom"
        amount: Pixels to scroll (for up/down). Default 500.
    
    Returns:
        JSON with success status and new scroll position
    """
    session = _get_or_create_session()
    
    if not session.page:
        return json.dumps({"success": False, "error": "Browser session not started"})
    
    try:
        if direction == "top":
            session.page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            session.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif direction == "up":
            session.page.evaluate(f"window.scrollBy(0, -{amount})")
        else:  # down
            session.page.evaluate(f"window.scrollBy(0, {amount})")
        
        # Get new scroll position
        scroll_y = session.page.evaluate("window.scrollY")
        scroll_height = session.page.evaluate("document.body.scrollHeight")
        viewport_height = session.page.evaluate("window.innerHeight")
        
        return json.dumps({
            "success": True,
            "scroll_y": scroll_y,
            "scroll_height": scroll_height,
            "viewport_height": viewport_height,
            "at_bottom": scroll_y + viewport_height >= scroll_height - 10,
            "at_top": scroll_y <= 10,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@_is_tool
def browser_close() -> str:
    """Close the browser session and release resources.
    
    If using a profile, the auth state (cookies, localStorage) is saved
    automatically before closing.
    
    Call this when done with browser automation to free up the Steel session.
    A new session will be created automatically on the next browser action.
    Thread-safe via _session_lock.
    
    Returns:
        JSON with success status and profile save status
    """
    global _global_session, _current_profile_name
    
    with _session_lock:
        profile_saved = None
        if _global_session:
            # If we had a profile and persist was enabled, the profile is saved on release
            if _current_profile_name and _global_session.persist_profile:
                profile_saved = _current_profile_name
            
            _global_session.stop()
            _global_session = None
        
        _current_profile_name = None
        
        result = {"success": True, "message": "Browser session closed"}
        if profile_saved:
            result["profile_saved"] = profile_saved
        
        return json.dumps(result, indent=2)
