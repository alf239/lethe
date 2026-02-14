"""OAuth login flow for Anthropic Claude Max/Pro subscription.

Uses PKCE (Proof Key for Code Exchange) OAuth flow:
1. Generate code verifier + challenge
2. Open browser to Anthropic's OAuth authorize URL
3. User authenticates and gets a code
4. Exchange code for access + refresh tokens
5. Save tokens to ~/.lethe/oauth_tokens.json

Usage:
    uv run lethe oauth-login
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
import webbrowser
from urllib.parse import urlencode

import httpx

from lethe.memory.anthropic_oauth import CLIENT_ID, TOKEN_FILE

# OAuth endpoints
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code"


def _generate_pkce() -> tuple:
    """Generate PKCE code verifier and challenge.
    
    Returns:
        (verifier, challenge) tuple
    """
    # 43-128 chars of URL-safe base64
    verifier = secrets.token_urlsafe(32)
    
    # S256 challenge = base64url(sha256(verifier))
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    
    return verifier, challenge


def _build_authorize_url(verifier: str, challenge: str) -> str:
    """Build the OAuth authorization URL."""
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def _exchange_code(code: str, verifier: str) -> dict:
    """Exchange authorization code for tokens.
    
    Args:
        code: The authorization code (may contain #state suffix)
        verifier: The PKCE code verifier
        
    Returns:
        Token response dict with access_token, refresh_token, expires_in
    """
    # Parse code#state format
    if "#" in code:
        auth_code, state = code.split("#", 1)
    else:
        auth_code = code
        state = None
    
    body = {
        "code": auth_code,
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    }
    if state:
        body["state"] = state
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(TOKEN_URL, json=body)
    
    if response.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {response.status_code} {response.text}")
    
    return response.json()


def run_oauth_login():
    """Run the interactive OAuth login flow."""
    print("\n🔐 Anthropic OAuth Login (Claude Max/Pro Subscription)\n")
    print("This will open your browser to sign in with your Anthropic account.")
    print("After signing in, you'll get a code to paste back here.\n")
    
    # Generate PKCE
    verifier, challenge = _generate_pkce()
    url = _build_authorize_url(verifier, challenge)
    
    print(f"Opening browser to:\n{url}\n")
    
    # Try to open browser
    try:
        webbrowser.open(url)
    except Exception:
        print("(Could not open browser automatically — please open the URL above manually)")
    
    print("After authenticating, paste the authorization code below.")
    print("The code may look like: abc123#xyz789\n")
    
    code = input("Authorization code: ").strip()
    
    if not code:
        print("❌ No code provided. Aborting.")
        sys.exit(1)
    
    print("\nExchanging code for tokens...")
    
    try:
        result = asyncio.run(_exchange_code(code, verifier))
    except Exception as e:
        print(f"❌ Token exchange failed: {e}")
        sys.exit(1)
    
    access_token = result.get("access_token")
    refresh_token = result.get("refresh_token")
    expires_in = result.get("expires_in", 3600)
    
    if not access_token:
        print(f"❌ No access token in response: {result}")
        sys.exit(1)
    
    # Save tokens
    import time
    token_data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in,
    }
    
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(token_data, indent=2))
    TOKEN_FILE.chmod(0o600)
    
    print(f"\n✅ OAuth tokens saved to {TOKEN_FILE}")
    print(f"   Access token: {access_token[:20]}...")
    print(f"   Refresh token: {'yes' if refresh_token else 'no'}")
    print(f"   Expires in: {expires_in}s")
    print(f"\nYou can now start Lethe with OAuth authentication.")
    print(f"Make sure LLM_PROVIDER=anthropic in your .env (no ANTHROPIC_API_KEY needed).")
