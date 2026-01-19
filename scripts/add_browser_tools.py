#!/usr/bin/env python3
"""Add browser tools to existing Lethe agent without losing memory."""
import asyncio
import os
import sys

# Change to project root so .env is found
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "src")


async def add_browser_tools():
    from letta_client import AsyncLetta
    from lethe.config import get_settings
    
    # Use existing settings (loads .env automatically via pydantic-settings)
    settings = get_settings()
    
    print(f"Connecting to: {settings.letta_base_url}")
    
    if settings.letta_api_key:
        client = AsyncLetta(base_url=settings.letta_base_url, api_key=settings.letta_api_key)
    else:
        client = AsyncLetta(base_url=settings.letta_base_url)
    
    # Find existing agent
    agent_name = settings.lethe_agent_name
    agents = []
    agents_iter = await client.agents.list(name=agent_name)
    if hasattr(agents_iter, '__aiter__'):
        async for agent in agents_iter:
            agents.append(agent)
    else:
        agents = list(agents_iter)
    
    if not agents:
        print(f"❌ Agent '{agent_name}' not found")
        return
    
    agent = agents[0]
    print(f"Found agent: {agent.name} ({agent.id})")
    print(f"Current tools: {len(agent.tools)}")
    
    # Define browser tool stubs (sessions auto-create on first use)
    browser_stubs = []
    
    def browser_navigate(url: str, wait_until: str = "domcontentloaded") -> str:
        """Navigate to URL.
        
        Args:
            url: URL to navigate to
            wait_until: Wait condition (domcontentloaded, load, networkidle)
        
        Returns:
            JSON with status, URL, title
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_navigate)
    
    def browser_get_context(max_elements: int = 100) -> str:
        """Get interactive elements via accessibility tree.
        
        Args:
            max_elements: Max elements to return
        
        Returns:
            JSON with elements (role, name, state)
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_get_context)
    
    def browser_get_text(max_length: int = 15000) -> str:
        """Get visible text content from page.
        
        Args:
            max_length: Max characters
        
        Returns:
            JSON with text content
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_get_text)
    
    def browser_click(selector: str = "", text: str = "") -> str:
        """Click an element.
        
        Args:
            selector: CSS selector
            text: Text to find and click
        
        Returns:
            JSON with success status
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_click)
    
    def browser_fill(value: str, selector: str = "", label: str = "") -> str:
        """Fill input field.
        
        Args:
            value: Text to enter
            selector: CSS selector
            label: Label text to find input
        
        Returns:
            JSON with success status
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_fill)
    
    def browser_wait_for(selector: str = "", text: str = "", timeout_seconds: int = 30) -> str:
        """Wait for element to appear.
        
        Args:
            selector: CSS selector
            text: Text to wait for
            timeout_seconds: Timeout
        
        Returns:
            JSON with success status
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_wait_for)
    
    def browser_scroll(direction: str = "down", amount: int = 500) -> str:
        """Scroll the page.
        
        Args:
            direction: down, up, top, bottom
            amount: Pixels for up/down
        
        Returns:
            JSON with scroll position
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_scroll)
    
    def browser_screenshot(full_page: bool = False) -> str:
        """Take screenshot.
        
        Args:
            full_page: Capture full page
        
        Returns:
            JSON with base64 PNG
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_screenshot)
    
    def browser_extract_text(selector: str = "") -> str:
        """Extract text from element.
        
        Args:
            selector: CSS selector (empty for body)
        
        Returns:
            JSON with text
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_extract_text)
    
    def browser_close() -> str:
        """Close browser and save profile.
        
        Returns:
            JSON with success status
        """
        raise Exception("Client-side")
    browser_stubs.append(browser_close)
    
    # Register tools with Letta
    print("\nRegistering browser tools...")
    new_tools = {}  # name -> id
    for func in browser_stubs:
        try:
            tool = await client.tools.upsert_from_function(
                func=func,
                default_requires_approval=True,
            )
            new_tools[tool.name] = tool.id
            print(f"  ✓ {tool.name} ({tool.id[:8]}...)")
        except Exception as e:
            print(f"  ✗ {func.__name__}: {e}")
    
    # Get current tool names
    current_tool_names = [t.name for t in agent.tools]
    
    # Attach new tools that aren't already attached
    tools_to_add = {name: tid for name, tid in new_tools.items() if name not in current_tool_names}
    
    if tools_to_add:
        print(f"\nAttaching {len(tools_to_add)} new tools to agent...")
        for tool_name, tool_id in tools_to_add.items():
            try:
                await client.agents.tools.attach(
                    tool_id=tool_id,
                    agent_id=agent.id,
                )
                print(f"  ✓ {tool_name}")
            except Exception as e:
                print(f"  ✗ {tool_name}: {e}")
        print(f"✅ Done")
    else:
        print("\n✅ All browser tools already attached")
    
    # Show final tool list
    all_tools = sorted(set(current_tool_names) | set(tools_to_add.keys()))
    print(f"\nAgent tools ({len(all_tools)}):")
    for t in all_tools:
        marker = "🌐" if t.startswith("browser_") else "  "
        print(f"  {marker} {t}")


if __name__ == "__main__":
    asyncio.run(add_browser_tools())
