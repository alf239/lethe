"""Main entry point for Lethe."""

import asyncio
import logging
import os
import signal
import sys

# Load .env file before anything else
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.logging import RichHandler

from lethe.agent import Agent, set_oauth_callbacks
from lethe.config import get_settings
from lethe.conversation import ConversationManager
from lethe.telegram import TelegramBot
from lethe.heartbeat import Heartbeat

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )

    # Reduce noise from libraries
    logging.getLogger("aiogram").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


async def run():
    """Run the Lethe application."""
    logger = logging.getLogger(__name__)

    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        console.print("\nMake sure you have a .env file with TELEGRAM_BOT_TOKEN set.")
        console.print("Also ensure OPENROUTER_API_KEY is set in your environment.")
        sys.exit(1)

    console.print("[bold blue]Lethe[/bold blue] - Autonomous AI Assistant")
    console.print(f"Model: {settings.llm_model}")
    console.print(f"Memory: {settings.memory_dir}")
    console.print()
    
    # Get primary user ID for OAuth messages
    allowed_ids = settings.telegram_allowed_user_ids
    primary_chat_id = int(allowed_ids.split(",")[0]) if allowed_ids else None
    
    # Check if Claude Max needs OAuth setup
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    oauth_pending = False
    oauth_instance = None
    
    if provider == "claude-max":
        from lethe.oauth import ClaudeOAuth
        oauth_instance = ClaudeOAuth()
        if not oauth_instance.has_valid_tokens():
            oauth_pending = True
            # Need to authenticate - create minimal bot to send auth URL and wait
            from aiogram import Bot, Dispatcher
            from aiogram.filters import Command
            from aiogram.types import Message
            
            oauth_bot = Bot(token=settings.telegram_bot_token)
            oauth_dp = Dispatcher()
            oauth_complete = asyncio.Event()
            
            auth_url = oauth_instance.start_auth_flow()
            message = (
                "🔐 Claude Max Authentication Required\n\n"
                "1️⃣ Click this link to authenticate:\n"
                f"{auth_url}\n\n"
                "2️⃣ After logging in, you'll see a page that won't load.\n"
                "3️⃣ Copy the ENTIRE URL from your browser.\n"
                "4️⃣ Send it here with: /oauth <url>\n\n"
                "Example: /oauth http://localhost:19532/callback?code=abc&state=xyz"
            )
            
            @oauth_dp.message(Command("oauth"))
            async def handle_oauth(msg: Message):
                if msg.from_user.id != primary_chat_id:
                    return
                text = msg.text or ""
                parts = text.split(maxsplit=1)
                if len(parts) < 2:
                    await msg.answer("Usage: /oauth <redirect_url>")
                    return
                try:
                    await oauth_instance.complete_auth_flow(parts[1].strip())
                    await msg.answer("✅ Authentication successful! Starting Lethe...")
                    oauth_complete.set()
                except Exception as e:
                    await msg.answer(f"❌ Error: {e}")
            
            # Check if we're running locally (can use local HTTP server)
            # vs remotely (need Telegram-based flow)
            is_local = os.environ.get("LETHE_REMOTE", "").lower() != "true"
            
            if is_local:
                # Use local HTTP server to catch callback
                console.print("[yellow]Claude Max authentication required![/yellow]")
                console.print("[dim]Starting local server for OAuth callback...[/dim]")
                await oauth_instance._authenticate_with_local_server()
                console.print("[green]OAuth complete![/green]")
                oauth_pending = False
            elif primary_chat_id:
                # Remote mode: use Telegram for OAuth flow
                console.print("[yellow]Claude Max authentication required![/yellow]")
                await oauth_bot.send_message(primary_chat_id, message)
                console.print("[dim]Auth URL sent to Telegram. Waiting for /oauth command...[/dim]")
                
                # Run minimal bot until OAuth completes
                async def wait_for_oauth():
                    await oauth_complete.wait()
                
                polling_task = asyncio.create_task(
                    oauth_dp.start_polling(oauth_bot, handle_signals=False)
                )
                wait_task = asyncio.create_task(wait_for_oauth())
                
                # Wait for OAuth to complete
                done, pending = await asyncio.wait(
                    [polling_task, wait_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel polling
                polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    pass
                
                await oauth_bot.session.close()
                console.print("[green]OAuth complete![/green]")
                oauth_pending = False
            else:
                console.print("[yellow]Claude Max authentication required![/yellow]")
                console.print(f"\nVisit: {auth_url}\n")
                console.print("Then paste the redirect URL below:")
                redirect_url = input("Redirect URL: ").strip()
                await oauth_instance.complete_auth_flow(redirect_url)
                oauth_pending = False
        
        # Store oauth instance for agent to use
        import lethe.agent as agent_module
        agent_module._oauth_instance = oauth_instance

    # Initialize agent (tools auto-loaded)
    console.print("[dim]Initializing agent...[/dim]")
    agent = Agent(settings)
    agent.refresh_memory_context()
    
    stats = agent.get_stats()
    console.print(f"[green]Agent ready[/green] - {stats['memory_blocks']} blocks, {stats['archival_memories']} memories")

    # Initialize conversation manager
    conversation_manager = ConversationManager(debounce_seconds=settings.debounce_seconds)
    logger.info(f"Conversation manager initialized (debounce: {settings.debounce_seconds}s)")

    # Message processing callback
    async def process_message(chat_id: int, user_id: int, message: str, metadata: dict, interrupt_check):
        """Process a message from Telegram."""
        logger.info(f"Processing message from {user_id}: {message[:50]}...")
        
        # Start typing indicator
        await telegram_bot.start_typing(chat_id)
        
        try:
            # Callback for intermediate messages (reasoning/thinking)
            async def on_intermediate(content: str):
                """Send intermediate updates while agent is working."""
                if not content or len(content) < 10:
                    return
                # Check for interrupt before sending
                if interrupt_check():
                    return
                # Send thinking/reasoning as-is (no emoji prefix)
                await telegram_bot.send_message(chat_id, content)
            
            # Callback for image attachments (screenshots, etc.)
            async def on_image(image_path: str):
                """Send image to user."""
                if interrupt_check():
                    return
                await telegram_bot.send_photo(chat_id, image_path)
            
            # Get response from agent
            response = await agent.chat(message, on_message=on_intermediate, on_image=on_image)
            
            # Check for interrupt
            if interrupt_check():
                logger.info("Processing interrupted")
                return
            
            # Send response
            await telegram_bot.send_message(chat_id, response)
            
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await telegram_bot.send_message(chat_id, f"Error: {e}")
        finally:
            await telegram_bot.stop_typing(chat_id)

    # OAuth callback for Claude Max (if pending auth)
    oauth_callback = None
    if provider == "claude-max":
        from lethe.oauth import ClaudeOAuth
        # Check if we have a pending OAuth flow
        import lethe.agent as agent_module
        if hasattr(agent_module, '_oauth_instance') and agent_module._oauth_instance:
            oauth_instance = agent_module._oauth_instance
            if oauth_instance._pending_auth:  # Auth flow started but not completed
                async def complete_oauth(redirect_url: str):
                    """Complete pending OAuth flow."""
                    await oauth_instance.complete_auth_flow(redirect_url)
                oauth_callback = complete_oauth
    
    # Initialize Telegram bot
    telegram_bot = TelegramBot(
        settings,
        conversation_manager=conversation_manager,
        process_callback=process_message,
        oauth_callback=oauth_callback,
    )
    # heartbeat_callback will be set below after Heartbeat is created

    # Initialize heartbeat
    heartbeat_interval = int(os.environ.get("HEARTBEAT_INTERVAL", 15 * 60))  # Default 15 min
    heartbeat_enabled = os.environ.get("HEARTBEAT_ENABLED", "true").lower() == "true"
    
    # Get the first allowed user ID for heartbeat messages
    allowed_ids = settings.telegram_allowed_user_ids
    heartbeat_chat_id = int(allowed_ids.split(",")[0]) if allowed_ids else None
    
    async def heartbeat_process(message: str) -> str:
        """Process heartbeat through full agent (tools allowed)."""
        return await agent.chat(message, use_hippocampus=False)
    
    async def heartbeat_send(response: str):
        """Send heartbeat response to user."""
        if heartbeat_chat_id:
            await telegram_bot.send_message(heartbeat_chat_id, response)
    
    async def heartbeat_summarize(prompt: str) -> str:
        """Summarize/evaluate heartbeat response before sending."""
        return await agent.llm.complete(prompt)
    
    heartbeat = Heartbeat(
        process_callback=heartbeat_process,
        send_callback=heartbeat_send,
        summarize_callback=heartbeat_summarize,
        interval=heartbeat_interval,
        enabled=heartbeat_enabled and heartbeat_chat_id is not None,
    )
    
    # Set heartbeat trigger on telegram bot for /heartbeat command
    telegram_bot.heartbeat_callback = heartbeat.trigger

    # Set up shutdown handling
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal...")
        shutdown_event.set()
        # Force exit after 3 seconds using a thread (not event loop)
        # This ensures exit even if event loop is blocked
        import threading
        def force_exit():
            import time
            time.sleep(3)
            logger.warning("Graceful shutdown timed out, forcing exit")
            os._exit(0)
        threading.Thread(target=force_exit, daemon=True).start()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start services
    console.print("[green]Starting services...[/green]")

    bot_task = asyncio.create_task(telegram_bot.start())
    heartbeat_task = asyncio.create_task(heartbeat.start())

    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        console.print("\n[yellow]Shutting down...[/yellow]")
        
        # Shutdown with timeout to avoid hanging on native threads
        try:
            async with asyncio.timeout(5):
                await heartbeat.stop()
                await telegram_bot.stop()
                await agent.close()
        except asyncio.TimeoutError:
            logger.warning("Shutdown timed out, forcing exit")
            os._exit(0)  # Force exit - LanceDB/OpenBLAS threads don't respect Python shutdown
        
        bot_task.cancel()
        heartbeat_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        
        console.print("[green]Shutdown complete.[/green]")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Lethe - Autonomous AI Assistant")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
