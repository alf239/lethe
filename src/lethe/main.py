"""Main entry point for Lethe."""

import asyncio
import logging
import signal
import sys


from rich.console import Console
from rich.logging import RichHandler

from lethe.agent import AgentManager
from lethe.config import get_settings
from lethe.queue import TaskQueue
from lethe.tasks import TaskManager
from lethe.tasks.worker import TaskWorker
from lethe.telegram import TelegramBot
from lethe.worker import HeartbeatWorker, Worker

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


async def run():
    """Run the Lethe application."""
    logger = logging.getLogger(__name__)

    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        console.print("\nMake sure you have a .env file with TELEGRAM_BOT_TOKEN set.")
        sys.exit(1)

    console.print("[bold blue]Lethe[/bold blue] - Autonomous Executive Assistant")
    console.print(f"Letta server: {settings.letta_base_url}")
    console.print(f"Agent name: {settings.lethe_agent_name}")
    console.print()

    # Initialize components
    task_queue = TaskQueue(settings.db_path)
    await task_queue.initialize()
    logger.info("Task queue initialized")
    
    # Initialize background task manager
    task_db_path = settings.db_path.parent / "tasks.db"
    bg_task_manager = TaskManager(task_db_path)
    await bg_task_manager.initialize()
    logger.info("Background task manager initialized")

    agent_manager = AgentManager(settings)
    
    # Initialize agent to get ID (needed for callbacks)
    agent_id = await agent_manager.get_or_create_agent()
    
    # Callback when user stops tasks via /stop command
    async def on_task_stopped(task_ids: list[str], chat_id: int):
        """Notify agent that tasks were stopped by user."""
        tasks_info = []
        for tid in task_ids:
            task = await bg_task_manager.get_task(tid)
            if task:
                tasks_info.append(f"- {task.description[:50]}...")
        
        if tasks_info:
            # Send a message to the agent informing about cancelled tasks
            msg = f"[SYSTEM] User cancelled {len(task_ids)} background task(s):\n" + "\n".join(tasks_info)
            try:
                await agent_manager.send_message(
                    message=msg,
                    on_message=lambda content: None,  # Agent response goes to user via normal channel
                )
            except Exception as e:
                logger.warning(f"Failed to notify agent about stopped tasks: {e}")
    
    telegram_bot = TelegramBot(
        settings, 
        task_queue,
        bg_task_manager=bg_task_manager,
        on_task_stopped=on_task_stopped,
    )

    # Create message worker (passes task_manager for spawn_task tool)
    worker = Worker(task_queue, agent_manager, telegram_bot, bg_task_manager, settings)
    
    # Wire up worker reference for /stop command
    telegram_bot.worker = worker

    # Create heartbeat worker if we have a primary user
    heartbeat = None
    primary_user_id = None
    if settings.allowed_user_ids:
        primary_user_id = settings.allowed_user_ids[0]
        heartbeat = HeartbeatWorker(
            agent_manager=agent_manager,
            telegram_bot=telegram_bot,
            chat_id=primary_user_id,
            interval_minutes=15,
            identity_refresh_hours=2,
            enabled=True,
        )
        logger.info(f"Heartbeat enabled for user {primary_user_id} (every 15 min, identity refresh every 2h)")
    else:
        logger.info("Heartbeat disabled (no allowed_user_ids configured)")
    
    # Callback to notify user when a background task completes
    async def on_task_complete(task):
        if primary_user_id:
            try:
                status_emoji = "✅" if task.status.value == "completed" else "❌"
                msg = f"{status_emoji} Background task completed: {task.description[:50]}..."
                if task.result:
                    # Truncate very long results but give more context (2000 chars)
                    result_preview = task.result[:2000]
                    if len(task.result) > 2000:
                        result_preview += f"\n\n... (truncated, {len(task.result)} chars total)"
                    msg += f"\n\nResult:\n{result_preview}"
                if task.error:
                    msg += f"\n\nError: {task.error[:500]}"
                await telegram_bot.send_message(primary_user_id, msg)
                logger.info(f"Sent task completion notification for {task.id}")
            except Exception as e:
                logger.error(f"Failed to send task completion notification: {e}")
    
    bg_task_worker = TaskWorker(
        task_manager=bg_task_manager,
        letta_client=agent_manager.client,
        main_agent_id=agent_id,
        tool_handlers=agent_manager._tool_handlers,
        on_task_complete=on_task_complete,
    )

    # Set up shutdown handling
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start all components
    console.print("[green]Starting services...[/green]")

    # Create tasks
    worker_task = asyncio.create_task(worker.start())
    bot_task = asyncio.create_task(telegram_bot.start())
    heartbeat_task = asyncio.create_task(heartbeat.start()) if heartbeat else None
    bg_task_worker_task = asyncio.create_task(bg_task_worker.start())

    try:
        # Wait for shutdown signal
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        # Cleanup
        console.print("\n[yellow]Shutting down...[/yellow]")
        
        # Stop components
        await worker.stop()
        await telegram_bot.stop()
        if heartbeat:
            await heartbeat.stop()
        await bg_task_worker.stop()
        
        # Cancel tasks
        worker_task.cancel()
        bot_task.cancel()
        if heartbeat_task:
            heartbeat_task.cancel()
        bg_task_worker_task.cancel()
        
        # Wait for tasks to finish
        tasks_to_wait = [worker_task, bot_task, bg_task_worker_task]
        if heartbeat_task:
            tasks_to_wait.append(heartbeat_task)
        for task in tasks_to_wait:
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        await task_queue.close()
        await bg_task_manager.close()
        console.print("[green]Shutdown complete.[/green]")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Lethe - Autonomous Executive Assistant")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
