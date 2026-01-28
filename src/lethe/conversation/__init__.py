"""Conversation manager for interruptible async processing.

Handles per-chat conversation state with support for:
- Interrupting current processing when new messages arrive
- Accumulating messages during processing
- Resuming with combined context after interrupt
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PendingMessage:
    """A message waiting to be processed."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConversationState:
    """State for a single chat conversation."""
    chat_id: int
    user_id: int
    pending_messages: list[PendingMessage] = field(default_factory=list)
    is_processing: bool = False
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    current_task: Optional[asyncio.Task] = None
    
    def add_message(self, content: str, metadata: Optional[dict] = None) -> bool:
        """Add a message to pending. Returns True if this interrupts current processing."""
        self.pending_messages.append(PendingMessage(
            content=content,
            metadata=metadata or {},
        ))
        
        if self.is_processing:
            self.interrupt_event.set()
            logger.info(f"Chat {self.chat_id}: Interrupt signaled (new message while processing)")
            return True
        return False
    
    def get_combined_message(self) -> tuple[str, dict]:
        """Get all pending messages combined into one, clearing the pending list.
        
        Returns:
            Tuple of (combined_content, merged_metadata)
        """
        if not self.pending_messages:
            return "", {}
        
        if len(self.pending_messages) == 1:
            msg = self.pending_messages.pop(0)
            return msg.content, msg.metadata
        
        # Multiple messages - combine them
        contents = []
        merged_metadata = {}
        
        for msg in self.pending_messages:
            contents.append(msg.content)
            # Merge metadata, later messages override earlier
            merged_metadata.update(msg.metadata)
        
        self.pending_messages.clear()
        
        # Format combined messages
        combined = "\n\n---\n[Additional message while processing:]\n".join(contents)
        return combined, merged_metadata
    
    def check_interrupt(self) -> bool:
        """Check if interrupt was requested. Clears the event."""
        if self.interrupt_event.is_set():
            self.interrupt_event.clear()
            return True
        return False


class ConversationManager:
    """Manages conversation state across multiple chats."""
    
    def __init__(self):
        self._states: dict[int, ConversationState] = {}
        self._lock = asyncio.Lock()
    
    def get_or_create_state(self, chat_id: int, user_id: int) -> ConversationState:
        """Get or create conversation state for a chat."""
        if chat_id not in self._states:
            self._states[chat_id] = ConversationState(chat_id=chat_id, user_id=user_id)
        return self._states[chat_id]
    
    async def add_message(
        self,
        chat_id: int,
        user_id: int,
        content: str,
        metadata: Optional[dict] = None,
        process_callback: Optional[Callable] = None,
    ) -> bool:
        """Add a message and start/interrupt processing.
        
        Args:
            chat_id: Telegram chat ID
            user_id: Telegram user ID
            content: Message content
            metadata: Optional metadata (username, attachments, etc.)
            process_callback: Async function to call for processing
                             Signature: async def callback(state: ConversationState) -> None
        
        Returns:
            True if message was added (always True currently)
        """
        async with self._lock:
            state = self.get_or_create_state(chat_id, user_id)
            was_interrupted = state.add_message(content, metadata)
            
            if was_interrupted:
                logger.info(f"Chat {chat_id}: Message queued for interrupt")
                # Processing task will pick up the new message
                return True
            
            if state.is_processing:
                # Already processing, message queued
                logger.info(f"Chat {chat_id}: Message queued (already processing)")
                return True
            
            # Start processing
            if process_callback:
                state.is_processing = True
                state.current_task = asyncio.create_task(
                    self._process_loop(state, process_callback)
                )
            
            return True
    
    async def _process_loop(
        self,
        state: ConversationState,
        process_callback: Callable,
    ):
        """Main processing loop for a conversation.
        
        Continues until no more pending messages.
        Handles interrupts by restarting with combined messages.
        """
        try:
            while state.pending_messages:
                # Clear interrupt flag
                state.interrupt_event.clear()
                
                # Get combined message
                combined, metadata = state.get_combined_message()
                
                if not combined:
                    break
                
                logger.info(f"Chat {state.chat_id}: Processing message ({len(combined)} chars)")
                
                try:
                    # Process the message
                    await process_callback(
                        chat_id=state.chat_id,
                        user_id=state.user_id,
                        message=combined,
                        metadata=metadata,
                        interrupt_check=state.interrupt_event.is_set,
                    )
                except asyncio.CancelledError:
                    logger.info(f"Chat {state.chat_id}: Processing cancelled")
                    raise
                except Exception as e:
                    logger.exception(f"Chat {state.chat_id}: Processing error: {e}")
                    # Continue to process remaining messages
                
                # If interrupted, there will be new messages - loop continues
                if state.interrupt_event.is_set():
                    logger.info(f"Chat {state.chat_id}: Interrupted, will process new messages")
                    state.interrupt_event.clear()
        finally:
            state.is_processing = False
            state.current_task = None
            logger.info(f"Chat {state.chat_id}: Processing loop finished")
    
    def is_processing(self, chat_id: int) -> bool:
        """Check if a chat is currently being processed."""
        state = self._states.get(chat_id)
        return state.is_processing if state else False
    
    def get_pending_count(self, chat_id: int) -> int:
        """Get number of pending messages for a chat."""
        state = self._states.get(chat_id)
        return len(state.pending_messages) if state else 0
    
    async def cancel(self, chat_id: int) -> bool:
        """Cancel processing for a chat.
        
        Returns True if there was something to cancel.
        """
        state = self._states.get(chat_id)
        if state and state.current_task and not state.current_task.done():
            state.current_task.cancel()
            try:
                await state.current_task
            except asyncio.CancelledError:
                pass
            state.pending_messages.clear()
            state.is_processing = False
            return True
        return False
