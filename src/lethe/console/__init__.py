"""Lethe Console - Mind State Visualization.

A web-based dashboard showing the agent's current context assembly:
- Chat messages
- Memory blocks
- System prompt
- What's actually sent to the LLM
"""

import asyncio
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConsoleState:
    """Shared state for the console UI."""
    
    # Memory blocks (label -> block data)
    memory_blocks: Dict[str, Dict] = field(default_factory=dict)
    
    # Identity/system prompt
    identity: str = ""
    
    # Conversation summary
    summary: str = ""
    
    # Recent messages (role, content, timestamp)
    messages: List[Dict] = field(default_factory=list)
    
    # Last built context (what was sent to LLM)
    last_context: List[Dict] = field(default_factory=list)
    last_context_tokens: int = 0
    last_context_time: Optional[datetime] = None
    
    # Agent status
    status: str = "idle"  # idle, thinking, tool_call
    current_tool: Optional[str] = None
    
    # Stats
    total_messages: int = 0
    archival_count: int = 0
    
    # Model info
    model: str = ""
    model_aux: str = ""
    
    # Token tracking
    tokens_today: int = 0
    api_calls_today: int = 0
    
    # Cache stats (from API response usage.prompt_tokens_details)
    cache_read_tokens: int = 0       # Total cached tokens read today
    cache_write_tokens: int = 0      # Total cached tokens written today
    last_cache_read: int = 0         # Cached tokens read in last request
    last_cache_write: int = 0        # Cached tokens written in last request
    last_prompt_tokens: int = 0      # Total prompt tokens in last request
    
    # Change tracking (incremented on data changes that need UI rebuild)
    version: int = 0


# Global state instance
_state = ConsoleState()


def get_state() -> ConsoleState:
    """Get the global console state."""
    return _state


def update_memory_blocks(blocks: List[Dict]):
    """Update memory blocks in console state."""
    _state.memory_blocks = {b["label"]: b for b in blocks}


def update_identity(identity: str):
    """Update identity/system prompt."""
    _state.identity = identity


def update_summary(summary: str):
    """Update conversation summary."""
    _state.summary = summary


def update_messages(messages):
    """Update recent messages.
    
    Args:
        messages: List of Message objects or dicts
    """
    result = []
    for msg in messages:
        if hasattr(msg, 'role'):
            # Message object
            timestamp = None
            if hasattr(msg, 'created_at') and msg.created_at:
                timestamp = msg.created_at.strftime("%H:%M:%S") if hasattr(msg.created_at, 'strftime') else str(msg.created_at)[:19]
            result.append({
                "role": msg.role,
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                "timestamp": timestamp,
            })
        elif isinstance(msg, dict):
            result.append(msg)
    _state.messages = result


def update_context(context: List[Dict], tokens: int):
    """Update last built context."""
    _state.last_context = context
    _state.last_context_tokens = tokens
    _state.last_context_time = datetime.now()
    _state.version += 1


def update_status(status: str, tool: Optional[str] = None):
    """Update agent status."""
    _state.status = status
    _state.current_tool = tool


def update_stats(total_messages: int, archival_count: int):
    """Update stats."""
    _state.total_messages = total_messages
    _state.archival_count = archival_count


def update_model_info(model: str, model_aux: str = ""):
    """Update model info."""
    _state.model = model
    _state.model_aux = model_aux


def track_tokens(tokens: int):
    """Track tokens consumed."""
    _state.tokens_today += tokens
    _state.api_calls_today += 1


def track_cache_usage(usage: dict):
    """Track cache usage from API response.
    
    Works with all providers via OpenRouter's unified format:
    - Anthropic: cache_creation_input_tokens, cache_read_input_tokens
    - OpenRouter unified: prompt_tokens_details.cached_tokens, cache_write_tokens
    - Moonshot/Kimi: automatic caching, same unified format
    """
    # OpenRouter unified format
    details = usage.get("prompt_tokens_details", {})
    if details:
        cached = details.get("cached_tokens", 0)
        written = details.get("cache_write_tokens", 0)
        if cached:
            _state.last_cache_read = cached
            _state.cache_read_tokens += cached
        if written:
            _state.last_cache_write = written
            _state.cache_write_tokens += written
    
    # Anthropic direct format (via litellm)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    if cache_read:
        _state.last_cache_read = cache_read
        _state.cache_read_tokens += cache_read
    if cache_write:
        _state.last_cache_write = cache_write
        _state.cache_write_tokens += cache_write
    
    _state.last_prompt_tokens = usage.get("prompt_tokens", 0)
