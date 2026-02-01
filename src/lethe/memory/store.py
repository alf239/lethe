"""Main memory store coordinating all memory subsystems."""

import os
from pathlib import Path
from typing import Optional
import lancedb
import logging

logger = logging.getLogger(__name__)

from lethe.memory.blocks import BlockManager
from lethe.memory.archival import ArchivalMemory
from lethe.memory.messages import MessageHistory


class MemoryStore:
    """Unified memory store using LanceDB.
    
    Provides:
    - blocks: Core memory (persona, human, project, etc.)
    - archival: Long-term semantic memory with hybrid search
    - messages: Conversation history
    """
    
    def __init__(self, data_dir: str = "data/memory"):
        """Initialize memory store.
        
        Args:
            data_dir: Directory for storing memory data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.data_dir / "lancedb"))
        logger.info(f"Connected to LanceDB at {self.data_dir / 'lancedb'}")
        
        # Initialize subsystems
        self.blocks = BlockManager(self.db)
        self.archival = ArchivalMemory(self.db)
        self.messages = MessageHistory(self.db)
        
        logger.info("Memory store initialized")
    
    def get_context_for_prompt(self, max_tokens: int = 8000) -> str:
        """Get formatted memory context for LLM prompt.
        
        Args:
            max_tokens: Approximate max tokens for context
            
        Returns:
            Formatted string with all memory blocks
        """
        sections = []
        
        # Add memory blocks
        blocks = self.blocks.list_blocks()
        for block in blocks:
            if block.get("hidden"):
                continue
            label = block["label"]
            value = block["value"]
            description = block.get("description", "")
            
            section = f"<{label}>\n"
            if description:
                section += f"<description>{description}</description>\n"
            section += f"{value}\n</{label}>"
            sections.append(section)
        
        return "\n\n".join(sections)
    
    def search(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "hybrid"
    ) -> list[dict]:
        """Search across archival memory.
        
        Args:
            query: Search query
            limit: Max results
            search_type: "hybrid", "vector", or "fts"
            
        Returns:
            List of matching passages
        """
        return self.archival.search(query, limit=limit, search_type=search_type)
    
    def add_memory(self, text: str, metadata: Optional[dict] = None) -> str:
        """Add a memory to archival storage.
        
        Args:
            text: Memory text
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        return self.archival.add(text, metadata=metadata)
    
    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        """Get recent conversation messages.
        
        Args:
            limit: Max messages to return
            
        Returns:
            List of messages
        """
        return self.messages.get_recent(limit=limit)
    
    def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> str:
        """Add a message to history.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        return self.messages.add(role, content, metadata=metadata)
