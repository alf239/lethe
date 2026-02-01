"""Memory blocks - Core memory that's always in context.

Similar to Letta's memory blocks but simpler and local.
"""

import json
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
import uuid

import lancedb
import logging

logger = logging.getLogger(__name__)


# Default block schema
BLOCK_SCHEMA = {
    "id": str,
    "label": str,
    "value": str,
    "description": str,
    "limit": int,
    "read_only": bool,
    "hidden": bool,
    "created_at": str,
    "updated_at": str,
}

DEFAULT_LIMIT = 5000  # Characters


class BlockManager:
    """Manages memory blocks (core memory).
    
    Blocks are key-value pairs that are always included in the LLM context.
    Examples: persona, human, project, system
    """
    
    TABLE_NAME = "memory_blocks"
    
    def __init__(self, db: lancedb.DBConnection):
        """Initialize block manager.
        
        Args:
            db: LanceDB connection
        """
        self.db = db
        self._ensure_table()
    
    def _ensure_table(self):
        """Create table if it doesn't exist."""
        if self.TABLE_NAME not in self.db.table_names():
            # Create with empty initial data
            self.db.create_table(
                self.TABLE_NAME,
                data=[{
                    "id": "_init_",
                    "label": "_init_",
                    "value": "",
                    "description": "",
                    "limit": DEFAULT_LIMIT,
                    "read_only": False,
                    "hidden": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }]
            )
            logger.info(f"Created table {self.TABLE_NAME}")
    
    def _get_table(self):
        """Get the blocks table."""
        return self.db.open_table(self.TABLE_NAME)
    
    def create(
        self,
        label: str,
        value: str = "",
        description: str = "",
        limit: int = DEFAULT_LIMIT,
        read_only: bool = False,
        hidden: bool = False,
    ) -> str:
        """Create a new memory block.
        
        Args:
            label: Block label (e.g., "persona", "human")
            value: Initial value
            description: Block description
            limit: Character limit
            read_only: Whether block is read-only
            hidden: Whether block is hidden from context
            
        Returns:
            Block ID
        """
        # Check if block with label exists
        existing = self.get_by_label(label)
        if existing:
            raise ValueError(f"Block with label '{label}' already exists")
        
        if len(value) > limit:
            raise ValueError(f"Value length ({len(value)}) exceeds limit ({limit})")
        
        block_id = f"block-{uuid.uuid4()}"
        now = datetime.now(timezone.utc).isoformat()
        
        table = self._get_table()
        table.add([{
            "id": block_id,
            "label": label,
            "value": value,
            "description": description,
            "limit": limit,
            "read_only": read_only,
            "hidden": hidden,
            "created_at": now,
            "updated_at": now,
        }])
        
        logger.info(f"Created block '{label}' ({block_id})")
        return block_id
    
    def get(self, block_id: str) -> Optional[dict]:
        """Get a block by ID.
        
        Args:
            block_id: Block ID
            
        Returns:
            Block dict or None
        """
        table = self._get_table()
        results = table.search().where(f"id = '{block_id}'").limit(1).to_list()
        return results[0] if results else None
    
    def get_by_label(self, label: str) -> Optional[dict]:
        """Get a block by label.
        
        Args:
            label: Block label
            
        Returns:
            Block dict or None
        """
        table = self._get_table()
        results = table.search().where(f"label = '{label}'").limit(1).to_list()
        return results[0] if results else None
    
    def update(
        self,
        label: str,
        value: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update a block's value or description.
        
        Args:
            label: Block label
            value: New value (if provided)
            description: New description (if provided)
            
        Returns:
            True if updated
        """
        block = self.get_by_label(label)
        if not block:
            return False
        
        if block.get("read_only") and value is not None:
            raise ValueError(f"Block '{label}' is read-only")
        
        if value is not None and len(value) > block["limit"]:
            raise ValueError(f"Value length ({len(value)}) exceeds limit ({block['limit']})")
        
        # Build update
        updates = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if value is not None:
            updates["value"] = value
        if description is not None:
            updates["description"] = description
        
        # LanceDB update via delete + add
        table = self._get_table()
        table.delete(f"id = '{block['id']}'")
        
        updated_block = {**block, **updates}
        table.add([updated_block])
        
        logger.debug(f"Updated block '{label}'")
        return True
    
    def delete(self, label: str) -> bool:
        """Delete a block.
        
        Args:
            label: Block label
            
        Returns:
            True if deleted
        """
        block = self.get_by_label(label)
        if not block:
            return False
        
        table = self._get_table()
        table.delete(f"id = '{block['id']}'")
        
        logger.info(f"Deleted block '{label}'")
        return True
    
    def list_blocks(self, include_hidden: bool = False) -> list[dict]:
        """List all blocks.
        
        Args:
            include_hidden: Include hidden blocks
            
        Returns:
            List of blocks
        """
        table = self._get_table()
        results = table.search().limit(1000).to_list()
        
        # Filter out init block and optionally hidden
        blocks = []
        for block in results:
            if block["label"] == "_init_":
                continue
            if not include_hidden and block.get("hidden"):
                continue
            blocks.append(block)
        
        return sorted(blocks, key=lambda b: b["label"])
    
    def str_replace(self, label: str, old_str: str, new_str: str) -> bool:
        """Replace text in a block (like Letta's core_memory_replace).
        
        Args:
            label: Block label
            old_str: Text to find
            new_str: Replacement text
            
        Returns:
            True if replaced
        """
        block = self.get_by_label(label)
        if not block:
            raise ValueError(f"Block '{label}' not found")
        
        if old_str not in block["value"]:
            raise ValueError(f"Text '{old_str}' not found in block '{label}'")
        
        new_value = block["value"].replace(old_str, new_str, 1)
        return self.update(label, value=new_value)
    
    def append(self, label: str, text: str) -> bool:
        """Append text to a block.
        
        Args:
            label: Block label
            text: Text to append
            
        Returns:
            True if appended
        """
        block = self.get_by_label(label)
        if not block:
            raise ValueError(f"Block '{label}' not found")
        
        new_value = block["value"] + text
        return self.update(label, value=new_value)
