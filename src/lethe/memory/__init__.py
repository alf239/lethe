"""Lethe Memory Layer - Local memory management with LanceDB.

Replaces Letta Cloud with a local, reliable memory backend.
- Memory blocks (core memory)
- Archival memory with hybrid search (vector + FTS)
- Message history
"""

from lethe.memory.store import MemoryStore
from lethe.memory.blocks import BlockManager
from lethe.memory.archival import ArchivalMemory
from lethe.memory.messages import MessageHistory

__all__ = ["MemoryStore", "BlockManager", "ArchivalMemory", "MessageHistory"]
