"""Hippocampus - Pattern completion memory retrieval.

Inspired by biological hippocampus CA3 region which performs autoassociative
pattern completion: given a partial cue, retrieve the complete memory.

Uses LLM to decide if recall would help and generate concise search queries.
This produces better results than raw message similarity search.
"""

import json
import logging
import re
from typing import Optional, Callable, Awaitable
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Max lines of recalled memories before summarization
MAX_RECALL_LINES = 500

# Minimum score threshold for including memories
MIN_SCORE_THRESHOLD = 0.3

# Decision prompt - should we recall?
ANALYZE_PROMPT = """You are a memory retrieval assistant. Decide if looking up memories would benefit the current conversation.

RECENT CONTEXT:
{context}

NEW USER MESSAGE:
{message}

Would looking up memories (past conversations, archival notes, credentials, previous decisions) benefit the response?

Think about:
- Does this reference something from before?
- Would past context improve the answer?
- Are there credentials/configs/patterns we discussed?
- Is this a continuation of previous work?

Look for:
- References to people, places, projects, or things mentioned before
- Questions that might have been answered previously
- Credentials, API keys, configurations discussed before
- Patterns, preferences, or decisions made in the past

Do NOT recall for:
- Simple greetings ("Hello!", "Hi")
- Self-contained questions ("What's 2+2?")
- New topics with no prior context
- Explicit "forget" or "start fresh" requests

Respond ONLY with valid JSON:
{{"should_recall": true/false, "search_query": "2-5 word query or null", "reason": "brief reason"}}

Examples:
- "Deploy the app" -> {{"should_recall": true, "search_query": "server deployment config", "reason": "may need server details"}}
- "What did we decide about the API?" -> {{"should_recall": true, "search_query": "API design decisions", "reason": "explicit reference to past"}}
- "Hello!" -> {{"should_recall": false, "search_query": null, "reason": "simple greeting"}}
- "Fix the bug in auth.py" -> {{"should_recall": true, "search_query": "auth.py issues", "reason": "may have context"}}

JSON only:"""

# Summarization prompt - preserves reference data
SUMMARIZE_PROMPT = """Summarize these recalled memories concisely for context. 

CRITICAL: Preserve ALL of the following exactly as-is (do not paraphrase or omit):
- Timestamps and dates (keep [YYYY-MM-DD HH:MM] format)
- URLs, links, file paths
- Credentials, API keys, tokens
- IDs, reference numbers
- Names of people, projects, tools
- Code snippets, commands
- Specific numbers and measurements

Keep timing context - when things happened matters. Strip filler and redundancy, keep facts dense.

Memories to summarize:
{memories}

Summary (preserve timestamps and reference data):"""


# Warning added to recall block
ACAUSAL_WARNING = """WARNING: This recall is acausal - these memories may be from the past and do not reflect current state. Do NOT use recalled memories to determine what is done or pending. Use conversation history, todo tools, and memory blocks for current state."""


class Hippocampus:
    """Pattern completion memory retrieval with LLM-guided search.
    
    Uses LLM to:
    1. Decide if memory recall would benefit the conversation
    2. Generate concise search queries (2-5 words) for better similarity matching
    3. Summarize retrieved memories to compress context
    """
    
    def __init__(
        self, 
        memory_store, 
        summarizer: Optional[Callable[[str], Awaitable[str]]] = None,
        analyzer: Optional[Callable[[str], Awaitable[str]]] = None,
        enabled: bool = True,
    ):
        """Initialize hippocampus.
        
        Args:
            memory_store: MemoryStore instance with archival and messages
            summarizer: Async function to summarize memories (uses aux model)
            analyzer: Async function to analyze if recall needed (uses aux model)
            enabled: Whether to enable memory recall
        """
        self.memory = memory_store
        self.summarizer = summarizer
        self.analyzer = analyzer or summarizer  # Use same function if not specified
        self.enabled = enabled
        logger.info(f"Hippocampus initialized (enabled={enabled}, summarizer={summarizer is not None})")
    
    async def recall(
        self,
        message: str,
        recent_messages: Optional[list[dict]] = None,
        max_lines: int = MAX_RECALL_LINES,
    ) -> Optional[str]:
        """Recall relevant memories for a user message.
        
        Uses LLM to decide if recall is needed and generate optimized search query.
        
        Args:
            message: The new user message
            recent_messages: Recent conversation context (optional)
            max_lines: Maximum lines of memories before summarization
            
        Returns:
            Formatted (and optionally summarized) memory recall string
        """
        if not self.enabled:
            return None
        
        # Step 1: Ask LLM if we should recall and get optimized query
        analysis = await self._analyze_for_recall(message, recent_messages)
        
        if not analysis or not analysis.get("should_recall"):
            reason = analysis.get("reason") if analysis else "analysis failed"
            logger.info(f"Hippocampus: skipping recall - {reason}")
            return None
        
        search_query = analysis.get("search_query")
        if not search_query:
            logger.warning("Hippocampus: should_recall=True but no search_query")
            return None
        
        logger.info(f"Hippocampus: searching with query '{search_query}' (reason: {analysis.get('reason')})")
        
        # Step 2: Search with LLM-generated query
        archival_results = self._search_archival(search_query)
        conversation_results = self._search_conversations(search_query, exclude_recent=5)
        
        # Combine and format results
        memories = self._format_memories(archival_results, conversation_results, max_lines)
        
        if not memories:
            logger.info("Hippocampus: no memories found for query")
            return None
        
        # Step 3: Summarize if we have a summarizer
        if self.summarizer:
            return await self._summarize(memories)
        else:
            return (
                "<associative_memory_recall>\n"
                + ACAUSAL_WARNING + "\n\n"
                + memories
                + "\n</associative_memory_recall>"
            )
    
    async def _analyze_for_recall(
        self,
        message: str,
        recent_messages: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """Use LLM to decide if recall is needed and generate search query.
        
        Returns:
            Dict with keys: should_recall (bool), search_query (str|None), reason (str)
            Returns None if analysis fails
        """
        # Handle multimodal content (list of parts) - extract text
        if isinstance(message, list):
            text_parts = []
            for part in message:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            message = " ".join(text_parts) if text_parts else "(image)"
        
        if not self.analyzer:
            # Fallback: always recall with raw query
            return {"should_recall": True, "search_query": message[:100], "reason": "no analyzer"}
        
        try:
            # Build context string
            context = self._format_context(recent_messages)
            
            # Ask LLM
            prompt = ANALYZE_PROMPT.format(context=context, message=message)
            response = await self.analyzer(prompt)
            
            if not response:
                return None
            
            # Parse JSON response
            try:
                # Try direct parse
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.warning(f"Hippocampus: invalid JSON response: {response[:200]}")
                    return None
            
            return result
            
        except Exception as e:
            logger.warning(f"Hippocampus analysis failed: {e}")
            return None
    
    def _format_context(
        self,
        recent_messages: Optional[list[dict]] = None,
    ) -> str:
        """Format recent messages as context for the analyzer."""
        if not recent_messages:
            return "(new conversation)"
        
        context_lines = []
        for msg in recent_messages[-5:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content 
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines) if context_lines else "(new conversation)"
    
    def _search_archival(self, query: str, limit: int = 5) -> list[dict]:
        """Search archival memory."""
        try:
            results = self.memory.archival.search(
                query,
                limit=limit,
                search_type="hybrid"
            )
            # Filter by score threshold
            return [r for r in results if r.get("score", 0) >= MIN_SCORE_THRESHOLD]
        except Exception as e:
            logger.warning(f"Archival search failed: {e}")
            return []
    
    def _search_conversations(
        self,
        query: str,
        limit: int = 5,
        exclude_recent: int = 5,
    ) -> list[dict]:
        """Search conversation history, excluding very recent messages."""
        try:
            results = self.memory.messages.search(query, limit=limit + exclude_recent)
            # Skip the most recent messages (they're already in context)
            return results[exclude_recent:] if len(results) > exclude_recent else []
        except Exception as e:
            logger.warning(f"Conversation search failed: {e}")
            return []
    
    def _format_memories(
        self,
        archival: list[dict],
        conversations: list[dict],
        max_lines: int,
    ) -> Optional[str]:
        """Format retrieved memories into a context block."""
        if not archival and not conversations:
            return None
        
        sections = []
        total_lines = 0
        
        # Format archival memories
        if archival:
            archival_lines = []
            for mem in archival:
                if total_lines >= max_lines:
                    break
                    
                text = mem.get("text", "")
                created = mem.get("created_at", "")[:16].replace("T", " ")  # YYYY-MM-DD HH:MM
                
                archival_lines.append(f"- [{created}] {text}")
                total_lines += 1
            
            if archival_lines:
                sections.append("**From long-term memory:**\n" + "\n".join(archival_lines))
        
        # Format conversation memories
        if conversations and total_lines < max_lines:
            conv_lines = []
            for msg in conversations:
                if total_lines >= max_lines:
                    break
                    
                role = msg.get("role", "?")
                content = msg.get("content", "")
                created = msg.get("created_at", "")[:16].replace("T", " ")  # YYYY-MM-DD HH:MM
                
                conv_lines.append(f"- [{created}] {role}: {content}")
                total_lines += 1
            
            if conv_lines:
                sections.append("**From past conversations:**\n" + "\n".join(conv_lines))
        
        if not sections:
            return None
        
        return "\n\n".join(sections)
    
    async def _summarize(self, memories: str) -> str:
        """Summarize memories using the configured summarizer."""
        try:
            prompt = SUMMARIZE_PROMPT.format(memories=memories)
            summary = await self.summarizer(prompt)
            
            if summary:
                logger.info(f"Summarized {len(memories)} -> {len(summary)} chars")
                return (
                    "<associative_memory_recall summarized=\"true\">\n"
                    + ACAUSAL_WARNING + "\n\n"
                    + summary.strip()
                    + "\n</associative_memory_recall>"
                )
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
        
        # Fallback to unsummarized
        return (
            "<associative_memory_recall>\n"
            + ACAUSAL_WARNING + "\n\n"
            + memories
            + "\n</associative_memory_recall>"
        )
    
    async def augment_message(
        self,
        message,  # Can be str or list (multimodal)
        recent_messages: Optional[list[dict]] = None,
    ):
        """Augment a user message with recalled memories.
        
        Args:
            message: The user message (str or multimodal list)
            recent_messages: Recent conversation context
            
        Returns:
            Original message, possibly with appended memory context
        """
        recall = await self.recall(message, recent_messages)
        
        if recall:
            logger.info(f"Hippocampus recalled {len(recall)} chars of context")
            # Handle multimodal content
            if isinstance(message, list):
                # Append recall as text part
                return message + [{"type": "text", "text": f"\n\n{recall}"}]
            return f"{message}\n\n{recall}"
        
        return message
