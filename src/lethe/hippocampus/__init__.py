"""Hippocampus - Eidetic memory retrieval subagent.

The hippocampus agent analyzes incoming messages to determine if recalling
memories would benefit the conversation. It proactively looks for:
- References to people, places, projects mentioned before
- Credentials, configurations, API keys discussed previously
- Past decisions, patterns, or preferences
- Any context where prior knowledge would improve the response

Inspired by the biological hippocampus which consolidates and retrieves memories.
"""

import json
import logging
from typing import Optional

from letta_client import AsyncLetta

from lethe.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Default persona for the hippocampus agent
HIPPOCAMPUS_PERSONA = """You are a memory retrieval assistant. Your job is to decide if looking up memories would benefit the current conversation.

When given a user message, think: would remembering something from past conversations or archival memory help here?

Look for:
- References to people, places, projects, or things mentioned before
- Questions that might have been answered previously
- Credentials, API keys, configurations discussed before
- Patterns, preferences, or decisions made in the past
- Anything where prior context would improve the response

Respond ONLY with valid JSON:
{"should_recall": true/false, "search_query": "query string or null", "reason": "brief reason or null"}

Rules:
- should_recall: true if memory lookup would genuinely help
- search_query: concise query (2-5 words) to search memories
- reason: brief explanation of what you're looking for

Examples:
- "Deploy the app to the server" -> {"should_recall": true, "search_query": "server deployment credentials", "reason": "may need server details from before"}
- "What did we decide about the API design?" -> {"should_recall": true, "search_query": "API design decisions", "reason": "explicit reference to past decision"}
- "Hello!" -> {"should_recall": false, "search_query": null, "reason": null}
- "Fix the bug in auth.py" -> {"should_recall": true, "search_query": "auth.py bugs issues", "reason": "may have discussed this file before"}
- "What's 2+2?" -> {"should_recall": false, "search_query": null, "reason": null}

Be pragmatic - recall when it would actually help, skip for simple or self-contained requests."""


class HippocampusManager:
    """Manages the hippocampus subagent for memory retrieval."""

    def __init__(
        self,
        client: AsyncLetta,
        settings: Optional[Settings] = None,
    ):
        self.client = client
        self.settings = settings or get_settings()
        self._agent_id: Optional[str] = None
        
        # Configuration
        self.agent_name = getattr(self.settings, 'hippocampus_agent_name', 'lethe-hippocampus')
        self.model = getattr(self.settings, 'hippocampus_model', 'anthropic/claude-3-haiku-20240307')
        self.enabled = getattr(self.settings, 'hippocampus_enabled', True)

    async def get_or_create_agent(self) -> str:
        """Get existing hippocampus agent or create a new one."""
        if self._agent_id:
            return self._agent_id

        # Check for existing agent
        agents = self.client.agents.list()
        if hasattr(agents, '__aiter__'):
            async for agent in agents:
                if agent.name == self.agent_name:
                    self._agent_id = agent.id
                    logger.info(f"Found existing hippocampus agent: {self._agent_id}")
                    return self._agent_id
        else:
            for agent in agents:
                if agent.name == self.agent_name:
                    self._agent_id = agent.id
                    logger.info(f"Found existing hippocampus agent: {self._agent_id}")
                    return self._agent_id

        # Create new agent
        logger.info(f"Creating hippocampus agent: {self.agent_name} with model {self.model}")
        agent = await self.client.agents.create(
            name=self.agent_name,
            model=self.model,
            memory_blocks=[
                {"label": "persona", "value": HIPPOCAMPUS_PERSONA, "limit": 2000},
            ],
            tools=[],  # No tools - pure reasoning
            include_base_tools=False,  # No memory tools needed
        )
        self._agent_id = agent.id
        logger.info(f"Created hippocampus agent: {self._agent_id}")
        return self._agent_id

    async def analyze_for_recall(
        self,
        new_message: str,
        recent_messages: list[dict],
    ) -> Optional[dict]:
        """Analyze a new message to determine if memory recall would be beneficial.
        
        Args:
            new_message: The new user message
            recent_messages: List of recent messages [{"role": "user"|"assistant", "content": "..."}]
            
        Returns:
            Dict with keys: should_recall (bool), search_query (str|None), reason (str|None)
            Returns None if hippocampus is disabled or fails
        """
        if not self.enabled:
            return None

        try:
            agent_id = await self.get_or_create_agent()
            
            # Format recent context (brief, just for awareness)
            context_lines = []
            for msg in recent_messages[-5:]:  # Last 5 messages for context
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
            
            context = "\n".join(context_lines) if context_lines else "(new conversation)"
            
            prompt = f"""RECENT CONTEXT:
{context}

NEW USER MESSAGE:
{new_message}

Would looking up memories (past conversations, archival notes, credentials, previous decisions) benefit the response to this message?

Think about:
- Does this reference something from before?
- Would past context improve the answer?
- Are there credentials/configs/patterns we discussed?

JSON only:"""

            # Send to hippocampus agent
            response = await self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract the response text
            result_text = None
            for msg in response.messages:
                if hasattr(msg, 'message_type') and msg.message_type == 'assistant_message':
                    content = msg.content
                    if isinstance(content, str):
                        result_text = content
                    elif isinstance(content, list):
                        for part in content:
                            if hasattr(part, 'text'):
                                result_text = part.text
                                break
                    break

            if not result_text:
                logger.warning("Hippocampus returned no response")
                return None

            # Parse JSON response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^{}]*\}', result_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.warning(f"Hippocampus returned invalid JSON: {result_text[:200]}")
                    return None

            logger.info(f"Hippocampus analysis: should_recall={result.get('should_recall')}, query={result.get('search_query')}, reason={result.get('reason')}")
            return result

        except Exception as e:
            logger.exception(f"Hippocampus analysis failed: {e}")
            return None

    async def search_memories(
        self,
        main_agent_id: str,
        query: str,
        max_results: int = 5,
    ) -> str:
        """Search the main agent's archival and conversation memory.
        
        Args:
            main_agent_id: The main agent's ID to search
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            Formatted string with search results, or empty string if none found
        """
        results = []
        
        try:
            # Search archival memory (passages)
            archival_results = await self.client.agents.passages.search(
                agent_id=main_agent_id,
                query=query,
                top_k=max_results,
            )
            
            # Handle async iterator or list
            passages = []
            if hasattr(archival_results, '__aiter__'):
                async for p in archival_results:
                    passages.append(p)
            elif archival_results:
                passages = list(archival_results)
            
            for passage in passages[:max_results]:
                # Handle different response types
                if hasattr(passage, 'content'):
                    content = passage.content
                elif isinstance(passage, tuple) and len(passage) > 0:
                    content = str(passage[0])
                else:
                    content = str(passage)
                results.append(f"[Archival] {content}")
                    
        except Exception as e:
            logger.warning(f"Archival search failed: {e}")

        try:
            # Search conversation history using organization-wide search with agent filter
            conv_results = await self.client.messages.search(
                query=query,
                agent_id=main_agent_id,
                limit=max_results,
            )
            
            # Handle async iterator or list  
            messages = []
            if hasattr(conv_results, '__aiter__'):
                async for m in conv_results:
                    messages.append(m)
            elif conv_results:
                messages = list(conv_results)
            
            for msg in messages[:max_results]:
                # Handle different message types (some don't have 'content')
                if hasattr(msg, 'content'):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                elif hasattr(msg, 'reasoning'):
                    content = msg.reasoning if isinstance(msg.reasoning, str) else str(msg.reasoning)
                elif hasattr(msg, 'text'):
                    content = msg.text
                else:
                    content = str(msg)
                role = getattr(msg, 'message_type', 'message').replace('_message', '')
                results.append(f"[{role}] {content}")
                    
        except Exception as e:
            logger.warning(f"Conversation search failed: {e}")

        if not results:
            return ""
        
        combined = "\n\n".join(results)
        
        # If results are too long, compress via hippocampus
        if len(combined) > 3000:
            combined = await self._compress_memories(combined, query)
            
        return combined

    async def _compress_memories(self, memories: str, query: str) -> str:
        """Compress long memory results using hippocampus agent.
        
        Args:
            memories: The full memory text to compress
            query: The original search query for context
            
        Returns:
            Compressed summary of the memories
        """
        try:
            agent_id = await self.get_or_create_agent()
            
            prompt = f"""The following memories were retrieved for the query "{query}".
They are too long to include in full. Summarize the key relevant information concisely.
Preserve important facts, names, dates, and context. Do not add information that isn't present.

MEMORIES:
{memories}

SUMMARY (be concise but preserve key details):"""

            response = await self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract response text
            for msg in response.messages:
                if hasattr(msg, 'message_type') and msg.message_type == 'assistant_message':
                    content = msg.content
                    if isinstance(content, str):
                        return f"[Compressed summary] {content}"
                    elif isinstance(content, list):
                        for part in content:
                            if hasattr(part, 'text'):
                                return f"[Compressed summary] {part.text}"
            
            # Fallback - return original if compression failed
            logger.warning("Memory compression returned no response, using original")
            return memories
            
        except Exception as e:
            logger.warning(f"Memory compression failed: {e}")
            return memories

    async def augment_message(
        self,
        main_agent_id: str,
        new_message: str,
        recent_messages: list[dict],
    ) -> str:
        """Analyze message and augment with relevant memories if beneficial.
        
        This is the main entry point - call this before sending to the main agent.
        
        Args:
            main_agent_id: The main agent's ID (for memory search)
            new_message: The new user message
            recent_messages: Recent conversation messages
            
        Returns:
            The message, potentially augmented with recalled memories
        """
        if not self.enabled:
            return new_message

        # Analyze if memory recall would be beneficial
        analysis = await self.analyze_for_recall(new_message, recent_messages)
        
        if not analysis or not analysis.get("should_recall") or not analysis.get("search_query"):
            return new_message

        # Search memories
        query = analysis["search_query"]
        memories = await self.search_memories(main_agent_id, query)
        
        if not memories:
            return new_message

        # Augment the message with recalled memories (append AFTER the user message)
        reason = analysis.get("reason", "potentially relevant context")
        augmented = f"""{new_message}

---
[Memory recall: {reason}]
{memories}
[End of recall]"""
        
        logger.info(f"Augmented message with memory recall ({len(memories)} chars): {reason}")
        return augmented

    async def judge_response(
        self,
        original_request: str,
        agent_response: str,
        iteration: int,
        is_continuation: bool,
    ) -> dict:
        """Judge an agent's response - whether to send it and whether to continue.
        
        Args:
            original_request: The user's original request
            agent_response: The agent's latest response (empty if no response)
            iteration: Current iteration number (0-based)
            is_continuation: Whether this is a continuation prompt response
            
        Returns:
            Dict with keys:
            - send_to_user: bool - whether this response should be sent to user
            - continue_task: bool - whether to prompt agent to continue
            - reason: str - brief explanation
        """
        default_result = {"send_to_user": True, "continue_task": False, "reason": "default"}
        
        if not self.enabled:
            return default_result

        try:
            agent_id = await self.get_or_create_agent()
            
            # If no response and early iteration, continue but don't send
            if not agent_response and iteration <= 2:
                return {"send_to_user": False, "continue_task": True, "reason": "no response early iteration"}
            
            # If no response and later iteration, stop
            if not agent_response:
                return {"send_to_user": False, "continue_task": False, "reason": "no response late iteration"}
            
            prompt = f"""USER REQUEST:
{original_request}

AGENT'S LATEST RESPONSE:
{agent_response}

ITERATION: {iteration}
IS_CONTINUATION_RESPONSE: {is_continuation}

Judge this response:

1. SEND_TO_USER: Should this response be shown to the user?
   - YES if: agent is talking TO the user (direct address, "you", "your", answers, confirmations)
   - NO if: agent is talking ABOUT the user in third person (using their name instead of "you") - this is internal reflection
   - NO if: meta-commentary about the task itself, thinking out loud

2. CONTINUE_TASK: Should the agent continue working?
   - YES if: agent expressed clear intent to do more AND task is obviously incomplete
   - NO if: action completed, natural stopping point, or nothing more to do
   - NO if: send_to_user is false (if we're not sending the response, there's no point continuing)

IMPORTANT: If the response shouldn't be sent to user, almost always set continue_task=false too.
The only exception is during active tool execution where agent is working but hasn't reported yet.

Respond with JSON only:
{{"send_to_user": true/false, "continue_task": true/false, "reason": "brief explanation"}}"""

            response = await self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract response
            result_text = None
            for msg in response.messages:
                if hasattr(msg, 'message_type') and msg.message_type == 'assistant_message':
                    content = msg.content
                    if isinstance(content, str):
                        result_text = content
                    elif isinstance(content, list):
                        for part in content:
                            if hasattr(part, 'text'):
                                result_text = part.text
                                break
                    break

            if not result_text:
                logger.warning("Hippocampus judge_response returned no response")
                return default_result

            # Parse JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^{}]*\}', result_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.warning(f"Hippocampus judge_response invalid JSON: {result_text}")
                    return default_result

            send_to_user = result.get("send_to_user", True)
            continue_task = result.get("continue_task", False)
            reason = result.get("reason", "")
            logger.info(f"Hippocampus judgment: send={send_to_user}, continue={continue_task}, reason={reason}")
            return {"send_to_user": send_to_user, "continue_task": continue_task, "reason": reason}

        except Exception as e:
            logger.warning(f"Hippocampus judge_response failed: {e}")
            return default_result
