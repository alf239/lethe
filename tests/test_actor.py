"""Tests for the actor model.

Tests cover:
- Actor lifecycle (create, run, terminate)
- Inter-actor messaging
- Group discovery
- Principal vs subagent roles
- Actor tools (spawn, send, discover, terminate)
- Runner execution loop
- Edge cases (timeout, max turns, orphaned actors)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lethe.actor import (
    Actor,
    ActorConfig,
    ActorInfo,
    ActorMessage,
    ActorRegistry,
    ActorState,
)
from lethe.actor.tools import create_actor_tools
from lethe.actor.runner import ActorRunner


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def registry():
    return ActorRegistry()


@pytest.fixture
def principal_config():
    return ActorConfig(
        name="butler",
        group="main",
        goals="Serve the user. Delegate subtasks to subagents.",
    )


@pytest.fixture
def worker_config():
    return ActorConfig(
        name="researcher",
        group="main",
        goals="Research the topic and report findings.",
        tools=["web_search", "read_file"],
        max_turns=5,
    )


@pytest.fixture
def principal(registry, principal_config):
    return registry.spawn(principal_config, is_principal=True)


@pytest.fixture
def worker(registry, principal, worker_config):
    return registry.spawn(worker_config, spawned_by=principal.id)


# ── Basic Lifecycle ───────────────────────────────────────────


class TestActorLifecycle:
    def test_create_actor(self, principal):
        assert principal.state == ActorState.RUNNING
        assert principal.is_principal is True
        assert principal.config.name == "butler"
        assert len(principal.id) == 8

    def test_create_worker(self, worker, principal):
        assert worker.state == ActorState.RUNNING
        assert worker.is_principal is False
        assert worker.spawned_by == principal.id
        assert worker.config.name == "researcher"

    def test_terminate(self, worker):
        worker.terminate("Task complete: found 3 papers")
        assert worker.state == ActorState.TERMINATED
        assert worker._result == "Task complete: found 3 papers"

    def test_actor_info(self, worker, principal):
        info = worker.info
        assert isinstance(info, ActorInfo)
        assert info.name == "researcher"
        assert info.group == "main"
        assert info.spawned_by == principal.id
        assert info.state == ActorState.RUNNING


# ── Registry ──────────────────────────────────────────────────


class TestActorRegistry:
    def test_spawn_principal(self, registry, principal_config):
        principal = registry.spawn(principal_config, is_principal=True)
        assert registry.get_principal() is principal
        assert registry.active_count == 1

    def test_spawn_multiple(self, registry, principal):
        w1 = registry.spawn(ActorConfig(name="w1", group="main", goals="task1"), spawned_by=principal.id)
        w2 = registry.spawn(ActorConfig(name="w2", group="main", goals="task2"), spawned_by=principal.id)
        assert registry.active_count == 3  # principal + 2 workers

    def test_discover_group(self, registry, principal, worker):
        actors = registry.discover("main")
        assert len(actors) == 2
        names = {a.name for a in actors}
        assert names == {"butler", "researcher"}

    def test_discover_empty_group(self, registry, principal):
        actors = registry.discover("nonexistent")
        assert len(actors) == 0

    def test_discover_excludes_terminated(self, registry, principal, worker):
        worker.terminate("done")
        actors = registry.discover("main")
        assert len(actors) == 1
        assert actors[0].name == "butler"

    def test_get_children(self, registry, principal):
        w1 = registry.spawn(ActorConfig(name="w1", group="main", goals="t1"), spawned_by=principal.id)
        w2 = registry.spawn(ActorConfig(name="w2", group="main", goals="t2"), spawned_by=principal.id)
        children = registry.get_children(principal.id)
        assert len(children) == 2

    def test_cleanup_terminated(self, registry, principal, worker):
        worker.terminate("done")
        assert len(registry._actors) == 2
        registry.cleanup_terminated()
        assert len(registry._actors) == 1
        assert registry.get(worker.id) is None


# ── Messaging ─────────────────────────────────────────────────


class TestActorMessaging:
    @pytest.mark.asyncio
    async def test_send_message(self, principal, worker):
        await principal.send_to(worker.id, "Hello worker!")
        msg = await worker.wait_for_reply(timeout=1.0)
        assert msg is not None
        assert msg.content == "Hello worker!"
        assert msg.sender == principal.id

    @pytest.mark.asyncio
    async def test_bidirectional_messaging(self, principal, worker):
        await principal.send_to(worker.id, "Do the task")
        msg = await worker.wait_for_reply(timeout=1.0)
        assert msg.content == "Do the task"
        
        await worker.send_to(principal.id, "Task done!")
        reply = await principal.wait_for_reply(timeout=1.0)
        assert reply.content == "Task done!"

    @pytest.mark.asyncio
    async def test_wait_timeout(self, principal):
        msg = await principal.wait_for_reply(timeout=0.1)
        assert msg is None

    @pytest.mark.asyncio
    async def test_message_history(self, principal, worker):
        await principal.send_to(worker.id, "msg1")
        await worker.send_to(principal.id, "msg2")
        await principal.send_to(worker.id, "msg3")
        
        # Both actors should have all messages they participated in
        assert len(principal._messages) == 3
        assert len(worker._messages) == 3

    @pytest.mark.asyncio
    async def test_send_to_nonexistent(self, principal):
        with pytest.raises(ValueError, match="not found"):
            await principal.send_to("nonexistent", "hello")

    def test_message_format(self):
        msg = ActorMessage(
            sender="abc",
            recipient="def",
            content="Hello there",
        )
        formatted = msg.format()
        assert "abc" in formatted
        assert "Hello there" in formatted


# ── Termination Notification ──────────────────────────────────


class TestTerminationNotification:
    @pytest.mark.asyncio
    async def test_parent_notified_on_child_termination(self, registry, principal, worker):
        """When a child terminates, parent receives notification."""
        worker.terminate("Found 5 results")
        
        # Give the async notification a moment to propagate
        await asyncio.sleep(0.1)
        
        msg = await principal.wait_for_reply(timeout=1.0)
        assert msg is not None
        assert "TERMINATED" in msg.content
        assert "Found 5 results" in msg.content


# ── Actor Tools ───────────────────────────────────────────────


class TestActorTools:
    def test_principal_gets_spawn_tool(self, principal, registry):
        tools = create_actor_tools(principal, registry)
        tool_names = {func.__name__ for func, _ in tools}
        assert "spawn_subagent" in tool_names
        assert "send_message" in tool_names
        assert "discover_actors" in tool_names
        assert "terminate" in tool_names

    def test_worker_no_spawn_by_default(self, worker, registry):
        tools = create_actor_tools(worker, registry)
        tool_names = {func.__name__ for func, _ in tools}
        assert "spawn_subagent" not in tool_names
        assert "send_message" in tool_names

    def test_worker_with_spawn_permission(self, registry, principal):
        config = ActorConfig(
            name="manager",
            group="main",
            goals="Manage tasks",
            tools=["spawn"],
        )
        manager = registry.spawn(config, spawned_by=principal.id)
        tools = create_actor_tools(manager, registry)
        tool_names = {func.__name__ for func, _ in tools}
        assert "spawn_subagent" in tool_names

    @pytest.mark.asyncio
    async def test_send_message_tool(self, principal, worker, registry):
        tools = create_actor_tools(principal, registry)
        send_fn = next(func for func, _ in tools if func.__name__ == "send_message")
        
        result = await send_fn(actor_id=worker.id, content="Hello from tool")
        assert "Message sent" in result
        
        msg = await worker.wait_for_reply(timeout=1.0)
        assert msg.content == "Hello from tool"

    @pytest.mark.asyncio
    async def test_send_message_to_terminated(self, principal, worker, registry):
        worker.terminate("done")
        tools = create_actor_tools(principal, registry)
        send_fn = next(func for func, _ in tools if func.__name__ == "send_message")
        
        result = await send_fn(actor_id=worker.id, content="Are you there?")
        assert "terminated" in result

    def test_discover_tool(self, principal, worker, registry):
        tools = create_actor_tools(principal, registry)
        discover_fn = next(func for func, _ in tools if func.__name__ == "discover_actors")
        
        result = discover_fn()  # Default: own group
        assert "researcher" in result
        assert "butler" in result

    def test_terminate_tool(self, worker, registry):
        tools = create_actor_tools(worker, registry)
        terminate_fn = next(func for func, _ in tools if func.__name__ == "terminate")
        
        result = terminate_fn(result="All done")
        assert "Terminated" in result
        assert worker.state == ActorState.TERMINATED

    @pytest.mark.asyncio
    async def test_spawn_subagent_tool(self, principal, registry):
        tools = create_actor_tools(principal, registry)
        spawn_fn = next(func for func, _ in tools if func.__name__ == "spawn_subagent")
        
        result = await spawn_fn(
            name="coder",
            goals="Write the implementation",
            tools="read_file,write_file",
        )
        assert "Spawned actor 'coder'" in result
        assert registry.active_count == 2  # principal + coder


# ── System Prompt Building ────────────────────────────────────


class TestSystemPrompt:
    def test_principal_prompt(self, principal, worker):
        prompt = principal.build_system_prompt()
        assert "principal actor" in prompt
        assert "butler" in prompt.lower() or "ONLY actor" in prompt
        assert "spawn" in prompt.lower()

    def test_worker_prompt(self, worker, principal):
        prompt = worker.build_system_prompt()
        assert "subagent" in prompt
        assert "researcher" in prompt
        assert worker.config.goals in prompt
        assert "CANNOT talk to the user" in prompt

    def test_group_awareness_in_prompt(self, principal, worker):
        prompt = worker.build_system_prompt()
        assert "butler" in prompt  # Should see the principal in group
        assert "group_actors" in prompt

    @pytest.mark.asyncio
    async def test_inbox_in_prompt(self, principal, worker):
        await principal.send_to(worker.id, "Check the database")
        prompt = worker.build_system_prompt()
        assert "Check the database" in prompt
        assert "inbox" in prompt


# ── Context Messages ──────────────────────────────────────────


class TestContextMessages:
    @pytest.mark.asyncio
    async def test_get_context_messages(self, principal, worker):
        await principal.send_to(worker.id, "Do task")
        await worker.send_to(principal.id, "Done")
        
        ctx = worker.get_context_messages()
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"  # Message FROM principal
        assert "Do task" in ctx[0]["content"]
        assert ctx[1]["role"] == "assistant"  # Message FROM self
        assert "Done" in ctx[1]["content"]


# ── Runner ────────────────────────────────────────────────────


class TestActorRunner:
    @pytest.mark.asyncio
    async def test_runner_basic(self, registry, principal):
        """Runner should execute and terminate actor."""
        worker = registry.spawn(
            ActorConfig(name="test_worker", group="main", goals="Say hello and terminate"),
            spawned_by=principal.id,
        )
        
        # Mock LLM that terminates on first turn
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value="Done!")
        mock_llm.add_tool = MagicMock()
        mock_llm.context = MagicMock()
        
        # Make terminate get called by simulating the tool execution
        call_count = 0
        async def fake_chat(message):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                worker.terminate("Task complete")
            return "Working..."
        mock_llm.chat = fake_chat
        
        async def mock_factory(actor):
            return mock_llm
        
        runner = ActorRunner(worker, registry, mock_factory)
        result = await runner.run()
        
        assert worker.state == ActorState.TERMINATED
        assert "Task complete" in result

    @pytest.mark.asyncio
    async def test_runner_max_turns(self, registry, principal):
        """Runner should force terminate after max turns."""
        worker = registry.spawn(
            ActorConfig(
                name="slow_worker",
                group="main",
                goals="Take forever",
                max_turns=3,
            ),
            spawned_by=principal.id,
        )
        
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value="Still working...")
        mock_llm.add_tool = MagicMock()
        mock_llm.context = MagicMock()
        
        async def mock_factory(actor):
            return mock_llm
        
        runner = ActorRunner(worker, registry, mock_factory)
        result = await runner.run()
        
        assert worker.state == ActorState.TERMINATED
        assert "Max turns" in result


# ── Multi-Group Isolation ─────────────────────────────────────


class TestGroupIsolation:
    def test_groups_are_isolated(self, registry):
        """Actors in different groups can't discover each other."""
        a1 = registry.spawn(ActorConfig(name="a1", group="team_a", goals="task_a"))
        a2 = registry.spawn(ActorConfig(name="a2", group="team_b", goals="task_b"))
        
        group_a = registry.discover("team_a")
        group_b = registry.discover("team_b")
        
        assert len(group_a) == 1
        assert group_a[0].name == "a1"
        assert len(group_b) == 1
        assert group_b[0].name == "a2"


# ── Example Scenarios ─────────────────────────────────────────


class TestExampleScenarios:
    """Integration-style tests showing real usage patterns."""

    @pytest.mark.asyncio
    async def test_research_delegation(self, registry):
        """Principal spawns a researcher, researcher reports back."""
        # Principal
        butler = registry.spawn(
            ActorConfig(name="butler", group="research", goals="Help user with research"),
            is_principal=True,
        )
        
        # Spawn researcher
        researcher = registry.spawn(
            ActorConfig(
                name="researcher",
                group="research",
                goals="Find papers about transformer architectures",
                tools=["web_search"],
            ),
            spawned_by=butler.id,
        )
        
        # Researcher works and reports back
        await researcher.send_to(butler.id, "Found 5 papers on transformer architectures")
        researcher.terminate("Found 5 papers: Attention Is All You Need, ...")
        
        # Butler receives both the message and termination notice
        await asyncio.sleep(0.1)
        
        # Butler should have received messages
        assert len(butler._messages) > 0
        # Researcher should be terminated
        assert researcher.state == ActorState.TERMINATED
        # Only butler active
        assert registry.active_count == 1

    @pytest.mark.asyncio
    async def test_multi_actor_collaboration(self, registry):
        """Multiple actors working together in the same group."""
        butler = registry.spawn(
            ActorConfig(name="butler", group="project", goals="Coordinate the project"),
            is_principal=True,
        )
        
        coder = registry.spawn(
            ActorConfig(name="coder", group="project", goals="Write the code"),
            spawned_by=butler.id,
        )
        
        reviewer = registry.spawn(
            ActorConfig(name="reviewer", group="project", goals="Review the code"),
            spawned_by=butler.id,
        )
        
        # All three can discover each other
        group = registry.discover("project")
        assert len(group) == 3
        
        # Coder finishes and tells reviewer
        await coder.send_to(reviewer.id, "Code ready for review: src/main.py")
        
        msg = await reviewer.wait_for_reply(timeout=1.0)
        assert "Code ready" in msg.content
        
        # Reviewer approves and tells butler
        await reviewer.send_to(butler.id, "Code LGTM, approved")
        reviewer.terminate("Review complete: approved")
        
        approval = await butler.wait_for_reply(timeout=1.0)
        assert "LGTM" in approval.content
        
        # Coder terminates
        coder.terminate("Code written and approved")
        
        await asyncio.sleep(0.1)
        
        # Only butler remains
        assert registry.active_count == 1
        assert butler.state == ActorState.RUNNING
