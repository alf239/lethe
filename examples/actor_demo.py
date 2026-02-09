#!/usr/bin/env python3
"""Demo: Actor model usage patterns.

Shows how to:
1. Create a principal actor (butler)
2. Spawn subagent actors for tasks
3. Inter-actor communication
4. Group discovery
5. Lifecycle management

This is a standalone example — no LLM calls, just demonstrating the framework.
"""

import asyncio
from lethe.actor import ActorConfig, ActorRegistry, ActorState


async def demo_basic_spawn():
    """Demo 1: Principal spawns a worker, worker reports back."""
    print("\n=== Demo 1: Basic Spawn & Report ===\n")
    
    registry = ActorRegistry()
    
    # Create principal
    butler = registry.spawn(
        ActorConfig(name="butler", group="demo", goals="Coordinate tasks for the user"),
        is_principal=True,
    )
    print(f"Principal: {butler.config.name} (id={butler.id})")
    
    # Spawn a researcher
    researcher = registry.spawn(
        ActorConfig(
            name="researcher",
            group="demo",
            goals="Find information about quantum computing advances in 2025",
            tools=["web_search", "read_file"],
            max_turns=10,
        ),
        spawned_by=butler.id,
    )
    print(f"Spawned: {researcher.config.name} (id={researcher.id})")
    
    # Researcher sends progress update
    await researcher.send_to(butler.id, "Found 3 papers on quantum error correction")
    msg = await butler.wait_for_reply(timeout=1.0)
    print(f"Butler received: {msg.content}")
    
    # Researcher finishes
    researcher.terminate("Found 3 papers: [1] Surface codes 2025, [2] Logical qubits milestone, [3] IBM Condor results")
    
    # Butler gets termination notice
    await asyncio.sleep(0.1)
    notice = await butler.wait_for_reply(timeout=1.0)
    print(f"Butler got notice: {notice.content[:80]}...")
    
    print(f"\nActive actors: {registry.active_count}")
    print(f"Butler state: {butler.state.value}")
    print(f"Researcher state: {researcher.state.value}")


async def demo_multi_actor():
    """Demo 2: Multiple actors collaborating on a project."""
    print("\n=== Demo 2: Multi-Actor Collaboration ===\n")
    
    registry = ActorRegistry()
    
    butler = registry.spawn(
        ActorConfig(name="butler", group="project", goals="Build a web app"),
        is_principal=True,
    )
    
    # Spawn team
    designer = registry.spawn(
        ActorConfig(name="designer", group="project", goals="Design the UI mockups"),
        spawned_by=butler.id,
    )
    coder = registry.spawn(
        ActorConfig(name="coder", group="project", goals="Implement the frontend"),
        spawned_by=butler.id,
    )
    reviewer = registry.spawn(
        ActorConfig(name="reviewer", group="project", goals="Review code quality"),
        spawned_by=butler.id,
    )
    
    print(f"Team assembled: {registry.active_count} actors")
    
    # Everyone can discover each other
    group = registry.discover("project")
    print(f"Group members:")
    for info in group:
        print(f"  {info.format()}")
    
    # Designer sends mockups to coder
    await designer.send_to(coder.id, "Mockups ready: 3 screens (login, dashboard, settings)")
    await coder.send_to(designer.id, "Got it, implementing now")
    designer.terminate("Mockups complete: login, dashboard, settings screens")
    
    # Coder sends to reviewer
    await coder.send_to(reviewer.id, "PR ready: implemented all 3 screens")
    review_msg = await reviewer.wait_for_reply(timeout=1.0)
    print(f"\nReviewer got: {review_msg.content}")
    
    # Reviewer reports to butler
    await reviewer.send_to(butler.id, "Code review: LGTM, all screens implemented correctly")
    reviewer.terminate("Review complete: approved")
    
    coder.terminate("Implementation complete: 3 screens built")
    
    await asyncio.sleep(0.1)
    
    print(f"\nFinal state: {registry.active_count} active actors")
    for info in registry.all_actors:
        print(f"  {info.name}: {info.state.value}")


async def demo_group_isolation():
    """Demo 3: Groups provide isolation between different tasks."""
    print("\n=== Demo 3: Group Isolation ===\n")
    
    registry = ActorRegistry()
    
    butler = registry.spawn(
        ActorConfig(name="butler", group="main", goals="Manage multiple projects"),
        is_principal=True,
    )
    
    # Two separate teams in different groups
    team_a_lead = registry.spawn(
        ActorConfig(name="alpha-lead", group="team_alpha", goals="Build the API"),
        spawned_by=butler.id,
    )
    team_a_dev = registry.spawn(
        ActorConfig(name="alpha-dev", group="team_alpha", goals="Implement endpoints"),
        spawned_by=team_a_lead.id,
    )
    
    team_b_lead = registry.spawn(
        ActorConfig(name="beta-lead", group="team_beta", goals="Build the frontend"),
        spawned_by=butler.id,
    )
    
    print(f"Total actors: {registry.active_count}")
    print(f"Team Alpha sees: {[a.name for a in registry.discover('team_alpha')]}")
    print(f"Team Beta sees: {[a.name for a in registry.discover('team_beta')]}")
    print(f"Main sees: {[a.name for a in registry.discover('main')]}")
    
    # Butler can see its direct children
    children = registry.get_children(butler.id)
    print(f"Butler's children: {[c.config.name for c in children]}")


async def demo_system_prompt():
    """Demo 4: Show how system prompts are built for different roles."""
    print("\n=== Demo 4: System Prompts ===\n")
    
    registry = ActorRegistry()
    
    butler = registry.spawn(
        ActorConfig(name="butler", group="demo", goals="Serve the user"),
        is_principal=True,
    )
    worker = registry.spawn(
        ActorConfig(name="researcher", group="demo", goals="Research AI papers"),
        spawned_by=butler.id,
    )
    
    print("--- Principal System Prompt ---")
    print(butler.build_system_prompt()[:500])
    print("...")
    print()
    print("--- Worker System Prompt ---")
    print(worker.build_system_prompt()[:500])
    print("...")


async def main():
    await demo_basic_spawn()
    await demo_multi_actor()
    await demo_group_isolation()
    await demo_system_prompt()
    print("\n=== All demos complete ===")


if __name__ == "__main__":
    asyncio.run(main())
