#!/usr/bin/env python3
"""Migrate Lethe from single-agent to actor model architecture.

Uses LLM to intelligently rewrite prompts (preserving persona, style, custom
sections) rather than fragile pattern matching. Falls back to template-based
migration if no API key is available.

This script updates memory blocks (identity.md, tools.md) to work with
the actor model where:
- Cortex = conscious executive layer (coordinator, never calls tools directly)
- DMN = Default Mode Network (background thinking, reflections)
- Subagents = spawned workers with specific tools and goals

Idempotent — safe to run multiple times. Backs up originals before overwriting.

Usage:
    python scripts/migrate_to_actors.py [--config-dir ./config/blocks]
    python scripts/migrate_to_actors.py --no-llm     # template-only, no API calls
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# ─── Actor model reference (fed to LLM for context) ──────────────────────

ACTOR_MODEL_REFERENCE = """
## Actor Model Architecture

The agent now operates as a multi-agent system:

### Cortex (Principal Actor)
- The conscious executive layer — the ONLY agent that talks to the user
- Pure COORDINATOR — never calls file/CLI/web tools directly
- Delegates ALL work to subagents via spawn_actor()
- Keeps: actor tools, memory tools, telegram tools

### Default Mode Network (DMN)
- Persistent background thinker, runs every 15 minutes
- Scans goals, todos, reminders
- Reorganizes memory, writes reflections
- Notifies cortex of urgent items

### Subagents
- Spawned on demand for specific tasks
- Have: bash, read_file, write_file, edit_file, list_directory, grep_search (always)
- Extra tools: web_search, fetch_webpage, browser tools (on request)
- Cannot access telegram — only actor messaging
- Report results back to cortex

### Cortex Tools
Actor management: spawn_actor, kill_actor, ping_actor, send_message,
  discover_actors, wait_for_response, terminate
Memory: memory_read, memory_update, memory_append, archival_search/insert,
  conversation_search
Telegram: telegram_send_message, telegram_send_file

### Subagent Extra Tools (specified in spawn_actor)
web_search, fetch_webpage, browser_open, browser_click, browser_fill,
browser_snapshot, memory_read, memory_update, memory_append
"""

# ─── LLM-powered migration ──────────────────────────────────────────────

IDENTITY_REWRITE_PROMPT = """You are migrating an AI agent's identity prompt to support a new actor model architecture.

The agent is moving from single-agent (does everything itself) to multi-agent:
- The agent becomes the "cortex" — a pure coordinator that delegates to subagents
- It NEVER calls file/CLI/web tools directly anymore
- It spawns subagents for ALL work

{actor_reference}

Here is the current identity prompt:

<current_identity>
{identity}
</current_identity>

Rewrite this identity prompt to be actor-model aware. Rules:
1. PRESERVE the persona, personality, character, communication style, and all custom sections EXACTLY
2. ADD an <actor_model> section that teaches the agent about delegation
3. If the identity mentions using tools directly (bash, file editing, etc.), reframe those as things to DELEGATE to subagents
4. If there's a tools/capabilities section, update it to reflect cortex-only tools
5. Keep the same voice, structure, and format as the original
6. Do NOT remove or rewrite sections that don't need changing
7. The <actor_model> section should go inside <purpose> if it exists, or at the end

Output ONLY the rewritten identity prompt, nothing else."""

TOOLS_REWRITE_PROMPT = """You are migrating an AI agent's tools configuration to support a new actor model architecture.

{actor_reference}

Here is the current tools block:

<current_tools>
{tools}
</current_tools>

Rewrite this tools block for the actor model. Rules:
1. Split tools into "Your Tools (cortex)" and "Subagent Tools"
2. Cortex tools: actor management, memory, telegram
3. Subagent default tools: bash, read_file, write_file, edit_file, list_directory, grep_search
4. Subagent extra tools: web_search, fetch_webpage, browser tools, memory tools
5. PRESERVE any custom tool documentation, skill references, or safety rules
6. Keep it concise and scannable

Output ONLY the rewritten tools block, nothing else."""


async def llm_rewrite(prompt: str, api_key: str, model: str = "anthropic/claude-haiku-4-5") -> str:
    """Call LLM to rewrite a prompt. Returns rewritten text."""
    try:
        import httpx
    except ImportError:
        print("  httpx not installed, falling back to template migration")
        return ""
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


# ─── Template-based fallback ─────────────────────────────────────────────

IDENTITY_ACTOR_SECTION = """
<actor_model>
## Actor Architecture

You operate as the **cortex** — the conscious executive layer of a multi-agent system.

### Your Role
- You are the ONLY agent that communicates with the user
- You are a COORDINATOR — you NEVER do work yourself
- For ANY task requiring tools (file editing, CLI, web search, etc.), spawn a subagent
- You keep: actor tools, memory tools, and telegram tools

### Your Agents
- **DMN** (Default Mode Network): Always-on background thinker. Scans goals, reflects,
  reorganizes memory, notifies you of urgent items. Runs automatically every 15 minutes.
- **Subagents**: Spawned on demand for specific tasks. They have file, CLI, web, and
  browser tools. They report results back to you.

### How to Delegate
1. `spawn_actor(name, goals, tools)` — be DETAILED in goals, the subagent only knows what you tell it
2. `ping_actor(id)` — check on progress
3. `kill_actor(id)` — terminate stuck agents
4. `wait_for_response(timeout)` — block until a reply arrives
5. `discover_actors(group)` — see who's running

### What You Keep
- Memory: `memory_read`, `memory_update`, `memory_append`, `archival_search/insert`, `conversation_search`
- Telegram: `telegram_send_message`, `telegram_send_file`
- Actors: `spawn_actor`, `kill_actor`, `ping_actor`, `send_message`, `discover_actors`, `wait_for_response`, `terminate`
</actor_model>
"""

TOOLS_ACTOR_TEMPLATE = """# Tools

## Your Tools (cortex)
- **spawn_actor** / **kill_actor** / **ping_actor** — Manage subagents
- **send_message** / **wait_for_response** / **discover_actors** — Actor communication
- **terminate** — End your own execution
- **memory_read** / **memory_update** / **memory_append** — Core memory blocks
- **archival_search** / **archival_insert** / **conversation_search** — Long-term memory
- **telegram_send_message** / **telegram_send_file** — Telegram I/O

## Subagent Default Tools (always available to spawned actors)
bash, read_file, write_file, edit_file, list_directory, grep_search

## Subagent Extra Tools (specify in spawn_actor tools= parameter)
web_search, fetch_webpage, browser_open, browser_click, browser_fill, browser_snapshot,
memory_read, memory_update, memory_append, archival_search, archival_insert, conversation_search

## Skills
Extended capabilities are documented as skill files in `~/lethe/skills/`.
Tell subagents to check `~/lethe/skills/` for relevant skill docs.
"""


# ─── Migration logic ─────────────────────────────────────────────────────

def backup(path: Path):
    """Back up a file before modifying it."""
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(f".pre-actors.{ts}.bak")
        shutil.copy2(path, backup_path)
        print(f"  Backed up: {path.name} -> {backup_path.name}")


def check_already_migrated(identity_path: Path) -> bool:
    """Check if already migrated (idempotent)."""
    if identity_path.exists():
        content = identity_path.read_text()
        if "<actor_model>" in content:
            return True
    return False


async def migrate_identity_llm(config_dir: Path, api_key: str):
    """Rewrite identity.md using LLM."""
    identity_path = config_dir / "identity.md"
    
    if not identity_path.exists():
        print(f"  WARNING: {identity_path} not found, skipping")
        return False
    
    content = identity_path.read_text()
    
    if "<actor_model>" in content:
        print("  identity.md: already has <actor_model> section, skipping")
        return True
    
    prompt = IDENTITY_REWRITE_PROMPT.format(
        actor_reference=ACTOR_MODEL_REFERENCE,
        identity=content,
    )
    
    print("  identity.md: rewriting with LLM...")
    result = await llm_rewrite(prompt, api_key)
    
    if not result or "<actor_model>" not in result:
        print("  WARNING: LLM output missing <actor_model> section, falling back to template")
        return False
    
    # Sanity check: result should be roughly similar length (not truncated or hallucinated)
    if len(result) < len(content) * 0.5:
        print(f"  WARNING: LLM output suspiciously short ({len(result)} vs {len(content)} chars), falling back")
        return False
    
    backup(identity_path)
    identity_path.write_text(result)
    print(f"  identity.md: rewritten ({len(content)} -> {len(result)} chars)")
    return True


async def migrate_tools_llm(config_dir: Path, api_key: str):
    """Rewrite tools.md using LLM."""
    tools_path = config_dir / "tools.md"
    
    if not tools_path.exists():
        print("  tools.md: not found, creating from template")
        return False
    
    content = tools_path.read_text()
    
    if "spawn_actor" in content:
        print("  tools.md: already actor-aware, skipping")
        return True
    
    prompt = TOOLS_REWRITE_PROMPT.format(
        actor_reference=ACTOR_MODEL_REFERENCE,
        tools=content,
    )
    
    print("  tools.md: rewriting with LLM...")
    result = await llm_rewrite(prompt, api_key)
    
    if not result or "spawn_actor" not in result:
        print("  WARNING: LLM output missing actor tools, falling back to template")
        return False
    
    backup(tools_path)
    tools_path.write_text(result)
    print(f"  tools.md: rewritten ({len(content)} -> {len(result)} chars)")
    return True


def migrate_identity_template(config_dir: Path):
    """Add actor model section to identity.md (template fallback)."""
    identity_path = config_dir / "identity.md"
    
    if not identity_path.exists():
        print(f"  WARNING: {identity_path} not found, skipping")
        return
    
    content = identity_path.read_text()
    
    if "<actor_model>" in content:
        print("  identity.md: already has <actor_model> section, skipping")
        return
    
    backup(identity_path)
    
    if "</purpose>" in content:
        content = content.replace("</purpose>", IDENTITY_ACTOR_SECTION + "\n</purpose>")
    else:
        content += "\n" + IDENTITY_ACTOR_SECTION
    
    identity_path.write_text(content)
    print("  identity.md: added <actor_model> section (template)")


def migrate_tools_template(config_dir: Path):
    """Replace tools.md with actor-aware version (template fallback)."""
    tools_path = config_dir / "tools.md"
    
    if tools_path.exists():
        content = tools_path.read_text()
        if "spawn_actor" in content:
            print("  tools.md: already actor-aware, skipping")
            return
        backup(tools_path)
    
    tools_path.write_text(TOOLS_ACTOR_TEMPLATE)
    print("  tools.md: replaced with actor-aware template")


def check_env(project_dir: Path):
    """Check ACTORS_ENABLED in .env."""
    env_path = project_dir / ".env"
    
    if env_path.exists():
        content = env_path.read_text()
        if "ACTORS_ENABLED" in content:
            if "ACTORS_ENABLED=true" in content.lower().replace(" ", ""):
                print("  .env: ACTORS_ENABLED=true ✓")
                return
            else:
                print("  .env: ACTORS_ENABLED exists but not 'true' — update manually")
                return
    
    print("  .env: Add ACTORS_ENABLED=true to enable actor model")


async def main_async(args):
    """Async main for LLM-powered migration."""
    config_dir = args.config_dir.resolve()
    project_dir = args.project_dir.resolve()
    
    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}")
        sys.exit(1)
    
    print("Migrating to actor model...")
    print(f"Config dir: {config_dir}")
    print()
    
    if check_already_migrated(config_dir / "identity.md"):
        print("Already migrated (identity.md has <actor_model> section).")
        print("Delete the <actor_model> section to re-migrate, or edit files manually.")
        return
    
    if args.dry_run:
        mode = "template" if args.no_llm else "LLM-powered"
        print(f"[DRY RUN] Would modify ({mode}):")
        print(f"  - {config_dir / 'identity.md'}: add <actor_model> section")
        print(f"  - {config_dir / 'tools.md'}: rewrite for actor model")
        print(f"  - Check .env for ACTORS_ENABLED")
        return
    
    # Try LLM-powered migration first
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    use_llm = api_key and not args.no_llm
    
    if use_llm:
        print("Using LLM-powered migration (Haiku 4.5 via OpenRouter)")
        print()
        
        identity_ok = await migrate_identity_llm(config_dir, api_key)
        if not identity_ok:
            migrate_identity_template(config_dir)
        
        tools_ok = await migrate_tools_llm(config_dir, api_key)
        if not tools_ok:
            migrate_tools_template(config_dir)
    else:
        if not api_key:
            print("No OPENROUTER_API_KEY found — using template migration")
        else:
            print("Template-only mode (--no-llm)")
        print()
        
        migrate_identity_template(config_dir)
        migrate_tools_template(config_dir)
    
    check_env(project_dir)
    
    print()
    print("Migration complete!")
    print()
    print("Next steps:")
    print("  1. Review config/blocks/identity.md — check the <actor_model> section")
    print("  2. Review config/blocks/tools.md — verify actor-aware tools list")
    print("  3. Ensure ACTORS_ENABLED=true in .env")
    print("  4. Restart: systemctl --user restart lethe")
    print()
    print("Backups saved as *.pre-actors.*.bak")


def main():
    parser = argparse.ArgumentParser(description="Migrate Lethe to actor model")
    parser.add_argument("--config-dir", type=Path, default=Path("./config/blocks"),
                        help="Path to config/blocks directory")
    parser.add_argument("--project-dir", type=Path, default=Path("."),
                        help="Path to project root")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying files")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use template-based migration only (no API calls)")
    args = parser.parse_args()
    
    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
