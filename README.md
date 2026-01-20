# Lethe

Autonomous executive assistant with Letta memory layer.

Lethe is a 24/7 AI assistant that you communicate with via Telegram. It processes tasks asynchronously, maintains persistent memory across conversations, and has full access to your machine.

> **⚠️ WARNING: NO SANDBOXING**
> 
> Lethe has **unrestricted access** to your system with the same rights as the user running it. It can read, write, and delete any files you can access, execute any commands, and browse the web. There is **no sandboxing or isolation**. 
>
> Only run Lethe on systems where you trust the agent with full access. Consider running in a VM or container if you need isolation.

## Self-Modifying Agent

Lethe uses Letta's persistent memory, which means:

- **Teach it tools**: Show the agent how to use CLI tools, and it will remember for future sessions
- **Shape its identity**: Ask it to update its persona, communication style, or areas of expertise
- **Project context**: It maintains notes about your current projects and priorities
- **Learn your preferences**: It remembers how you like things done

The agent can modify its own memory blocks (`persona`, `human`, `project`, `tasks`, `tools`) using `core_memory_append` and `core_memory_replace`, and store unlimited long-term memories in archival storage.

## Proactive & Persistent

Unlike typical chatbots that wait for input, Lethe is designed to be genuinely autonomous:

- **Background Task Processing**: Handles long-running tasks asynchronously while staying responsive to new requests
- **Periodic Heartbeats**: Checks in every 15 minutes to review memory, conversation history, and proactively surface reminders or follow-ups
- **Self-Improvement**: Continuously learns from interactions, updating its own memory blocks and tool knowledge
- **Full System Access**: Can execute shell commands, manage files, browse the web, and interact with CLI tools—all without waiting for permission
- **Task Persistence**: Remembers commitments across sessions and follows up on promises made in previous conversations

The agent doesn't just respond—it actively manages ongoing work, reminds you of important items, and suggests next steps based on your project context.

## Heartbeats

Lethe performs periodic check-ins (default: every 15 minutes) to maintain continuity and proactivity:

**What happens during a heartbeat:**
1. Reviews all memory blocks for pending tasks or reminders
2. Scans recent conversation history for follow-up items
3. Considers time-sensitive actions or proactive suggestions
4. Sends a notification only if there's something genuinely useful to report

**Identity Refresh:**
Every 2 hours, the agent refreshes its identity context to stay aligned with its current persona and project state.

Heartbeats ensure that even if you're away, Lethe stays aware of commitments and can surface them when relevant—turning a reactive chatbot into an active assistant.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Telegram   │────▶│  Task Queue │────▶│   Worker    │
│    Bot      │     │  (SQLite)   │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Letta     │
                                        │   Agent     │
                                        │  (memory +  │
                                        │  reasoning) │
                                        └──────┬──────┘
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         │                     │                     │
                         ▼                     ▼                     ▼
                  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
                  │ Filesystem  │       │    CLI      │       │   Browser   │
                  │   Tools     │       │   Tools     │       │   Tools     │
                  └─────────────┘       └─────────────┘       └─────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- [Letta Cloud](https://app.letta.com) API key (free tier available) or local Letta server
- Telegram bot token from [@BotFather](https://t.me/BotFather)

### 2. Install

```bash
cd lethe
uv sync
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your settings:
# - TELEGRAM_BOT_TOKEN (required)
# - TELEGRAM_ALLOWED_USER_IDS (your Telegram user ID)
# - LETTA_API_KEY (get from https://app.letta.com)
```

### 4. Run Lethe

```bash
uv run lethe
# or
uv run python -m lethe.main
```

### (Optional) Use Local Letta Server

If you prefer to run Letta locally instead of using Letta Cloud:

```bash
# Start Letta server (Docker)
docker run -d -p 8283:8283 -v letta-data:/root/.letta letta/letta:latest

# Or via pip
pip install letta
letta server
```

Then set in `.env`:
```bash
LETTA_BASE_URL=http://localhost:8283
# LETTA_API_KEY not needed for local server
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather | (required) |
| `TELEGRAM_ALLOWED_USER_IDS` | Comma-separated user IDs | (empty = all) |
| `LETTA_API_KEY` | API key from [Letta Cloud](https://app.letta.com) | (required for cloud) |
| `LETTA_BASE_URL` | Letta server URL (for local server) | `https://api.letta.com` |
| `LETHE_AGENT_NAME` | Agent name in Letta | `lethe` |
| `LETHE_CONFIG_DIR` | Path to config files | `./config` |
| `DB_PATH` | SQLite database path | `./data/lethe.db` |

### Config Files (Initial Setup Only)

These files are loaded **only when the agent is first created**. After that, the agent's memory persists on Letta's servers and the config files are ignored. The agent can modify its own memory blocks at any time.

- `config/identity.md` - Agent persona and capabilities → `persona` memory block
- `config/project.md` - Current project context → `project` memory block
- `config/tools.md` - CLI tools documentation → `tools` memory block

To reset the agent to config file defaults, delete it from Letta and restart Lethe.

## Tools

The agent has access to:

### Filesystem
- `read_file` - Read files with line numbers
- `write_file` - Create/overwrite files
- `edit_file` - Replace text in files
- `list_directory` - List directory contents
- `glob_search` - Find files by pattern
- `grep_search` - Search file contents

### CLI
- `bash` - Execute shell commands with timeout support
- `bash_output` - Get output from background processes
- `kill_bash` - Terminate background processes
- `get_environment_info` - Get system/environment info
- `check_command_exists` - Check if a command is available

### Browser (requires Steel)
- `browser_navigate` - Navigate to a URL
- `browser_get_context` - Get page context (accessibility tree, token-efficient)
- `browser_get_text` - Get all visible text from page
- `browser_click` - Click elements by selector or text
- `browser_fill` - Fill input fields
- `browser_extract_text` - Extract text from specific element
- `browser_screenshot` - Take screenshots
- `browser_scroll` - Scroll the page
- `browser_wait_for` - Wait for elements to appear
- `browser_close` - Close browser session

### Telegram
- `telegram_send_message` - Send additional message to user
- `telegram_send_file` - Send file to user

### Letta Built-in
- `core_memory_append` - Append text to a memory block (persona, human, project, tasks, tools)
- `core_memory_replace` - Replace text in a memory block
- `archival_memory_insert` - Store facts/learnings in long-term semantic memory
- `archival_memory_search` - Search long-term memory by semantic similarity
- `web_search` - Search the web for information
- `fetch_webpage` - Fetch and read a webpage

## Browser Setup (Steel)

Browser tools require Steel Browser running locally:

```bash
# Option 1: Docker Compose (recommended)
docker compose -f docker-compose.steel.yml up -d

# Option 2: Direct Docker
docker run -d -p 3000:3000 -p 9223:9223 ghcr.io/steel-dev/steel-browser

# Install browser dependencies
uv sync --extra browser
playwright install chromium
```

Steel provides:
- Persistent sessions (cookies, localStorage across requests)
- Anti-bot protection and stealth
- Session viewer at http://localhost:3000/ui

The browser tools use the **Accessibility Tree** instead of raw DOM, reducing context sent to the LLM by ~90% while preserving semantic meaning.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Format/lint
uv run ruff check --fix
```

## Adding Custom Tools

Create a new file in `src/lethe/tools/` and add tools using the `@_is_tool` decorator:

```python
def _is_tool(func):
    func._is_tool = True
    return func

@_is_tool
def my_custom_tool(arg1: str, arg2: int = 10) -> str:
    """Description of what the tool does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        What the tool returns
    """
    # Implementation
    return "result"
```

Then import the module in `src/lethe/tools/__init__.py`.

## License

MIT
