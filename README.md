# Lethe

Autonomous executive assistant with persistent memory.

Lethe is a 24/7 AI assistant that you communicate with via Telegram. It remembers everything - your preferences, your projects, conversations from months ago. The more you use it, the more useful it becomes.

**This branch (`lethe2`) is a complete rewrite** - local-first, no cloud dependencies except the LLM API.

## Architecture

```
Telegram Bot → Conversation Manager → Agent → LLM (OpenRouter)
     ↑              (debounce,           ↓
     │              interrupts)       Tools (bash, files, browser, web search)
     │                                   ↓
     └─────────────────────────────── Memory (LanceDB)
                                        ├── blocks (files in workspace/)
                                        ├── archival (vector + FTS)
                                        └── messages (conversation history)
```

### Key Differences from `main` Branch

| Feature | `lethe2` (this branch) | `main` (Letta Cloud) |
|---------|------------------------|----------------------|
| **LLM** | Direct OpenRouter API | Letta Cloud |
| **Memory** | Local LanceDB + files | Letta Cloud |
| **Tools** | Execute directly | Approval loops |
| **Latency** | Fast (~2-5s) | Slow (30-60s) |
| **Dependencies** | Minimal | Letta server |
| **Cost** | Pay per token only | Letta + token costs |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Telegram bot token from [@BotFather](https://t.me/BotFather)
- [OpenRouter](https://openrouter.ai/) API key

### 2. Install

```bash
git clone https://github.com/atemerev/lethe.git
cd lethe
git checkout lethe2
uv sync
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_USER_IDS=your_telegram_id

OPENROUTER_API_KEY=sk-or-...

# Optional: Exa web search
EXA_API_KEY=your_exa_key
```

### 4. Run

```bash
uv run lethe
```

### 5. (Optional) Run as Service

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/lethe.service << EOF
[Unit]
Description=Lethe Autonomous AI Agent
After=network.target

[Service]
Type=simple
WorkingDirectory=$(pwd)
ExecStart=$(which uv) run lethe
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now lethe
```

## Memory System

### Memory Blocks (Core Memory)

Always in context. Stored as editable files in `workspace/memory/`:

```
workspace/memory/
├── persona.md      # Who the agent is
├── human.md        # What it knows about you
├── project.md      # Current project context
├── tools.md        # Available CLI tools
└── tasks.md        # Active tasks/reminders
```

Edit these files directly - changes are picked up on next message.

### Archival Memory

Long-term semantic storage with hybrid search (vector + full-text). Used for:
- Facts and learnings
- Detailed information that doesn't fit in blocks
- Searchable via `archival_search` tool

### Message History

Conversation history stored locally. Searchable via `conversation_search` tool.

## Tools

### Filesystem
- `read_file` - Read with smart truncation (2000 lines / 50KB)
- `write_file` - Create/overwrite files
- `edit_file` - Replace text in files
- `list_directory` - List directory contents
- `glob_search` - Find files by pattern
- `grep_search` - Search file contents (500 char line limit)

### CLI
- `bash` - Execute commands with tail truncation (keeps errors visible)
- `bash_output` - Get output from background processes
- `kill_bash` - Terminate background processes

### Browser (via agent-browser)
- `browser_open` - Open URL
- `browser_snapshot` - Get accessibility tree with refs (@e1, @e2...)
- `browser_click` - Click element by ref
- `browser_fill` - Fill input field
- `send_image` - Send image to user (maintains message order)

### Web Search (optional, requires EXA_API_KEY)
- `web_search` - AI-powered semantic search
- `fetch_webpage` - Extract clean text from URL

### Memory
- `memory_read` - Read a memory block
- `memory_update` - Update a memory block
- `memory_append` - Append to a memory block
- `archival_search` - Search long-term memory
- `archival_insert` - Store in long-term memory
- `conversation_search` - Search message history

## Conversation Manager

Handles async message processing with:

- **No debounce on first message** - Responds immediately
- **Debounce on interrupt** - Waits 5s for follow-up messages
- **Message batching** - Combines rapid messages into one
- **Interrupt handling** - New messages during processing are queued

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather | (required) |
| `TELEGRAM_ALLOWED_USER_IDS` | Comma-separated user IDs | (required) |
| `OPENROUTER_API_KEY` | OpenRouter API key | (required) |
| `EXA_API_KEY` | Exa web search API key | (optional) |
| `LLM_MODEL` | Model to use | `moonshotai/kimi-k2.5` |
| `LLM_CONTEXT_LIMIT` | Context window size | `131072` |
| `WORKSPACE_DIR` | Agent workspace | `./workspace` |
| `MEMORY_DIR` | Memory data storage | `./data/memory` |

### Identity Configuration

Edit `config/identity.md` to customize the agent's persona. This is loaded as the system prompt.

## Development

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_tools.py -v

# Run with coverage
uv run pytest --cov=lethe
```

### Test Coverage

- `test_tools.py` - 51 tests (filesystem, CLI, browser, web search)
- `test_blocks.py` - 15 tests (file-based memory blocks)
- `test_truncate.py` - 20 tests (smart truncation utilities)
- `test_conversation.py` - 16 tests (conversation manager)

## Adding Custom Tools

Create a function with the `@_is_tool` decorator:

```python
def _is_tool(func):
    func._is_tool = True
    return func

@_is_tool
def my_tool(arg1: str, arg2: int = 10) -> str:
    """Description of what the tool does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        What the tool returns
    """
    return "result"
```

Add to `src/lethe/tools/__init__.py`.

## License

MIT
