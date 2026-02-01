"""Tools for the Lethe agent.

Provides file, CLI, and other tools that can be registered with the agent.
"""

from lethe.tools.cli import (
    bash,
    bash_output,
    get_terminal_screen,
    send_terminal_input,
    kill_bash,
    get_environment_info,
    check_command_exists,
)

from lethe.tools.filesystem import (
    read_file,
    write_file,
    edit_file,
    list_directory,
    glob_search,
    grep_search,
)


def register_tools(agent):
    """Register all external tools with an agent.
    
    Args:
        agent: Agent instance with register_tool method
    """
    # Bash / CLI tools
    agent.register_tool(
        "bash",
        bash,
        {
            "name": "bash",
            "description": "Execute a bash command. Use run_in_background=True for long-running commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 600)"},
                    "description": {"type": "string", "description": "Short description of the command"},
                    "run_in_background": {"type": "boolean", "description": "Run in background and return ID"},
                    "use_pty": {"type": "boolean", "description": "Use PTY for TUI apps (htop, vim)"},
                },
                "required": ["command"]
            }
        }
    )
    
    agent.register_tool(
        "bash_output",
        bash_output,
        {
            "name": "bash_output",
            "description": "Get output from a background bash process",
            "parameters": {
                "type": "object",
                "properties": {
                    "shell_id": {"type": "string", "description": "Background shell ID (e.g., bash_1)"},
                    "filter_pattern": {"type": "string", "description": "Filter output lines"},
                    "last_lines": {"type": "integer", "description": "Only return last N lines"},
                },
                "required": ["shell_id"]
            }
        }
    )
    
    agent.register_tool(
        "kill_bash",
        kill_bash,
        {
            "name": "kill_bash",
            "description": "Kill a background bash process",
            "parameters": {
                "type": "object",
                "properties": {
                    "shell_id": {"type": "string", "description": "Background shell ID to kill"},
                },
                "required": ["shell_id"]
            }
        }
    )
    
    # File tools
    agent.register_tool(
        "read_file",
        read_file,
        {
            "name": "read_file",
            "description": "Read a file from the filesystem with line numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to read"},
                    "offset": {"type": "integer", "description": "Starting line (0-indexed)"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["file_path"]
            }
        }
    )
    
    agent.register_tool(
        "write_file",
        write_file,
        {
            "name": "write_file",
            "description": "Write content to a file (creates parent dirs if needed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["file_path", "content"]
            }
        }
    )
    
    agent.register_tool(
        "edit_file",
        edit_file,
        {
            "name": "edit_file",
            "description": "Edit a file by replacing text",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to edit"},
                    "old_string": {"type": "string", "description": "Text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    )
    
    agent.register_tool(
        "list_directory",
        list_directory,
        {
            "name": "list_directory",
            "description": "List directory contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "show_hidden": {"type": "boolean", "description": "Include hidden files"},
                },
                "required": []
            }
        }
    )
    
    agent.register_tool(
        "glob_search",
        glob_search,
        {
            "name": "glob_search",
            "description": "Find files matching a glob pattern (e.g., **/*.py)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Base directory"},
                },
                "required": ["pattern"]
            }
        }
    )
    
    agent.register_tool(
        "grep_search",
        grep_search,
        {
            "name": "grep_search",
            "description": "Search for regex pattern in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Directory to search"},
                    "file_pattern": {"type": "string", "description": "File glob filter"},
                },
                "required": ["pattern"]
            }
        }
    )


__all__ = [
    "register_tools",
    "bash",
    "bash_output",
    "get_terminal_screen",
    "send_terminal_input",
    "kill_bash",
    "get_environment_info",
    "check_command_exists",
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "glob_search",
    "grep_search",
]
