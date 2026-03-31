"""
Tool executor — actual implementations of BashTool, FileReadTool, FileEditTool, etc.

These are the real working tools, not stubs. They execute locally with no network.
"""

from __future__ import annotations

import fnmatch
import glob as _glob
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .models import ToolCall, ToolResult


# ── Safety helpers ──────────────────────────────────────────────────────────

_BLOCKED_COMMANDS = frozenset(
    [
        "rm -rf /",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",  # fork bomb
        "chmod -R 777 /",
    ]
)


def _is_safe_command(cmd: str) -> tuple[bool, str]:
    lower = cmd.lower().strip()
    for blocked in _BLOCKED_COMMANDS:
        if blocked in lower:
            return False, f"Blocked: '{blocked}' is not allowed."
    return True, ""


# ── Individual tool implementations ─────────────────────────────────────────

def execute_bash(command: str, timeout: int = 30, cwd: Optional[str] = None) -> ToolResult:
    safe, reason = _is_safe_command(command)
    if not safe:
        return ToolResult(call_id="", tool_name="bash", output="", error=reason)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or os.getcwd(),
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output = output + result.stderr if output else result.stderr
        return ToolResult(
            call_id="",
            tool_name="bash",
            output=output or "(no output)",
            error=None if result.returncode == 0 else f"exit code {result.returncode}",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(call_id="", tool_name="bash", output="", error=f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult(call_id="", tool_name="bash", output="", error=str(e))


def execute_read_file(
    path: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> ToolResult:
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(call_id="", tool_name="read_file", output="", error=f"File not found: {path}")
        if not p.is_file():
            return ToolResult(call_id="", tool_name="read_file", output="", error=f"Not a file: {path}")

        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

        start = (offset - 1) if offset and offset > 0 else 0
        end = (start + limit) if limit else len(lines)
        selected = lines[start:end]

        # Add line numbers
        numbered = "".join(
            f"{start + i + 1}\t{line}" for i, line in enumerate(selected)
        )
        return ToolResult(call_id="", tool_name="read_file", output=numbered or "(empty file)")
    except Exception as e:
        return ToolResult(call_id="", tool_name="read_file", output="", error=str(e))


def execute_edit_file(path: str, old_string: str, new_string: str) -> ToolResult:
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(call_id="", tool_name="edit_file", output="", error=f"File not found: {path}")

        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)

        if count == 0:
            return ToolResult(
                call_id="", tool_name="edit_file", output="",
                error=f"old_string not found in {path}. Verify the exact text with read_file first.",
            )
        if count > 1:
            return ToolResult(
                call_id="", tool_name="edit_file", output="",
                error=f"old_string appears {count} times in {path}. Provide more context to make it unique.",
            )

        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return ToolResult(call_id="", tool_name="edit_file", output=f"Edited {path} successfully.")
    except Exception as e:
        return ToolResult(call_id="", tool_name="edit_file", output="", error=str(e))


def execute_write_file(path: str, content: str) -> ToolResult:
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        lines = content.count("\n") + 1
        return ToolResult(call_id="", tool_name="write_file", output=f"Wrote {lines} lines to {path}.")
    except Exception as e:
        return ToolResult(call_id="", tool_name="write_file", output="", error=str(e))


def execute_glob(pattern: str, directory: Optional[str] = None) -> ToolResult:
    try:
        base = directory or os.getcwd()
        full_pattern = os.path.join(base, pattern) if not os.path.isabs(pattern) else pattern
        matches = sorted(_glob.glob(full_pattern, recursive=True))
        if not matches:
            return ToolResult(call_id="", tool_name="glob", output="(no matches)")
        return ToolResult(call_id="", tool_name="glob", output="\n".join(matches))
    except Exception as e:
        return ToolResult(call_id="", tool_name="glob", output="", error=str(e))


def execute_grep(pattern: str, path: Optional[str] = None, file_pattern: Optional[str] = None) -> ToolResult:
    try:
        search_path = path or os.getcwd()
        cmd_parts = ["grep", "-rn", "--color=never"]

        if file_pattern:
            cmd_parts += ["--include", file_pattern]

        cmd_parts += [pattern, search_path]

        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        if not output and result.returncode != 0:
            return ToolResult(call_id="", tool_name="grep", output="(no matches)")
        return ToolResult(call_id="", tool_name="grep", output=output or "(no matches)")
    except Exception as e:
        return ToolResult(call_id="", tool_name="grep", output="", error=str(e))


# ── Dispatcher ───────────────────────────────────────────────────────────────

def dispatch_tool(call: ToolCall) -> ToolResult:
    """Route a ToolCall to the appropriate executor and return a ToolResult."""
    args = call.arguments

    if call.tool_name == "bash":
        result = execute_bash(
            command=args.get("command", ""),
            timeout=int(args.get("timeout", 30)),
        )
    elif call.tool_name == "read_file":
        result = execute_read_file(
            path=args.get("path", ""),
            offset=args.get("offset"),
            limit=args.get("limit"),
        )
    elif call.tool_name == "edit_file":
        result = execute_edit_file(
            path=args.get("path", ""),
            old_string=args.get("old_string", ""),
            new_string=args.get("new_string", ""),
        )
    elif call.tool_name == "write_file":
        result = execute_write_file(
            path=args.get("path", ""),
            content=args.get("content", ""),
        )
    elif call.tool_name == "glob":
        result = execute_glob(
            pattern=args.get("pattern", ""),
            directory=args.get("directory"),
        )
    elif call.tool_name == "grep":
        result = execute_grep(
            pattern=args.get("pattern", ""),
            path=args.get("path"),
            file_pattern=args.get("file_pattern"),
        )
    else:
        result = ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output="",
            error=f"Unknown tool: {call.tool_name}",
        )

    # carry through the call_id
    return ToolResult(
        call_id=call.call_id,
        tool_name=result.tool_name,
        output=result.output,
        error=result.error,
    )
