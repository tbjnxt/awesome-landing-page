"""
Tool registry — mirrors claw-code's tools.py.

Loads tool metadata from snapshot JSON and tracks which tools are
available vs. restricted in the offline runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .models import PermissionDenial, PortingModule


_SNAPSHOT_PATH = Path(__file__).parent / "reference_data" / "tools_snapshot.json"

# Tools that are fully implemented in offline_claw
IMPLEMENTED_TOOLS = frozenset(
    ["bash", "read_file", "edit_file", "write_file", "glob", "grep"]
)

# Original tool names → offline_claw executor names
_TOOL_NAME_MAP = {
    "BashTool": "bash",
    "FileReadTool": "read_file",
    "FileEditTool": "edit_file",
    "FileWriteTool": "write_file",
    "GlobTool": "glob",
    "GrepTool": "grep",
}


@lru_cache(maxsize=1)
def _load_snapshot() -> list[PortingModule]:
    data = json.loads(_SNAPSHOT_PATH.read_text())
    return [
        PortingModule(
            name=item["name"],
            responsibility=item["responsibility"],
            source_hint=item["source_hint"],
            status="mirrored",
        )
        for item in data
    ]


# ── Public API ───────────────────────────────────────────────────────────────

def get_tools(simple: bool = False) -> list[PortingModule]:
    tools = _load_snapshot()
    if simple:
        core = {"BashTool", "FileReadTool", "FileEditTool"}
        tools = [t for t in tools if t.name in core]
    return tools


def get_tool(name: str) -> PortingModule | None:
    name_lower = name.lower()
    for tool in _load_snapshot():
        if tool.name.lower() == name_lower:
            return tool
    return None


def tool_names() -> list[str]:
    seen: set[str] = set()
    names = []
    for t in _load_snapshot():
        if t.name not in seen:
            names.append(t.name)
            seen.add(t.name)
    return names


def find_tools(query: str) -> list[PortingModule]:
    q = query.lower().split()
    results = []
    for tool in _load_snapshot():
        score = sum(
            1
            for token in q
            if token in tool.name.lower() or token in tool.source_hint.lower()
        )
        if score > 0:
            results.append((score, tool))
    results.sort(key=lambda x: -x[0])
    return [t for _, t in results]


def is_implemented(tool_name: str) -> bool:
    """True if this tool has a working offline executor."""
    executor_name = _TOOL_NAME_MAP.get(tool_name, tool_name.lower())
    return executor_name in IMPLEMENTED_TOOLS


@dataclass(frozen=True)
class ToolExecution:
    found: bool
    name: str
    implemented: bool
    executor: str
    denial: PermissionDenial | None = None


def execute_tool(name: str) -> ToolExecution:
    tool = get_tool(name)
    if not tool:
        return ToolExecution(found=False, name=name, implemented=False, executor="")
    executor = _TOOL_NAME_MAP.get(name, name.lower())
    impl = executor in IMPLEMENTED_TOOLS
    return ToolExecution(
        found=True,
        name=name,
        implemented=impl,
        executor=executor if impl else "(stub)",
    )


def render_tool_index(query: str = "") -> str:
    tools = find_tools(query) if query else get_tools()
    if not tools:
        return f"No tools matching '{query}'."
    lines = [f"## Tools ({len(tools)} total)\n"]
    seen: set[str] = set()
    for t in tools:
        if t.name not in seen:
            status = "✓ offline" if is_implemented(t.name) else "· mirrored"
            lines.append(f"- **{t.name}** [{status}] — {t.source_hint}")
            seen.add(t.name)
    return "\n".join(lines)
