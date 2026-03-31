"""
Command registry — mirrors claw-code's commands.py.

Loads command metadata from the snapshot JSON and provides
lookup/filtering helpers used by the runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .models import PermissionDenial, PortingModule


_SNAPSHOT_PATH = Path(__file__).parent / "reference_data" / "commands_snapshot.json"


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

def get_commands(query: str = "") -> list[PortingModule]:
    cmds = _load_snapshot()
    if query:
        q = query.lower()
        cmds = [c for c in cmds if q in c.name.lower() or q in c.source_hint.lower()]
    return cmds


def get_command(name: str) -> PortingModule | None:
    name_lower = name.lower()
    for cmd in _load_snapshot():
        if cmd.name.lower() == name_lower:
            return cmd
    return None


def find_commands(query: str) -> list[PortingModule]:
    q = query.lower().split()
    results = []
    for cmd in _load_snapshot():
        score = sum(
            1
            for token in q
            if token in cmd.name.lower() or token in cmd.source_hint.lower()
        )
        if score > 0:
            results.append((score, cmd))
    results.sort(key=lambda x: -x[0])
    return [cmd for _, cmd in results]


@dataclass(frozen=True)
class CommandExecution:
    found: bool
    name: str
    action: str
    denial: PermissionDenial | None = None


def execute_command(name: str) -> CommandExecution:
    cmd = get_command(name)
    if not cmd:
        return CommandExecution(found=False, name=name, action="not_found")
    return CommandExecution(
        found=True,
        name=cmd.name,
        action=f"routed:{cmd.source_hint}",
    )


def render_command_index(query: str = "") -> str:
    cmds = get_commands(query)
    if not cmds:
        return f"No commands matching '{query}'."
    lines = [f"## Commands ({len(cmds)} total)\n"]
    seen: set[str] = set()
    for cmd in cmds:
        if cmd.name not in seen:
            lines.append(f"- **{cmd.name}** — {cmd.source_hint}")
            seen.add(cmd.name)
    return "\n".join(lines)
