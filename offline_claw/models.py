"""Core dataclasses — ported from claw-code's models.py with offline extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Subsystem:
    name: str
    path: str
    file_count: int
    notes: str


@dataclass(frozen=True)
class PortingModule:
    name: str
    responsibility: str
    source_hint: str
    status: str = "mirrored"


@dataclass(frozen=True)
class PermissionDenial:
    tool_name: str
    reason: str


@dataclass(frozen=True)
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0

    def add_turn(self, prompt: str, output: str) -> "UsageSummary":
        return UsageSummary(
            input_tokens=self.input_tokens + len(prompt.split()),
            output_tokens=self.output_tokens + len(output.split()),
        )


@dataclass
class PortingBacklog:
    title: str
    modules: list[PortingModule] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        return [
            f"- {m.name} [{m.status}] — {m.responsibility} (from {m.source_hint})"
            for m in self.modules
        ]


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the LLM."""
    tool_name: str
    arguments: dict
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool."""
    call_id: str
    tool_name: str
    output: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class AgentConfig:
    """Runtime configuration for the offline agent."""
    model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
    max_turns: int = 10
    max_tokens: int = 4096
    temperature: float = 0.2
    system_prompt: str = (
        "You are an expert software engineering assistant. "
        "You have access to tools for reading files, editing files, and running bash commands. "
        "Use these tools to complete tasks efficiently. "
        "Be precise and careful with file edits. Always verify your work."
    )
