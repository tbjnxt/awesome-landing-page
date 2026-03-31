"""
QueryEngine — the agentic turn loop.

Mirrors claw-code's QueryEnginePort interface but drives a real
Ollama-backed model instead of the Anthropic API.

Each call to submit_message() runs the full tool-use loop:
  1. Send messages to local LLM
  2. If model requests tools → execute them → append results
  3. Repeat until model stops requesting tools or turn limit hit
  4. Return final TurnResult
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

from .llm_backend import OllamaBackend
from .models import AgentConfig, PermissionDenial, ToolCall, ToolResult, UsageSummary
from .query import QueryRequest, QueryResponse
from .tool_executor import dispatch_tool


# ── Config & per-turn result ─────────────────────────────────────────────────

@dataclass
class QueryEngineConfig:
    max_turns: int = 10
    max_budget_tokens: int = 32_000
    tool_use_enabled: bool = True
    stream: bool = False


@dataclass
class TurnResult:
    prompt: str
    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    permission_denials: list[PermissionDenial] = field(default_factory=list)
    usage: UsageSummary = field(default_factory=UsageSummary)
    stop_reason: str = "end_turn"
    turn_number: int = 0
    elapsed_seconds: float = 0.0

    def as_markdown(self) -> str:
        lines = [
            f"## Turn {self.turn_number}",
            f"**Prompt:** {self.prompt[:120]}{'…' if len(self.prompt) > 120 else ''}",
            f"**Stop reason:** {self.stop_reason}",
            f"**Tokens:** {self.usage.input_tokens} in / {self.usage.output_tokens} out",
            f"**Time:** {self.elapsed_seconds:.2f}s",
        ]
        if self.tool_calls:
            lines.append("\n**Tool calls:**")
            for tc in self.tool_calls:
                lines.append(f"  - `{tc.tool_name}` {json.dumps(tc.arguments)[:80]}")
        if self.tool_results:
            lines.append("\n**Tool results:**")
            for tr in self.tool_results:
                status = "✓" if tr.success else "✗"
                preview = (tr.output or tr.error or "")[:120]
                lines.append(f"  {status} `{tr.tool_name}`: {preview}")
        if self.output:
            lines.append(f"\n**Output:**\n{self.output[:500]}{'…' if len(self.output) > 500 else ''}")
        return "\n".join(lines)


# ── Session state ────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """Persistent conversation state across turns."""
    messages: list[dict] = field(default_factory=list)
    turn_count: int = 0
    total_usage: UsageSummary = field(default_factory=UsageSummary)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, tool_calls: Optional[list[dict]] = None) -> None:
        msg: dict = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, call_id: str, tool_name: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "content": content,
            "name": tool_name,
        })

    def compact(self, keep_last: int = 20) -> None:
        """Trim message history, keeping system prompt + last N messages."""
        system = [m for m in self.messages if m["role"] == "system"]
        non_system = [m for m in self.messages if m["role"] != "system"]
        if len(non_system) > keep_last:
            self.messages = system + non_system[-keep_last:]


# ── QueryEnginePort (matches claw-code interface) ───────────────────────────

class QueryEnginePort:
    """
    Base class — same interface as claw-code's QueryEnginePort.
    Subclass with OllamaQueryEngine to get real LLM execution.
    """

    def __init__(self, config: Optional[QueryEngineConfig] = None):
        self.config = config or QueryEngineConfig()
        self._state = SessionState()

    def submit_message(self, prompt: str) -> TurnResult:
        raise NotImplementedError

    def stream_submit_message(self, prompt: str) -> Generator[str, None, None]:
        raise NotImplementedError

    def persist_session(self) -> dict:
        return {
            "messages": self._state.messages,
            "turn_count": self._state.turn_count,
        }

    def from_saved_session(self, data: dict) -> None:
        self._state.messages = data.get("messages", [])
        self._state.turn_count = data.get("turn_count", 0)

    def summary_report(self) -> str:
        s = self._state
        return (
            f"Session: {s.turn_count} turns | "
            f"{s.total_usage.input_tokens} input tokens | "
            f"{s.total_usage.output_tokens} output tokens"
        )


# ── Concrete Ollama-backed implementation ────────────────────────────────────

class OllamaQueryEngine(QueryEnginePort):
    """
    Full agentic query engine backed by Ollama.

    Runs the tool-use loop locally — no internet required after `ollama pull`.
    """

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        engine_config: Optional[QueryEngineConfig] = None,
    ):
        super().__init__(engine_config)
        self.agent_config = agent_config or AgentConfig()
        self.backend = OllamaBackend(self.agent_config)
        # Prime the conversation with the system prompt
        self._state.messages = [
            {"role": "system", "content": self.agent_config.system_prompt}
        ]

    # ── Public API ──────────────────────────────────────────────────────────

    def submit_message(self, prompt: str) -> TurnResult:
        """
        Submit a user message and run the tool loop until completion.
        Returns the final TurnResult.
        """
        start = time.monotonic()
        self._state.turn_count += 1
        self._state.add_user(prompt)

        all_tool_calls: list[ToolCall] = []
        all_tool_results: list[ToolResult] = []
        final_text = ""
        stop_reason = "end_turn"

        # Inner tool loop — runs until model stops requesting tools
        for inner_turn in range(20):
            text, tool_calls = self.backend.complete(
                messages=self._state.messages,
                use_tools=self.config.tool_use_enabled,
            )

            if not tool_calls:
                # Model finished — no more tool calls
                final_text = text
                self._state.add_assistant(text)
                stop_reason = "end_turn"
                break

            # Model requested tools — record the assistant message
            raw_tcs = [
                {"function": {"name": tc.tool_name, "arguments": tc.arguments}}
                for tc in tool_calls
            ]
            self._state.add_assistant(text, tool_calls=raw_tcs)
            all_tool_calls.extend(tool_calls)

            # Execute each tool
            for tc in tool_calls:
                result = dispatch_tool(tc)
                all_tool_results.append(result)
                content = result.output if result.success else f"ERROR: {result.error}"
                self._state.add_tool_result(tc.call_id, tc.tool_name, content)
        else:
            stop_reason = "max_tool_iterations"

        # Compact history if it's getting long
        if len(self._state.messages) > 60:
            self._state.compact(keep_last=40)

        elapsed = time.monotonic() - start
        usage = UsageSummary().add_turn(prompt, final_text)
        self._state.total_usage = self._state.total_usage.add_turn(prompt, final_text)

        return TurnResult(
            prompt=prompt,
            output=final_text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            usage=usage,
            stop_reason=stop_reason,
            turn_number=self._state.turn_count,
            elapsed_seconds=elapsed,
        )

    def stream_submit_message(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream tokens from the model as they arrive.
        Tool calls are executed silently; only final text is yielded.
        """
        # Check for tool calls first (non-streaming pass)
        self._state.add_user(prompt)

        for inner_turn in range(20):
            text, tool_calls = self.backend.complete(
                messages=self._state.messages,
                use_tools=self.config.tool_use_enabled,
            )
            if not tool_calls:
                # Final response — stream it token by token (simulate from full text)
                self._state.add_assistant(text)
                words = text.split(" ")
                for word in words:
                    yield word + " "
                break

            raw_tcs = [
                {"function": {"name": tc.tool_name, "arguments": tc.arguments}}
                for tc in tool_calls
            ]
            self._state.add_assistant(text, tool_calls=raw_tcs)
            for tc in tool_calls:
                yield f"\n[tool: {tc.tool_name}]\n"
                result = dispatch_tool(tc)
                content = result.output if result.success else f"ERROR: {result.error}"
                self._state.add_tool_result(tc.call_id, tc.tool_name, content)

        self._state.turn_count += 1
