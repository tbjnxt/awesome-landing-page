"""
PortRuntime — prompt routing and session orchestration.

Mirrors claw-code's runtime.py with enhancements for offline execution.
Routes user prompts to matching commands/tools and drives the query engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .commands import find_commands, execute_command
from .models import AgentConfig, PermissionDenial, PortingModule, UsageSummary
from .query_engine import OllamaQueryEngine, QueryEngineConfig, TurnResult
from .tools import find_tools, execute_tool


# ── Routing structures ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoutedMatch:
    kind: str           # "command" or "tool"
    name: str
    source_hint: str
    score: float


@dataclass
class RuntimeSession:
    prompt: str
    context: dict = field(default_factory=dict)
    routed_matches: list[RoutedMatch] = field(default_factory=list)
    permission_denials: list[PermissionDenial] = field(default_factory=list)
    turn_results: list[TurnResult] = field(default_factory=list)
    setup_ok: bool = True

    def as_markdown(self) -> str:
        lines = [
            "# Runtime Session",
            f"**Prompt:** {self.prompt[:200]}",
            f"**Setup OK:** {self.setup_ok}",
            "",
            f"## Routed Matches ({len(self.routed_matches)})",
        ]
        for m in self.routed_matches[:10]:
            lines.append(f"- [{m.kind}] **{m.name}** (score={m.score:.2f}) `{m.source_hint}`")

        if self.permission_denials:
            lines += ["", "## Permission Denials"]
            for d in self.permission_denials:
                lines.append(f"- `{d.tool_name}`: {d.reason}")

        if self.turn_results:
            lines += ["", "## Turn Results"]
            for tr in self.turn_results:
                lines.append(tr.as_markdown())

        return "\n".join(lines)


# ── PortRuntime ──────────────────────────────────────────────────────────────

class PortRuntime:
    """
    Orchestrates routing and execution for the offline agent.

    Usage:
        runtime = PortRuntime()
        session = runtime.bootstrap_session("fix the bug in utils.py")
        print(session.as_markdown())
    """

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        engine_config: Optional[QueryEngineConfig] = None,
    ):
        self.agent_config = agent_config or AgentConfig()
        self.engine_config = engine_config or QueryEngineConfig()
        self._engine: Optional[OllamaQueryEngine] = None

    @property
    def engine(self) -> OllamaQueryEngine:
        if self._engine is None:
            self._engine = OllamaQueryEngine(
                agent_config=self.agent_config,
                engine_config=self.engine_config,
            )
        return self._engine

    # ── Routing ──────────────────────────────────────────────────────────────

    def route_prompt(self, prompt: str, limit: int = 5) -> list[RoutedMatch]:
        """Match prompt tokens against the command/tool registry."""
        matches: list[tuple[float, RoutedMatch]] = []

        for cmd in find_commands(prompt)[:limit]:
            score = self._score(prompt, cmd)
            matches.append((score, RoutedMatch("command", cmd.name, cmd.source_hint, score)))

        for tool in find_tools(prompt)[:limit]:
            score = self._score(prompt, tool)
            matches.append((score, RoutedMatch("tool", tool.name, tool.source_hint, score)))

        matches.sort(key=lambda x: -x[0])
        return [m for _, m in matches[:limit]]

    def _score(self, prompt: str, module: PortingModule) -> float:
        tokens = set(prompt.lower().split())
        name_tokens = set(module.name.lower().replace("-", " ").replace("_", " ").split())
        hint_tokens = set(module.source_hint.lower().replace("/", " ").replace(".", " ").split())
        resp_tokens = set(module.responsibility.lower().split())

        name_hit = len(tokens & name_tokens) / max(len(name_tokens), 1)
        hint_hit = len(tokens & hint_tokens) / max(len(hint_tokens), 1)
        resp_hit = len(tokens & resp_tokens) / max(len(resp_tokens), 1)

        return round(name_hit * 3 + hint_hit * 2 + resp_hit, 3)

    def _infer_permission_denials(self) -> list[PermissionDenial]:
        """Return tools that are gated/restricted in the offline runtime."""
        return [
            PermissionDenial(
                tool_name="MCPTool",
                reason="MCP server connections require network; not available in offline mode.",
            ),
            PermissionDenial(
                tool_name="WebFetchTool",
                reason="Web fetch disabled in offline mode.",
            ),
            PermissionDenial(
                tool_name="WebSearchTool",
                reason="Web search disabled in offline mode.",
            ),
        ]

    # ── Session bootstrapping ─────────────────────────────────────────────────

    def bootstrap_session(self, prompt: str) -> RuntimeSession:
        """Build a RuntimeSession and submit the prompt to the query engine."""
        matches = self.route_prompt(prompt)
        denials = self._infer_permission_denials()

        session = RuntimeSession(
            prompt=prompt,
            context={"model": self.agent_config.model, "offline": True},
            routed_matches=matches,
            permission_denials=denials,
        )

        result = self.engine.submit_message(prompt)
        session.turn_results.append(result)
        return session

    # ── Multi-turn loop ───────────────────────────────────────────────────────

    def run_turn_loop(
        self,
        prompt: str,
        max_turns: Optional[int] = None,
    ) -> list[TurnResult]:
        """
        Run a conversation until the model signals completion or turn limit.
        Returns all TurnResults.
        """
        turns = max_turns or self.agent_config.max_turns
        results: list[TurnResult] = []

        current_prompt = prompt
        for _ in range(turns):
            result = self.engine.submit_message(current_prompt)
            results.append(result)

            if result.stop_reason == "end_turn" and not result.tool_calls:
                break

            # For subsequent turns, keep going if there's content
            if not result.output.strip():
                break
            current_prompt = ""  # empty follow-up keeps context flowing

        return results
