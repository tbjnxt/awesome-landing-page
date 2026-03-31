"""
offline_claw CLI — entry point.

Usage:
    python -m offline_claw                        # interactive REPL
    python -m offline_claw ask "fix tests"        # single shot
    python -m offline_claw check                  # health check
    python -m offline_claw summary                # workspace summary
    python -m offline_claw commands [query]       # list commands
    python -m offline_claw tools [query]          # list tools
    python -m offline_claw route <prompt>         # show routing for a prompt
"""

from __future__ import annotations

import argparse
import os
import sys

from .commands import render_command_index
from .llm_backend import OllamaBackend
from .models import AgentConfig
from .query_engine import OllamaQueryEngine, QueryEngineConfig
from .runtime import PortRuntime
from .tools import render_tool_index


# ── Handlers ─────────────────────────────────────────────────────────────────

def cmd_check(args: argparse.Namespace) -> int:
    config = AgentConfig(model=args.model, ollama_base_url=args.url)
    backend = OllamaBackend(config)
    ok, msg = backend.check_health()
    icon = "✓" if ok else "✗"
    print(f"{icon} {msg}")
    return 0 if ok else 1


def cmd_ask(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    if not prompt.strip():
        print("Error: provide a prompt.", file=sys.stderr)
        return 1

    config = AgentConfig(model=args.model, ollama_base_url=args.url)
    engine_config = QueryEngineConfig(tool_use_enabled=not args.no_tools)
    runtime = PortRuntime(agent_config=config, engine_config=engine_config)

    session = runtime.bootstrap_session(prompt)
    for result in session.turn_results:
        if result.output:
            print(result.output)
        for tr in result.tool_results:
            icon = "✓" if tr.success else "✗"
            print(f"\n[{icon} {tr.tool_name}]\n{tr.output or tr.error}", file=sys.stderr)
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    config = AgentConfig(model=args.model, ollama_base_url=args.url)
    engine_config = QueryEngineConfig(tool_use_enabled=not args.no_tools)

    # Health check before entering REPL
    backend = OllamaBackend(config)
    ok, msg = backend.check_health()
    if not ok:
        print(f"✗ {msg}", file=sys.stderr)
        return 1

    engine = OllamaQueryEngine(agent_config=config, engine_config=engine_config)
    print(f"offline_claw REPL — model: {config.model}")
    print("Type your prompt. Enter 'quit' or Ctrl-D to exit.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        result = engine.submit_message(prompt)

        # Show tool activity on stderr
        for tr in result.tool_results:
            icon = "✓" if tr.success else "✗"
            preview = (tr.output or tr.error or "")[:200]
            print(f"  [{icon} {tr.tool_name}] {preview}", file=sys.stderr)

        if result.output:
            print(result.output)
        print()

    print(engine.summary_report())
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from .commands import get_commands
    from .tools import get_tools, is_implemented

    cmds = get_commands()
    tools = get_tools()
    impl = [t for t in tools if is_implemented(t.name)]

    print("# offline_claw workspace summary")
    print(f"\nCommands mirrored : {len(cmds)}")
    print(f"Tools mirrored    : {len(tools)}")
    print(f"Tools implemented : {len(impl)}")
    print("\nImplemented tools:")
    for t in impl:
        print(f"  ✓ {t.name}")
    return 0


def cmd_commands(args: argparse.Namespace) -> int:
    query = " ".join(args.query) if args.query else ""
    print(render_command_index(query))
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    query = " ".join(args.query) if args.query else ""
    print(render_tool_index(query))
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    runtime = PortRuntime()
    matches = runtime.route_prompt(prompt, limit=10)
    if not matches:
        print("No matches found.")
        return 0
    print(f"# Route results for: {prompt!r}\n")
    for m in matches:
        print(f"  [{m.kind}] {m.name} (score={m.score}) — {m.source_hint}")
    return 0


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offline_claw",
        description="Offline Claude Code harness — powered by Ollama.",
    )
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("OFFLINE_CLAW_MODEL", "qwen2.5-coder:7b"),
        help="Ollama model to use (default: qwen2.5-coder:7b)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool use (text-only mode)",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Check Ollama connectivity and model availability")

    ask_p = sub.add_parser("ask", help="Ask a single question and exit")
    ask_p.add_argument("prompt", nargs="+")

    sub.add_parser("repl", help="Start interactive REPL (default)")

    sub.add_parser("summary", help="Show workspace summary")

    cmds_p = sub.add_parser("commands", help="List mirrored commands")
    cmds_p.add_argument("query", nargs="*")

    tools_p = sub.add_parser("tools", help="List mirrored tools")
    tools_p.add_argument("query", nargs="*")

    route_p = sub.add_parser("route", help="Show routing matches for a prompt")
    route_p.add_argument("prompt", nargs="+")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return cmd_check(args)
    elif args.command == "ask":
        return cmd_ask(args)
    elif args.command == "summary":
        return cmd_summary(args)
    elif args.command == "commands":
        return cmd_commands(args)
    elif args.command == "tools":
        return cmd_tools(args)
    elif args.command == "route":
        return cmd_route(args)
    else:
        # default: REPL
        return cmd_repl(args)


if __name__ == "__main__":
    sys.exit(main())
