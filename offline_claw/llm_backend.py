"""
Ollama LLM backend — the core offline inference engine.

Replaces the Anthropic API with a locally-running Ollama instance.
Supports tool/function calling via Ollama's native tool call API.

Requires: `ollama pull <model>` before first use.
Default model: qwen2.5-coder:7b  (good code assistant, ~4GB)
Other good options: llama3.2:3b, deepseek-coder-v2:16b, codellama:13b
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Generator, Optional

from .models import AgentConfig, ToolCall, ToolResult


# ── Tool schemas exposed to the model ──────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the working directory. "
                "Use for running tests, listing files, git operations, etc. "
                "Avoid destructive commands unless explicitly requested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30).",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed, optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (optional).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact string in a file with new content. "
                "The old_string must match exactly (including whitespace). "
                "Use read_file first to confirm the exact text before editing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. 'src/**/*.py'.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (default: cwd).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default: cwd).",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob to filter files, e.g. '*.py'.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


# ── Ollama HTTP client ──────────────────────────────────────────────────────

class OllamaClient:
    """Minimal HTTP client for the Ollama API (no external dependencies)."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is it running? Start with: ollama serve\n{e}"
            ) from e

    def _post_stream(self, endpoint: str, payload: dict) -> Generator[dict, None, None]:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.strip()
                    if line:
                        yield json.loads(line.decode())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is it running? Start with: ollama serve\n{e}"
            ) from e

    def list_models(self) -> list[str]:
        try:
            result = self._post_stream.__func__  # just check connectivity
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        if stream:
            # collect streamed chunks into a final response
            full_content = ""
            tool_calls: list[dict] = []
            for chunk in self._post_stream("/api/chat", payload):
                msg = chunk.get("message", {})
                if msg.get("content"):
                    full_content += msg["content"]
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
                if chunk.get("done"):
                    return {
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                            "tool_calls": tool_calls or None,
                        },
                        "done": True,
                    }
            return {"message": {"role": "assistant", "content": full_content, "tool_calls": tool_calls or None}}
        else:
            return self._post("/api/chat", payload)

    def pull(self, model: str) -> None:
        """Pull a model (blocking). Shows progress lines."""
        print(f"Pulling {model} from Ollama registry...")
        for chunk in self._post_stream("/api/pull", {"name": model}):
            status = chunk.get("status", "")
            if status:
                print(f"  {status}", end="\r", flush=True)
        print(f"\n{model} ready.")


# ── High-level agent backend ────────────────────────────────────────────────

class OllamaBackend:
    """
    Wraps OllamaClient and provides the agentic chat loop interface
    consumed by QueryEngine.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = OllamaClient(base_url=config.ollama_base_url)

    def parse_tool_calls(self, message: dict) -> list[ToolCall]:
        """Extract tool calls from an assistant message."""
        raw = message.get("tool_calls") or []
        calls = []
        for i, tc in enumerate(raw):
            fn = tc.get("function", tc)
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(tool_name=name, arguments=args, call_id=str(i)))
        return calls

    def complete(
        self,
        messages: list[dict],
        use_tools: bool = True,
    ) -> tuple[str, list[ToolCall]]:
        """
        Send messages to Ollama and return (text_response, tool_calls).
        """
        tools = TOOL_SCHEMAS if use_tools else None
        response = self.client.chat(
            model=self.config.model,
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        msg = response.get("message", {})
        text = msg.get("content") or ""
        tool_calls = self.parse_tool_calls(msg)
        return text, tool_calls

    def stream_complete(
        self,
        messages: list[dict],
        use_tools: bool = True,
    ) -> Generator[str, None, None]:
        """Stream text tokens from the model."""
        tools = TOOL_SCHEMAS if use_tools else None
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        for chunk in self.client._post_stream("/api/chat", payload):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
            if chunk.get("done"):
                break

    def check_health(self) -> tuple[bool, str]:
        """Return (ok, message) for connectivity and model availability."""
        try:
            models = self.client.list_models()
        except ConnectionError as e:
            return False, str(e)

        if not models:
            return False, (
                f"Ollama is running but no models are installed. "
                f"Run: ollama pull {self.config.model}"
            )

        model_base = self.config.model.split(":")[0]
        available = any(model_base in m for m in models)
        if not available:
            return False, (
                f"Model '{self.config.model}' not found. "
                f"Available: {', '.join(models[:5])}. "
                f"Run: ollama pull {self.config.model}"
            )

        return True, f"OK — {self.config.model} via {self.config.ollama_base_url}"
