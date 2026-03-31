"""Query request/response structures — matches claw-code's query.py interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QueryRequest:
    prompt: str
    system: Optional[str] = None
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class QueryResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"
