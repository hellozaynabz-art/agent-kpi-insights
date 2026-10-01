"""Thin wrapper around the Anthropic SDK: model choice, request defaults and cost tracking."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("CLAUDE_EFFORT", "medium")
# On a safety decline the API retries the request on a fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens (Claude Opus 5.5).
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ = 4.00, 20.00, 0.20


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage) -> None:
        with self._lock:  # agents may call the API from several threads
            self.calls += 1
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_IN
            + self.output_tokens * PRICE_OUT
            + self.cache_read_tokens * PRICE_CACHE_READ
        ) / 1e6


@dataclass
class Claude:
    """Holds the client and running usage; ``create`` applies the project defaults."""

    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)
    usage: Usage = field(default_factory=Usage)

    def create(self, *, output_config: dict | None = None, **kwargs):
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=kwargs.pop("max_tokens", 16_000),
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT, **(output_config or {})},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            cache_control={"type": "ephemeral"},
            **kwargs,
        )
        self.usage.add(response.usage)
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined this request.")
        return response


def text_of(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()
