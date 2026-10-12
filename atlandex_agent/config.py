"""Configuration, read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    openrouter_api_key: str = field(default="", repr=False)  # never printed
    base_url: str = "https://openrouter.ai/api/v1"
    atlandex_api_url: str = field(default="", repr=False)
    mcp_command: str = "atlandex-mcp"

    # Free models by default. Paid fallbacks are only reachable with ALLOW_PAID=true.
    model_small: str = "poolside/laguna-xs-2.1:free"
    model_large: str = "nvidia/nemotron-3-super-120b-a12b:free"
    paid_small: str = "poolside/laguna-xs-2.1"
    paid_large: str = "nvidia/nemotron-3-super-120b-a12b"

    allow_paid: bool = False
    budget_usd: float = 0.0

    requests_per_minute: int = 20
    max_retries: int = 6
    request_timeout_sec: float = 120.0
    max_tokens: int = 1500
    # Cap on the output of an agent turn. A turn should only be a tool call or `finish`, but the "reply READY" turn
    # used to turn into a full draft answer (median 713 tokens, 17% of latency), so agent turns are capped.
    # The cap is per model because hidden reasoning tokens count against it: the large model
    # (nemotron-3-super) is a reasoning model, and a 600-token cap cut 17 of its 60 agent turns off before
    # a tool call (fixed v1). The small model (laguna) does not reason and fits in 600.
    agent_max_tokens_small: int = 600
    agent_max_tokens_large: int = 1500
    agent_max_tokens: int | None = None  # if set (AGENT_MAX_TOKENS), overrides both: for experiments
    # Second try when the answer comes back empty (reasoning used up the 1500-token answer budget).
    answer_retry_max_tokens: int = 4000
    temperature: float = 0.0

    max_tool_steps: int = 6
    evidence_char_budget: int = 12000

    cache_enabled: bool = True
    cache_dir: Path = ROOT / ".cache" / "llm"

    # Optional Langfuse tracing: on only when both keys are set (see tracing.py).
    langfuse_public_key: str = field(default="", repr=False)
    langfuse_secret_key: str = field(default="", repr=False)
    langfuse_host: str = "https://cloud.langfuse.com"
    tracing: bool = True

    def model_for(self, which: str) -> str:
        if which == "small":
            return self.model_small
        if which == "large":
            return self.model_large
        return which  # an explicit model id

    def agent_cap_for(self, model: str) -> int:
        """Output-token cap for an agent turn of `model` (the paid twin of a free model gets the same cap)."""
        if self.agent_max_tokens:
            return self.agent_max_tokens
        if model in (self.model_large, self.paid_large):
            return self.agent_max_tokens_large
        return self.agent_max_tokens_small  # the small model and any explicit model id

    def paid_fallback_for(self, model: str) -> str | None:
        free_to_paid = {self.model_small: self.paid_small, self.model_large: self.paid_large}
        return free_to_paid.get(model)

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "Settings":
        load_dotenv(dotenv_path or ROOT / ".env")
        env = os.environ.get
        d = cls()
        return cls(
            openrouter_api_key=env("OPENROUTER_API_KEY", ""),
            base_url=env("OPENROUTER_BASE_URL", d.base_url),
            atlandex_api_url=env("ATLANDEX_API_URL", ""),
            mcp_command=env("ATLANDEX_MCP_COMMAND", d.mcp_command),
            model_small=env("MODEL_SMALL", d.model_small),
            model_large=env("MODEL_LARGE", d.model_large),
            paid_small=env("PAID_MODEL_SMALL", d.paid_small),
            paid_large=env("PAID_MODEL_LARGE", d.paid_large),
            allow_paid=_flag("ALLOW_PAID", False),
            budget_usd=float(env("BUDGET_USD", "0") or 0),
            requests_per_minute=int(env("REQUESTS_PER_MINUTE", str(d.requests_per_minute))),
            max_tool_steps=int(env("MAX_TOOL_STEPS", str(d.max_tool_steps))),
            agent_max_tokens_small=int(env("AGENT_MAX_TOKENS_SMALL", str(d.agent_max_tokens_small))),
            agent_max_tokens_large=int(env("AGENT_MAX_TOKENS_LARGE", str(d.agent_max_tokens_large))),
            agent_max_tokens=int(env("AGENT_MAX_TOKENS") or 0) or None,
            answer_retry_max_tokens=int(env("ANSWER_RETRY_MAX_TOKENS", str(d.answer_retry_max_tokens))),
            cache_enabled=_flag("LLM_CACHE", True),
            langfuse_public_key=env("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=env("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=env("LANGFUSE_HOST") or env("LANGFUSE_BASE_URL") or d.langfuse_host,
            tracing=_flag("LANGFUSE_TRACING", True),
        )

