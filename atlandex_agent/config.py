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
    model_small: str = "google/gemma-4-26b-a4b-it:free"
    model_large: str = "nvidia/nemotron-3-super-120b-a12b:free"
    paid_small: str = "google/gemma-4-26b-a4b-it"
    paid_large: str = "nvidia/nemotron-3-super-120b-a12b"

    allow_paid: bool = False
    budget_usd: float = 0.0

    requests_per_minute: int = 20
    max_retries: int = 6
    request_timeout_sec: float = 120.0
    max_tokens: int = 1500
    temperature: float = 0.0

    max_tool_steps: int = 6
    evidence_char_budget: int = 12000

    cache_enabled: bool = True
    cache_dir: Path = ROOT / ".cache" / "llm"

    def model_for(self, which: str) -> str:
        if which == "small":
            return self.model_small
        if which == "large":
            return self.model_large
        return which  # an explicit model id

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
            cache_enabled=_flag("LLM_CACHE", True),
        )
