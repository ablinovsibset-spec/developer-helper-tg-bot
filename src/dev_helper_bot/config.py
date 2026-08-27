from __future__ import annotations

import os

from dev_helper_bot.llm import LLMClient, LLMUnavailable
from dev_helper_bot.llm.openai_compat import OpenAICompatibleClient

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "Llama-3.2-3B-Instruct-4bit"
DEFAULT_PROVIDER = "openai_compatible"


def make_llm() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("LLM_API_KEY") or None

    if provider == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        f"Supported: 'openai_compatible'."
    )


def telegram_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Copy .env.example to .env and fill in the token from @BotFather."
        )
    return token
