from __future__ import annotations

import os

from dev_helper_bot.llm import LLMClient, LLMUnavailable
from dev_helper_bot.llm.openai_compat import OpenAICompatibleClient

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_PROVIDER = "openai_compatible"

DEFAULT_MEMORY_DB_PATH = "~/.local/share/dev-helper-bot/memory.db"
"""VM-локальный диск сандбокса, не workspace-маунт (design D2): SQLite
поверх сетевой ФС нестабилен из-за локов/WAL."""

DEFAULT_OBS_DB_PATH = "~/.local/share/dev-helper-bot/observability.db"
"""БД телеметрии — та же причина VM-локального диска (design D4)."""

DEFAULT_OBS_PRICE_INPUT_PER_M = 0.11
DEFAULT_OBS_PRICE_OUTPUT_PER_M = 0.60
"""Дефолтный виртуальный прайс $/1M токенов — цены запуска gpt-oss-20b
в API OpenAI (design D5). Локальная модель реально стоит $0: стоимость
учётная, для сопоставимости экспериментов «до/после»; переопределяется
OBS_PRICE_INPUT_PER_M / OBS_PRICE_OUTPUT_PER_M."""


def make_llm() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = llm_model_name()
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


def llm_model_name() -> str:
    """Имя модели для записи в телеметрию (то же, что уходит поставщику)."""
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def telegram_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Copy .env.example to .env and fill in the token from @BotFather."
        )
    return token


def memory_db_path() -> str:
    """Путь к файлу БД переписки; переопределяется MEMORY_DB_PATH (design D2)."""
    return os.getenv("MEMORY_DB_PATH", DEFAULT_MEMORY_DB_PATH)


def obs_db_path() -> str:
    """Путь к БД телеметрии; переопределяется OBS_DB_PATH (design D4)."""
    return os.getenv("OBS_DB_PATH", DEFAULT_OBS_DB_PATH)


def obs_price_input_per_m() -> float:
    """Виртуальная цена входных токенов, $ за 1M (design D5)."""
    return float(os.getenv("OBS_PRICE_INPUT_PER_M", DEFAULT_OBS_PRICE_INPUT_PER_M))


def obs_price_output_per_m() -> float:
    """Виртуальная цена выходных токенов, $ за 1M (design D5)."""
    return float(os.getenv("OBS_PRICE_OUTPUT_PER_M", DEFAULT_OBS_PRICE_OUTPUT_PER_M))


def obs_label() -> str | None:
    """Метка прогона для маркировки экспериментов «до/после» (design D5/D7).

    Задаётся на запуск процесса: все прогоны этого процесса несут метку,
    dashboard фильтрует по ней (--label)."""
    return os.getenv("OBS_LABEL") or None


DEFAULT_OBS_WEB_PORT = 8765
"""Дефолтный порт веб-дашборда (design D6); bind всегда 127.0.0.1 на хосте."""

DEFAULT_OBS_WEB_HOST = "127.0.0.1"
"""Дефолтный host бинда дашборда — loopback хоста (design D6: без LAN).
В сандбоксе sbx переопределяется на 0.0.0.0: проброс sbx ports входит в VM
через её сетевой интерфейс, а не loopback, так что bind на 127.0.0.1 внутри VM
недостижим для форварда. 0.0.0.0 внутри VM безопасен — sbx ports публикует
только на loopback хоста, с LAN дашборд недоступен."""


def obs_web_port() -> int:
    """Порт дашборда; переопределяется OBS_WEB_PORT (design D6)."""
    return int(os.getenv("OBS_WEB_PORT", DEFAULT_OBS_WEB_PORT))


def obs_web_host() -> str:
    """Host бинда дашборда; переопределяется OBS_WEB_HOST (design D6).
    Дефолт 127.0.0.1 (хост); в сандбоксе sbx — 0.0.0.0, иначе проброс портов
    не достает до loopback VM."""
    return os.getenv("OBS_WEB_HOST", DEFAULT_OBS_WEB_HOST)
