from __future__ import annotations

import os

from dev_helper_bot.embeddings import EmbeddingClient
from dev_helper_bot.embeddings.openai_compat import OpenAICompatibleEmbeddingClient
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

DEFAULT_RAG_DB_PATH = "~/.local/share/dev-helper-bot/rag.db"
"""Индекс документов — отдельная БД от переписки (design D1 change
add-document-rag): другой lifecycle и расширение sqlite-vec. Тот же
VM-локальный диск, что у memory/obs."""

DEFAULT_EMBEDDING_MODEL = "baai/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024
"""Модель эмбеддингов и её размерность (design D5). Смена модели меняет
геометрию векторов: старый индекс несовместим — нужен wipe rag.db
и повторная загрузка документов."""

DEFAULT_RAG_MAX_UPLOAD_BYTES = 5_000_000
DEFAULT_RAG_MAX_EXTRACT_CHARS = 300_000
DEFAULT_RAG_MAX_CHUNKS_PER_DOC = 800
"""Лимиты индексации (design D8): байты проверяются до тяжёлой работы,
символы и число chunks — после извлечения и разбиения."""

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


def make_embeddings() -> EmbeddingClient:
    """Клиент эмбеддингов для RAG (design D5).

    По умолчанию тот же endpoint и ключ, что у chat (`LLM_BASE_URL` /
    `LLM_API_KEY`), но своя модель `EMBEDDING_MODEL`. Отдельные
    `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` нужны, когда эмбеддинги
    берутся у другого поставщика, чем диалог.
    """
    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv(
        "LLM_BASE_URL", DEFAULT_BASE_URL
    )
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or None
    return OpenAICompatibleEmbeddingClient(
        base_url=base_url,
        model=embedding_model_name(),
        dimension=embedding_dim(),
        api_key=api_key,
    )


def embedding_model_name() -> str:
    """Модель эмбеддингов; задаётся отдельно от LLM_MODEL (design D5)."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embedding_dim() -> int:
    """Размерность векторов модели эмбеддингов (design D5)."""
    return int(os.getenv("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM))


def rag_db_path() -> str:
    """Путь к БД индекса документов; переопределяется RAG_DB_PATH (design D1)."""
    return os.getenv("RAG_DB_PATH", DEFAULT_RAG_DB_PATH)


def rag_max_upload_bytes() -> int:
    """Лимит сырого размера загружаемого файла, байты (design D8)."""
    return int(os.getenv("RAG_MAX_UPLOAD_BYTES", DEFAULT_RAG_MAX_UPLOAD_BYTES))


def rag_max_extract_chars() -> int:
    """Лимит объёма извлечённого из документа текста, символы (design D8)."""
    return int(os.getenv("RAG_MAX_EXTRACT_CHARS", DEFAULT_RAG_MAX_EXTRACT_CHARS))


def rag_max_chunks_per_doc() -> int:
    """Лимит числа chunks одного документа после разбиения (design D8)."""
    return int(os.getenv("RAG_MAX_CHUNKS_PER_DOC", DEFAULT_RAG_MAX_CHUNKS_PER_DOC))


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
