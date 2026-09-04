from __future__ import annotations

import pytest

from dev_helper_bot.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_MODEL,
    make_llm,
    memory_db_path,
    telegram_token,
)
from dev_helper_bot.llm.openai_compat import OpenAICompatibleClient

LLM_ENV_VARS = ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY")


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_make_llm_defaults_to_local_openai_compatible():
    client = make_llm()

    assert isinstance(client, OpenAICompatibleClient)
    assert client._base_url == DEFAULT_BASE_URL
    assert client._model == DEFAULT_MODEL
    assert client._api_key is None
    assert client._timeout == 120.0


def test_llm_defaults_pin_gpt_oss_and_local_lm_studio():
    assert DEFAULT_BASE_URL == "http://localhost:1234/v1"
    assert DEFAULT_MODEL == "openai/gpt-oss-20b"


def test_make_llm_unknown_provider_raises_value_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_PROVIDER", "skynet")

    with pytest.raises(ValueError, match="LLM_PROVIDER.*skynet"):
        make_llm()


def test_telegram_token_missing_exits_with_hint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(SystemExit, match=r"TELEGRAM_BOT_TOKEN.*\.env.*BotFather"):
        telegram_token()


def test_telegram_token_present_is_returned(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")

    assert telegram_token() == "123:abc"


def test_memory_db_path_defaults_to_vm_local_disk(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MEMORY_DB_PATH", raising=False)

    assert memory_db_path() == DEFAULT_MEMORY_DB_PATH
    # VM-локальный диск, не workspace-маунт (design D2)
    assert not DEFAULT_MEMORY_DB_PATH.startswith(("/", "."))
    assert DEFAULT_MEMORY_DB_PATH.startswith("~/")


def test_memory_db_path_env_override_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMORY_DB_PATH", "/tmp/custom-memory.db")

    assert memory_db_path() == "/tmp/custom-memory.db"
