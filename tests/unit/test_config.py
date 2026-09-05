from __future__ import annotations

import pytest

from dev_helper_bot.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_MODEL,
    DEFAULT_OBS_DB_PATH,
    DEFAULT_OBS_PRICE_INPUT_PER_M,
    DEFAULT_OBS_PRICE_OUTPUT_PER_M,
    make_llm,
    llm_model_name,
    memory_db_path,
    obs_db_path,
    obs_label,
    obs_price_input_per_m,
    obs_price_output_per_m,
    telegram_token,
)
from dev_helper_bot.llm.openai_compat import OpenAICompatibleClient

LLM_ENV_VARS = ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY")
OBS_ENV_VARS = (
    "OBS_DB_PATH",
    "OBS_PRICE_INPUT_PER_M",
    "OBS_PRICE_OUTPUT_PER_M",
    "OBS_LABEL",
)


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var in OBS_ENV_VARS:
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


def test_llm_model_name_default_and_env_override(monkeypatch: pytest.MonkeyPatch):
    assert llm_model_name() == DEFAULT_MODEL
    monkeypatch.setenv("LLM_MODEL", "llama-3.2-3b-instruct")

    assert llm_model_name() == "llama-3.2-3b-instruct"


def test_obs_db_path_defaults_to_vm_local_disk(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OBS_DB_PATH", raising=False)

    assert obs_db_path() == DEFAULT_OBS_DB_PATH
    # VM-локальный диск, не workspace-маунт (design D4)
    assert DEFAULT_OBS_DB_PATH.startswith("~/")
    assert not DEFAULT_OBS_DB_PATH.startswith(("/", "."))


def test_obs_db_path_env_override_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OBS_DB_PATH", "/tmp/custom-obs.db")

    assert obs_db_path() == "/tmp/custom-obs.db"


def test_obs_price_defaults_document_gpt_oss_20b():
    """Дефолтный прайс — цены запуска gpt-oss-20b в API OpenAI (design D5)."""
    assert DEFAULT_OBS_PRICE_INPUT_PER_M == 0.11
    assert DEFAULT_OBS_PRICE_OUTPUT_PER_M == 0.60
    assert obs_price_input_per_m() == 0.11
    assert obs_price_output_per_m() == 0.60


def test_obs_price_env_override_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OBS_PRICE_INPUT_PER_M", "1.5")
    monkeypatch.setenv("OBS_PRICE_OUTPUT_PER_M", "9")

    assert obs_price_input_per_m() == 1.5
    assert obs_price_output_per_m() == 9.0


def test_obs_label_defaults_to_none_and_follows_env(monkeypatch: pytest.MonkeyPatch):
    assert obs_label() is None
    monkeypatch.setenv("OBS_LABEL", "before-optimization")

    assert obs_label() == "before-optimization"
