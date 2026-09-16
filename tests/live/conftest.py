"""Фикстуры live-набора: живой LLM из env и отдельные клиенты судьи на routerai.

Не вызывает load_dotenv: conftest подхватывается при сборке тестов, а
дефолтный pytest не должен читать секреты из workspace-.env.
"""
from __future__ import annotations

import os

import pytest

from dev_helper_bot.config import make_llm
from dev_helper_bot.llm import LLMUnavailable
from dev_helper_bot.llm.openai_compat import OpenAICompatibleClient
from dev_helper_bot.memory import MemoryStore
from dev_helper_bot.skills import default_skills_dir, load_skills
from tests.conftest import FakeCommandExecutor

ROUTERAI_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_SUBJECT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-sol"


def pytest_collection_modifyitems(items):
    for item in items:
        path = getattr(item, "path", None)
        if path is not None and path.parent.name == "live":
            item.add_marker(pytest.mark.live)


class SkipOnUnavailableLLM:
    """Прокси: недоступный LLM — skip, не падение дефолтного набора."""

    def __init__(self, inner: OpenAICompatibleClient) -> None:
        self._inner = inner
        self.requests: list = []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    async def complete(self, messages, tools=None, response_format=None):
        self.requests.append(list(messages))
        try:
            return await self._inner.complete(
                messages, tools=tools, response_format=response_format
            )
        except LLMUnavailable as exc:
            pytest.skip(f"LLM unavailable: {exc}")


@pytest.fixture
def live_llm():
    return SkipOnUnavailableLLM(make_llm())


@pytest.fixture
def fake_executor():
    return FakeCommandExecutor()


@pytest.fixture
def skills():
    return load_skills(default_skills_dir())


@pytest.fixture
async def store(tmp_path):
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.open()
    yield memory
    await memory.close()


def _routerai_client(model_env: str, default_model: str) -> SkipOnUnavailableLLM:
    api_key = os.getenv("LLM_API_KEY") or None
    if not api_key:
        pytest.skip("LLM_API_KEY is not set")
    model = os.getenv(model_env, default_model)
    return SkipOnUnavailableLLM(
        OpenAICompatibleClient(
            base_url=ROUTERAI_BASE_URL,
            model=model,
            api_key=api_key,
        )
    )


@pytest.fixture
def routerai_subject():
    return _routerai_client("JUDGE_SUBJECT_MODEL", DEFAULT_SUBJECT_MODEL)


@pytest.fixture
def routerai_judge():
    return _routerai_client("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
