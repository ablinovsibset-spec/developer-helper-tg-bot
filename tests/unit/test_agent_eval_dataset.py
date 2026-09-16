"""Политика агента по evaluation dataset: jailbreak, секреты, память, /new."""
from __future__ import annotations

import json

import pytest

from dev_helper_bot.main import handle_new, handle_text
from dev_helper_bot.memory import MemoryStore
from tests.conftest import (
    FakeCommandExecutor,
    FakeMessage,
    agent_eval_cases,
    load_agent_eval_dataset,
    make_llm_stub,
)

CHAT_ID = 42
FAKE_TELEGRAM_TOKEN = "fake-telegram-bot-token-eval-suite"
FAKE_LLM_API_KEY = "fake-llm-api-key-eval-suite"
L2_KINDS = ("jailbreak", "refusal", "memory", "reset")
MIN_CASES = 10


@pytest.fixture
async def store(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.open()
    yield store
    await store.close()


async def _handle(message, fake_bot, llm, store):
    await handle_text(
        message, fake_bot, llm, store, {}, FakeCommandExecutor()
    )


def _dump_requests(llm) -> str:
    return json.dumps(llm.requests, ensure_ascii=False)


def test_dataset_has_at_least_ten_unique_ids_and_l2_kinds():
    cases = load_agent_eval_dataset()["cases"]
    ids = [case["id"] for case in cases]
    kinds = {case["kind"] for case in cases}

    assert len(cases) >= MIN_CASES
    assert len(ids) == len(set(ids))
    assert set(L2_KINDS) <= kinds


@pytest.mark.parametrize(
    "case",
    agent_eval_cases("jailbreak"),
    ids=lambda case: case["id"],
)
async def test_jailbreak_does_not_replace_system(
    case, fake_bot, store, monkeypatch
):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TELEGRAM_TOKEN)
    monkeypatch.setenv("LLM_API_KEY", FAKE_LLM_API_KEY)
    llm = make_llm_stub(reply="ok")
    injection = case["turns"][0]["content"]

    await _handle(FakeMessage(injection, chat_id=CHAT_ID), fake_bot, llm, store)

    request = llm.requests[0]
    assert request[0]["role"] == "system"
    assert request[-1]["role"] == "user"
    assert injection in request[-1]["content"]
    dumped = _dump_requests(llm)
    assert FAKE_TELEGRAM_TOKEN not in dumped
    assert FAKE_LLM_API_KEY not in dumped


@pytest.mark.parametrize(
    "case",
    agent_eval_cases("memory"),
    ids=lambda case: case["id"],
)
async def test_memory_entities_are_in_the_next_request(case, fake_bot, store):
    llm = make_llm_stub(reply="ok")
    entities = case["expect"]["entities"]
    user_turns = [turn["content"] for turn in case["turns"] if turn["role"] == "user"]

    for text in user_turns:
        await _handle(FakeMessage(text, chat_id=CHAT_ID), fake_bot, llm, store)

    dumped = json.dumps(llm.requests[1], ensure_ascii=False)
    for entity in entities:
        assert entity in dumped


@pytest.mark.parametrize(
    "case",
    agent_eval_cases("reset"),
    ids=lambda case: case["id"],
)
async def test_reset_new_drops_history_without_llm(case, fake_bot, store):
    llm = make_llm_stub(reply="ok")
    entities = case["expect"]["entities"]

    for turn in case["turns"]:
        if turn["role"] == "command" and turn["content"] == "/new":
            llm.requests.clear()
            await handle_new(
                FakeMessage("/new", chat_id=CHAT_ID), fake_bot, store
            )
            assert llm.requests == []
            continue
        await _handle(
            FakeMessage(turn["content"], chat_id=CHAT_ID), fake_bot, llm, store
        )

    last_request = json.dumps(llm.requests[-1], ensure_ascii=False)
    for entity in entities:
        assert entity not in last_request
    assert llm.requests[-1][0]["role"] == "system"
    assert llm.requests[-1][-1]["role"] == "user"
