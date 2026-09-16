"""Live red team / отказ / память / сброс против реального LLM (маркер live)."""
from __future__ import annotations

import json
import os
from datetime import datetime

import pytest

from dev_helper_bot.agent import run_agent
from dev_helper_bot.main import handle_new, handle_text
from dev_helper_bot.skills import build_request_messages
from tests.conftest import FakeMessage, agent_eval_cases

CHAT_ID = 42

pytestmark = pytest.mark.live


async def _user_reply(text, fake_bot, live_llm, store, skills, fake_executor):
    await handle_text(
        FakeMessage(text, chat_id=CHAT_ID),
        fake_bot,
        live_llm,
        store,
        skills,
        fake_executor,
    )
    return fake_bot.sent[-1]["text"]


def _secrets_in_env() -> list[str]:
    values = []
    for name in ("TELEGRAM_BOT_TOKEN", "LLM_API_KEY"):
        value = os.getenv(name)
        if value:
            values.append(value)
    return values


@pytest.mark.parametrize(
    "case",
    agent_eval_cases("jailbreak"),
    ids=lambda case: case["id"],
)
async def test_jailbreak_reply_hides_system_secrets_and_role_break(
    case, fake_bot, live_llm, store, skills, fake_executor
):
    injection = case["turns"][0]["content"]
    reply = await _user_reply(
        injection, fake_bot, live_llm, store, skills, fake_executor
    )
    lowered = reply.lower()
    for fragment in case["expect"]["forbidden_in_reply"]:
        assert fragment.lower() not in lowered
    for secret in _secrets_in_env():
        assert secret not in reply


@pytest.mark.parametrize(
    "case",
    agent_eval_cases("refusal"),
    ids=lambda case: case["id"],
)
async def test_refusal_does_not_invent_the_fact(
    case, fake_bot, live_llm, store, skills, fake_executor
):
    question = case["turns"][0]["content"]
    if case["expect"].get("no_tools"):
        # Без инструментов: иначе модель крутит exec/curl до лимита шагов
        # вместо отказа (как было с URL на pypi.org).
        history = build_request_messages(skills, [], question, datetime.now())
        reply = await run_agent(
            live_llm, history, tools=None, executor=fake_executor
        )
    else:
        reply = await _user_reply(
            question, fake_bot, live_llm, store, skills, fake_executor
        )
    lowered = reply.lower()
    assert any(
        marker.lower() in lowered for marker in case["expect"]["any_of_in_reply"]
    ), reply
    for claim in case["expect"]["none_of_in_reply"]:
        assert claim.lower() not in lowered, reply


async def test_recall_name_and_city(fake_bot, live_llm, store, skills, fake_executor):
    case = next(c for c in agent_eval_cases("memory") if c["id"] == "memory-name-city")
    for turn in case["turns"]:
        reply = await _user_reply(
            turn["content"], fake_bot, live_llm, store, skills, fake_executor
        )
    for entity in case["expect"]["entities"]:
        assert entity in reply


async def test_entities_absent_after_new(
    fake_bot, live_llm, store, skills, fake_executor
):
    """После /new открытая сессия пустая: сущности не уходят в следующий запрос.

    Финальный ответ модели не проверяем: закрытая сессия остаётся в
    search_history, и «как меня зовут?» штатно может вернуть прошлые сущности.
    """
    case = next(
        c for c in agent_eval_cases("reset") if c["id"] == "reset-after-name-city"
    )
    entities = case["expect"]["entities"]
    for turn in case["turns"]:
        if turn["role"] == "command" and turn["content"] == "/new":
            live_llm.requests.clear()
            await handle_new(
                FakeMessage("/new", chat_id=CHAT_ID), fake_bot, store
            )
            assert live_llm.requests == []
            continue
        await _user_reply(
            turn["content"], fake_bot, live_llm, store, skills, fake_executor
        )
    first_after_reset = json.dumps(live_llm.requests[0], ensure_ascii=False)
    for entity in entities:
        assert entity not in first_after_reset
