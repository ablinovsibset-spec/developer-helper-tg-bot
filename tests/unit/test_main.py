from __future__ import annotations

import pytest

from dev_helper_bot.agent import STEPS_EXHAUSTED_MESSAGE
from dev_helper_bot.llm import LLMUnavailable
from dev_helper_bot.main import (
    NEW_CHAT_CONFIRMATION,
    TELEGRAM_MESSAGE_LIMIT,
    WAITING_MESSAGE,
    handle_new,
    handle_text,
    send_chunked,
)
from dev_helper_bot.tools import EXEC_TOOL_SPEC

from tests.conftest import FakeMessage, assistant_turn, make_llm_stub, make_scripted_llm, tool_call

CHAT_ID = 42
SYSTEM_PROMPT = "Reasoning: medium"


async def handle(message, fake_bot, llm, histories=None):
    histories = histories if histories is not None else {}
    await handle_text(message, fake_bot, llm, histories, SYSTEM_PROMPT)
    return histories


async def test_handle_text_sends_waiting_then_final_reply(fake_bot):
    llm = make_llm_stub(reply="ответ модели")

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm)

    assert fake_bot.sent == [
        {"chat_id": CHAT_ID, "text": WAITING_MESSAGE},
        {"chat_id": CHAT_ID, "text": "ответ модели"},
    ]


async def test_handle_text_prompt_goes_to_llm_with_system_and_tools(fake_bot):
    llm = make_llm_stub(reply="ok")

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm)

    assert llm.requests[0] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "привет"},
    ]
    assert llm.tools_per_request == [[EXEC_TOOL_SPEC]]


async def test_handle_text_llm_unavailable_sends_friendly_error(fake_bot):
    llm = make_llm_stub(error=LLMUnavailable("connection refused"))

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm)

    assert len(fake_bot.sent) == 2  # ⏳ и сообщение об ошибке
    assert fake_bot.sent[-1]["chat_id"] == CHAT_ID
    assert "недоступна" in fake_bot.sent[-1]["text"]


async def test_context_is_kept_between_messages(fake_bot):
    llm = make_llm_stub(reply="первый ответ")
    histories = await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm
    )

    llm.turns = [assistant_turn("второй ответ")]
    await handle(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, histories)

    assert llm.requests[1] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "первое"},
        {"role": "assistant", "content": "первый ответ"},
        {"role": "user", "content": "второе"},
    ]


async def test_tool_transcript_is_kept_in_history(fake_bot):
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(arguments='{"command": "echo hi"}')],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="готово"),
        ]
    )

    histories = await handle(
        FakeMessage("запрос", chat_id=CHAT_ID), fake_bot, llm
    )

    second_request = llm.requests[1]
    roles = [m["role"] for m in second_request]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "echo hi" in second_request[2]["tool_calls"][0]["arguments"]
    assert "hi" in second_request[3]["content"]
    assert "reasoning" not in histories[CHAT_ID][2]
    assert histories[CHAT_ID][-1] == {"role": "assistant", "content": "готово"}


async def test_steps_exhausted_message_is_sent_to_chat(fake_bot):
    endless = assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "echo x"}')],
        finish_reason="tool_calls",
    )
    llm = make_scripted_llm([endless])

    await handle(FakeMessage("сложный запрос", chat_id=CHAT_ID), fake_bot, llm)

    assert fake_bot.sent[-1]["text"] == STEPS_EXHAUSTED_MESSAGE


async def test_new_command_resets_context_without_llm_call(fake_bot):
    llm = make_llm_stub(reply="ответ")
    histories = await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm
    )
    llm.requests.clear()

    await handle_new(FakeMessage("/new", chat_id=CHAT_ID), fake_bot, histories)

    assert fake_bot.sent[-1] == {"chat_id": CHAT_ID, "text": NEW_CHAT_CONFIRMATION}
    assert llm.requests == []  # LLM не вызывался

    llm.turns = [assistant_turn("новый ответ")]
    await handle(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, histories)

    assert llm.requests[0] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "второе"},
    ]


async def test_send_chunked_short_reply_is_one_message(fake_bot):
    text = "а" * 200

    await send_chunked(fake_bot, CHAT_ID, text)

    assert fake_bot.sent == [{"chat_id": CHAT_ID, "text": text}]


async def test_send_chunked_long_reply_is_split_in_order(fake_bot):
    text = "б" * 9000

    await send_chunked(fake_bot, CHAT_ID, text)

    lengths = [len(m["text"]) for m in fake_bot.sent]
    assert lengths == [TELEGRAM_MESSAGE_LIMIT, TELEGRAM_MESSAGE_LIMIT, 808]
    assert all(m["chat_id"] == CHAT_ID for m in fake_bot.sent)
    assert "".join(m["text"] for m in fake_bot.sent) == text


@pytest.mark.parametrize("length", [1, TELEGRAM_MESSAGE_LIMIT])
async def test_send_chunked_boundary_lengths_fit_one_message(fake_bot, length):
    text = "в" * length

    await send_chunked(fake_bot, CHAT_ID, text)

    assert len(fake_bot.sent) == 1
