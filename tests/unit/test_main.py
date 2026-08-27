from __future__ import annotations

import pytest

from dev_helper_bot.llm import LLMUnavailable
from dev_helper_bot.main import TELEGRAM_MESSAGE_LIMIT, handle_text, send_chunked

from tests.conftest import FakeMessage, make_llm_stub

CHAT_ID = 42


async def test_handle_text_forwards_prompt_and_replies_to_same_chat(
    fake_bot,
):
    llm = make_llm_stub(reply="ответ модели")
    message = FakeMessage("привет", chat_id=CHAT_ID)

    await handle_text(message, fake_bot, llm)

    assert llm.requests == [[{"role": "user", "content": "привет"}]]
    assert fake_bot.sent == [{"chat_id": CHAT_ID, "text": "ответ модели"}]


async def test_handle_text_llm_unavailable_sends_friendly_error(fake_bot):
    llm = make_llm_stub(error=LLMUnavailable("connection refused"))
    message = FakeMessage("привет", chat_id=CHAT_ID)

    await handle_text(message, fake_bot, llm)

    assert len(fake_bot.sent) == 1
    assert CHAT_ID == fake_bot.sent[0]["chat_id"]
    assert "недоступна" in fake_bot.sent[0]["text"]


async def test_handle_text_no_memory_between_messages(fake_bot):
    llm = make_llm_stub(reply="ok")

    await handle_text(FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm)
    await handle_text(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm)

    assert llm.requests == [
        [{"role": "user", "content": "первое"}],
        [{"role": "user", "content": "второе"}],
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
