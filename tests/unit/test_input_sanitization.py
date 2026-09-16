"""Санитизация входа: текущее поведение handle_text и HTML-шаблонов бота.

Пустой текст не роняет цикл (LLM всё равно вызывается). Лимит Telegram
4096 и спецсимволы разметки доходят до запроса без искажения. Экранирование
HTML — только шаблоны бота, не сырой ответ модели.
"""
from __future__ import annotations

import pytest

from dev_helper_bot.main import (
    TELEGRAM_MESSAGE_LIMIT,
    escape_html,
    format_html,
    handle_text,
)
from dev_helper_bot.memory import MemoryStore
from tests.conftest import FakeCommandExecutor, FakeMessage, make_llm_stub

CHAT_ID = 42
SPECIAL_CHARS = "_ * [ ] < > &"


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


async def test_empty_and_none_text_does_not_raise(fake_bot, store):
    llm = make_llm_stub(reply="ok")

    await _handle(FakeMessage(None, chat_id=CHAT_ID), fake_bot, llm, store)
    await _handle(FakeMessage("", chat_id=CHAT_ID + 1), fake_bot, llm, store)

    assert len(llm.requests) == 2
    assert llm.requests[0][-1]["role"] == "user"
    assert llm.requests[1][-1]["role"] == "user"


async def test_telegram_limit_text_reaches_llm_intact(fake_bot, store):
    llm = make_llm_stub(reply="ok")
    text = "я" * TELEGRAM_MESSAGE_LIMIT

    await _handle(FakeMessage(text, chat_id=CHAT_ID), fake_bot, llm, store)

    user_content = llm.requests[0][-1]["content"]
    assert text in user_content
    assert user_content.count("я") >= TELEGRAM_MESSAGE_LIMIT


async def test_markup_special_chars_reach_llm_unaltered(fake_bot, store):
    llm = make_llm_stub(reply="ok")

    await _handle(
        FakeMessage(SPECIAL_CHARS, chat_id=CHAT_ID), fake_bot, llm, store
    )

    user_content = llm.requests[0][-1]["content"]
    assert SPECIAL_CHARS in user_content


def test_escape_html_escapes_tags_and_ampersand():
    assert escape_html("<b>x</b> & y") == "&lt;b&gt;x&lt;/b&gt; &amp; y"
    assert "<" not in escape_html("<script>alert(1)</script>")


def test_format_html_escapes_user_fields_so_tags_are_not_markup():
    text = format_html(
        "⚠️ Не удалось проиндексировать «{filename}»: {reason}",
        filename="<img src=x>",
        reason="<script>alert(1)</script>",
    )

    assert "<img" not in text
    assert "<script>" not in text
    assert "&lt;img src=x&gt;" in text
    assert "&lt;script&gt;" in text
