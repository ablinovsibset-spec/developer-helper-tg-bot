# Общие pytest-фикстуры для всего набора тестов.
from __future__ import annotations

from typing import Any

import pytest

from dev_helper_bot.llm import Message


class FakeChat:
    """Минимальный двойник aiogram Chat: используется только .id."""

    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class FakeMessage:
    """Минимальный двойник aiogram Message: используются .text и .chat.id."""

    def __init__(self, text: str, chat_id: int) -> None:
        self.text = text
        self.chat = FakeChat(chat_id)


class FakeBot:
    """Двойник aiogram Bot: записывает все отправленные сообщения."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append({"chat_id": chat_id, "text": text})


class FakeLLM:
    """Двойник LLMClient: возвращает заданный ответ или возбуждает ошибку.

    Все входящие запросы записываются в self.requests.
    """

    def __init__(self, reply: str = "ok", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.requests: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> str:
        self.requests.append(messages)
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture
def fake_bot() -> FakeBot:
    return FakeBot()


def make_llm_stub(reply: str = "ok", error: Exception | None = None) -> FakeLLM:
    return FakeLLM(reply=reply, error=error)
