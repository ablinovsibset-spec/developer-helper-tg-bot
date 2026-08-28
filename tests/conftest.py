# Общие pytest-фикстуры для всего набора тестов.
from __future__ import annotations

from typing import Any

import pytest

from dev_helper_bot.llm import AssistantTurn, Message, ToolCall, ToolSpec


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
    """Двойник LLMClient: возвращает scripted-ходы (последний повторяется).

    Все входящие запросы (история и tools) записываются в self.requests
    и self.tools_per_request.
    """

    def __init__(
        self,
        turns: list[AssistantTurn] | None = None,
        reply: str = "ok",
        error: Exception | None = None,
    ) -> None:
        self.turns: list[AssistantTurn] = (
            list(turns) if turns else [assistant_turn(content=reply)]
        )
        self.error = error
        self.requests: list[list[Message]] = []
        self.tools_per_request: list[ToolSpec | None] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AssistantTurn:
        self.requests.append(list(messages))
        self.tools_per_request.append(tools)
        if self.error is not None:
            raise self.error
        if len(self.turns) > 1:
            return self.turns.pop(0)
        return self.turns[0]


def assistant_turn(
    content: str | None = "ok",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> AssistantTurn:
    return {
        "content": content,
        "tool_calls": tool_calls or [],
        "finish_reason": finish_reason,
    }


def tool_call(
    id: str = "call_1",
    name: str = "exec",
    arguments: str = '{"command": "echo hi"}',
) -> ToolCall:
    return {"id": id, "name": name, "arguments": arguments}


@pytest.fixture
def fake_bot() -> FakeBot:
    return FakeBot()


def make_llm_stub(reply: str = "ok", error: Exception | None = None) -> FakeLLM:
    return FakeLLM(reply=reply, error=error)


def make_scripted_llm(turns: list[AssistantTurn]) -> FakeLLM:
    return FakeLLM(turns=turns)
