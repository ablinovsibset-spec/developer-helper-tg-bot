# Общие pytest-фикстуры для всего набора тестов.
from __future__ import annotations

import re
from typing import Any

import pytest

from dev_helper_bot.llm import (
    AssistantTurn,
    Message,
    ResponseFormat,
    ToolCall,
    ToolSpec,
)
from dev_helper_bot.tools import EXEC_TIMEOUT_SECONDS, ExecResult


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
        self.formats_per_request: list[ResponseFormat | None] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_format: ResponseFormat | None = None,
    ) -> AssistantTurn:
        self.requests.append(list(messages))
        self.tools_per_request.append(tools)
        self.formats_per_request.append(response_format)
        if self.error is not None:
            raise self.error
        if len(self.turns) > 1:
            return self.turns.pop(0)
        return self.turns[0]


def assistant_turn(
    content: str | None = "ok",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
    usage: dict | None = None,
) -> AssistantTurn:
    return {
        "content": content,
        "tool_calls": tool_calls or [],
        "finish_reason": finish_reason,
        "usage": usage,
    }


def tool_call(
    id: str = "call_1",
    name: str = "exec",
    arguments: str = '{"command": "echo hi"}',
) -> ToolCall:
    return {"id": id, "name": name, "arguments": arguments}


class FakeCommandExecutor:
    """Двойник CommandExecutor: жизнь без Docker и сети.

    Симулирует файловое состояние долгоживущего контейнера-жителя для команд
    вида `echo text > file` / `cat file` (состояние переживает вызовы),
    а также `echo text` и `exit N`; остальные команды возвращают
    scripted-результат (или default).
    """

    def __init__(
        self,
        scripted: dict[str, ExecResult] | None = None,
        default: ExecResult | None = None,
    ) -> None:
        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        self.scripted = scripted or {}
        self.default = default or ExecResult(
            exit_code=0, stdout="", stderr="", timed_out=False
        )
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1

    async def execute(
        self, command: str, timeout: float = EXEC_TIMEOUT_SECONDS
    ) -> ExecResult:
        self.commands.append(command)
        if command in self.scripted:
            return self.scripted[command]

        redirect = re.fullmatch(r"echo (.+?)\s*>\s*(\S+)", command)
        if redirect:
            self.files[redirect[2]] = redirect[1]
            return ExecResult(exit_code=0, stdout="", stderr="")
        echo = re.fullmatch(r"echo (.+)", command)
        if echo:
            return ExecResult(exit_code=0, stdout=f"{echo[1]}\n", stderr="")
        cat = re.fullmatch(r"cat (\S+)", command)
        if cat:
            if cat[1] in self.files:
                return ExecResult(exit_code=0, stdout=self.files[cat[1]], stderr="")
            return ExecResult(
                exit_code=1,
                stdout="",
                stderr=f"cat: can't open '{cat[1]}': No such file or directory",
            )
        exit_code = re.fullmatch(r"exit (\d+)", command)
        if exit_code:
            return ExecResult(exit_code=int(exit_code[1]), stdout="", stderr="")
        return self.default


class BrokenExecutor:
    """Двойник с падающей инфраструктурой: execute всегда бросает исключение."""

    async def stop(self) -> None: ...

    async def execute(
        self, command: str, timeout: float = EXEC_TIMEOUT_SECONDS
    ) -> ExecResult:
        raise RuntimeError("docker daemon is down")


class BrokenHistorySearcher:
    """Двойник шва HistorySearcher: оба метода падают ошибкой выполнения."""

    async def search(self, query: str) -> str:
        raise RuntimeError("database is closed")

    async def list_sessions(self) -> str:
        raise RuntimeError("database is closed")


@pytest.fixture
def fake_bot() -> FakeBot:
    return FakeBot()


def make_llm_stub(reply: str = "ok", error: Exception | None = None) -> FakeLLM:
    return FakeLLM(reply=reply, error=error)


def make_scripted_llm(turns: list[AssistantTurn]) -> FakeLLM:
    return FakeLLM(turns=turns)
