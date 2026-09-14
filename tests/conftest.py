# Общие pytest-фикстуры для всего набора тестов.
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import pytest

from dev_helper_bot.embeddings import EmbeddingsUnavailable
from dev_helper_bot.llm import (
    AssistantTurn,
    Message,
    ResponseFormat,
    ToolCall,
    ToolSpec,
)
from dev_helper_bot.tools import EXEC_TIMEOUT_SECONDS, ExecResult


DEFAULT_USER_ID = 777


class FakeChat:
    """Минимальный двойник aiogram Chat: используется только .id."""

    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class FakeUser:
    """Двойник aiogram User: владелец документов — только .id (design D3)."""

    def __init__(self, user_id: int) -> None:
        self.id = user_id


class FakeDocument:
    """Двойник aiogram Document: имя, размер и байты для скачивания."""

    def __init__(
        self,
        file_name: str | None,
        data: bytes = b"",
        file_size: int | None = None,
    ) -> None:
        self.file_name = file_name
        self.data = data
        self.file_size = len(data) if file_size is None else file_size
        self.file_id = f"file-{file_name}"


class FakeMessage:
    """Минимальный двойник aiogram Message: .text, .chat.id, .from_user.id
    и .document для загрузки документов."""

    def __init__(
        self,
        text: str | None = None,
        chat_id: int = 0,
        user_id: int | None = None,
        document: FakeDocument | None = None,
    ) -> None:
        self.text = text
        self.chat = FakeChat(chat_id)
        self.from_user = FakeUser(DEFAULT_USER_ID if user_id is None else user_id)
        self.document = document


class FakeSentMessage:
    """Возврат send_message: нужен message_id для edit_message_text."""

    def __init__(self, message_id: int, chat_id: int, text: str) -> None:
        self.message_id = message_id
        self.chat = FakeChat(chat_id)
        self.text = text


class FakeBot:
    """Двойник aiogram Bot: записывает отправленные сообщения и отдаёт
    байты «скачанного» документа из самого двойника документа."""

    def __init__(
        self,
        download_error: Exception | None = None,
        edit_error: Exception | None = None,
    ) -> None:
        self.sent: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.download_error = download_error
        self.edit_error = edit_error
        self._next_message_id = 1

    async def send_message(self, chat_id: int, text: str) -> FakeSentMessage:
        message_id = self._next_message_id
        self._next_message_id += 1
        self.sent.append({"chat_id": chat_id, "text": text})
        return FakeSentMessage(message_id=message_id, chat_id=chat_id, text=text)

    async def edit_message_text(
        self,
        text: str,
        chat_id: int | None = None,
        message_id: int | None = None,
        **_kwargs: Any,
    ) -> None:
        if self.edit_error is not None:
            raise self.edit_error
        self.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "text": text}
        )

    async def download(self, document: Any, destination: Any) -> Any:
        if self.download_error is not None:
            raise self.download_error
        destination.write(document.data)
        destination.seek(0)
        return destination


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
    вида `echo text > file` / `cat file` / `sed -n A,Bp file` (состояние
    переживает вызовы), а также `echo text` и `exit N`; остальные команды
    возвращают scripted-результат (или default).
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
        sed = re.fullmatch(r"sed -n (\d+),(\d+)p (\S+)", command)
        if sed:
            start, end, path = int(sed[1]), int(sed[2]), sed[3]
            if path in self.files:
                picked = self.files[path].splitlines()[start - 1 : end]
                stdout = "".join(f"{line}\n" for line in picked)
                return ExecResult(exit_code=0, stdout=stdout, stderr="")
            return ExecResult(
                exit_code=1,
                stdout="",
                stderr=f"sed: can't read '{path}': No such file or directory",
            )
        head = re.fullmatch(r"head -n (\d+) (\S+)", command)
        if head:
            count, path = int(head[1]), head[2]
            if path in self.files:
                picked = self.files[path].splitlines()[:count]
                stdout = "".join(f"{line}\n" for line in picked)
                return ExecResult(exit_code=0, stdout=stdout, stderr="")
            return ExecResult(
                exit_code=1,
                stdout="",
                stderr=f"head: can't open '{path}': No such file or directory",
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


FAKE_EMBEDDING_DIM = 256


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class FakeEmbeddingClient:
    """Двойник EmbeddingClient: векторы без сети, ключей и моделей.

    Хешированный bag-of-words: токен попадает в бакет по стабильному хешу
    (blake2b, а не рандомизированный hash()), вектор нормируется. Отсюда
    два свойства, на которые опираются тесты: один и тот же текст даёт один
    и тот же вектор, а лексическое пересечение запроса и чанка повышает
    близость — значит retrieval проверяется на релевантности, а не на шуме.
    """

    def __init__(self, dimension: int = FAKE_EMBEDDING_DIM) -> None:
        self._dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # Текст без словарных токенов: единичный вектор, чтобы косинусная
            # метрика оставалась определённой.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


class BrokenEmbeddingClient:
    """Двойник шва эмбеддингов с недоступным провайдером."""

    def __init__(self, dimension: int = FAKE_EMBEDDING_DIM) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingsUnavailable("embeddings endpoint is down")


@pytest.fixture
def fake_bot() -> FakeBot:
    return FakeBot()


def make_llm_stub(reply: str = "ok", error: Exception | None = None) -> FakeLLM:
    return FakeLLM(reply=reply, error=error)


def make_scripted_llm(turns: list[AssistantTurn]) -> FakeLLM:
    return FakeLLM(turns=turns)
