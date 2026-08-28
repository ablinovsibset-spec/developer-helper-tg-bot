from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable


class ToolCall(TypedDict):
    """Вызов инструмента из хода ассистента.

    `arguments` — JSON-строка в том виде, в каком её отдал поставщик;
    разбор выполняет исполнитель инструмента, чтобы некорректный JSON
    мог быть возвращён модели как ошибка, а не исключением.
    """

    id: str
    name: str
    arguments: str


class _MessageRequired(TypedDict):
    role: str
    content: str | None


class Message(_MessageRequired, total=False):
    """Сообщение диалога; роли system/user/assistant/tool.

    `tool_calls` встречается только на assistant-ходах с вызовами
    инструментов, `tool_call_id` — только на tool-сообщениях с результатом.
    """

    tool_calls: list[ToolCall]
    tool_call_id: str


class AssistantTurn(TypedDict):
    """Ход ассистента: финальный текст и/или вызовы инструментов.

    Reasoning-поле ответа поставщика сюда не попадает и в историю не пишется.
    """

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str


ToolSpec = dict[str, Any]


@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AssistantTurn:
        ...


class LLMUnavailable(Exception):
    pass
