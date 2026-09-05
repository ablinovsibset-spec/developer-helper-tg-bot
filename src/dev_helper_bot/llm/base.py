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


class Usage(TypedDict):
    """Метаданные использования поставщика из ответа (design D1/D8).

    Поля, которых ответ не содержит, — null, а не ноль: absence ≠ 0.
    `raw` — исходный usage-объект ответа как есть (для доизвлечения
    без миграций).
    """

    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    raw: dict[str, Any]


class _AssistantTurnRequired(TypedDict):
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str


class AssistantTurn(_AssistantTurnRequired, total=False):
    """Ход ассистента: финальный текст и/или вызовы инструментов.

    Reasoning-поле ответа поставщика сюда не попадает и в историю не пишется.
    `usage` — метаданные использования поставщика, если ответ их содержит
    (иначе null); на разбор хода не влияет.
    """

    usage: Usage | None


ToolSpec = dict[str, Any]

ResponseFormat = dict[str, Any]
"""Указание формата ответа (например, json_schema) в wire-формате
OpenAI-совместимого API. Поставщик без поддержки получает запрос
без этого поля — деградация, а не ошибка."""


@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_format: ResponseFormat | None = None,
    ) -> AssistantTurn:
        ...


class LLMUnavailable(Exception):
    pass
