from __future__ import annotations

import json
from typing import Any

from dev_helper_bot.llm import LLMClient, Message, ToolCall, ToolSpec
from dev_helper_bot.tools import (
    EXEC_TOOL_NAME,
    CommandExecutor,
    exec_command,
)

MAX_LLM_STEPS = 8

STEPS_EXHAUSTED_MESSAGE = (
    "Не смог получить ответ за отведённое число шагов "
    f"({MAX_LLM_STEPS}). Попробуйте переформулировать запрос "
    "или начните новый диалог командой /new."
)

MAX_JSON_RETRIES = 3

JSON_RETRIES_EXHAUSTED_MESSAGE = (
    "Модель не смогла сформировать корректный структурированный ответ "
    f"({MAX_JSON_RETRIES} попытки подряд). Попробуйте упростить запрос "
    "или начните новый диалог командой /new."
)

RESPONSE_TRUNCATED_MESSAGE = (
    "Ответ модели оборван лимитом длины и не может быть корректно "
    "обработан. Попробуйте упростить запрос или начните новый диалог "
    "командой /new."
)


def _required_params(spec: ToolSpec) -> list[tuple[str, str | None]]:
    """Обязательные параметры спецификации инструмента: пары (имя, JSON-тип)."""
    function = spec.get("function") or {}
    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    return [
        (name, (properties.get(name) or {}).get("type"))
        for name in parameters.get("required") or []
    ]


def _expected_form(spec: ToolSpec) -> str:
    """Описание ожидаемой формы аргументов для текста ошибки валидации."""
    params = _required_params(spec)
    if len(params) == 1:
        name, declared = params[0]
        if declared == "string":
            return f'JSON-объект со строковым полем "{name}"'
        return f'JSON-объект с обязательным полем "{name}"'
    described = ", ".join(
        f'"{name}" (строка)' if declared == "string" else f'"{name}"'
        for name, declared in params
    )
    return f"JSON-объект с обязательными полями {described}" if described else "JSON-объект"


def validate_tool_call(call: ToolCall, spec: ToolSpec) -> str | None:
    """Валидирует аргументы вызова инструмента против его спецификации.

    Проверяет синтаксис JSON и обязательные строковые параметры; возвращает
    текст ошибки для модели (конкретная ошибка разбора + ожидаемая форма)
    или None, если аргументы корректны.
    """
    try:
        arguments = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError as exc:
        return f"Ошибка разбора arguments (ожидается {_expected_form(spec)}): {exc}"
    if not isinstance(arguments, dict):
        return "Ошибка аргументов: arguments должен быть JSON-объектом."
    for name, declared in _required_params(spec):
        value = arguments.get(name)
        if value is None or (isinstance(value, str) and not value):
            if declared == "string":
                return (
                    f'Ошибка аргументов: обязательный строковый параметр '
                    f'"{name}" отсутствует.'
                )
            return f'Ошибка аргументов: обязательный параметр "{name}" отсутствует.'
        if declared == "string" and not isinstance(value, str):
            return f'Ошибка аргументов: параметр "{name}" должен быть строкой.'
    return None


def validate_final(content: str) -> str | None:
    """Точка расширения: валидация финального текстового ответа.

    Пока финальный ответ — свободный текст, всегда валиден (None).
    С первым потребителем финального JSON меняется только эта реализация,
    каркас цикла не трогается (design D2).
    """
    return None


def _validate_call(
    call: ToolCall, specs_by_name: dict[str, ToolSpec]
) -> str | None:
    """Валидация одного вызова хода; неизвестный инструмент — дело исполнителя."""
    spec = specs_by_name.get(call["name"])
    if spec is None:
        return None
    return validate_tool_call(call, spec)


async def execute_tool_call(
    call: ToolCall, executor: CommandExecutor
) -> str:
    """Исполняет валидированный вызов инструмента; ошибки возвращает как текст.

    Некорректные аргументы валидируются в агентном цикле до исполнителя
    (validate_tool_call); неизвестный инструмент — по-прежнему не исключение,
    а tool-сообщение с ошибкой: модель видит её и корректирует вызов.
    """
    if call["name"] != EXEC_TOOL_NAME:
        return (
            f"Ошибка: неизвестный инструмент {call['name']!r}. "
            f"Доступен только {EXEC_TOOL_NAME!r}."
        )
    arguments = json.loads(call["arguments"] or "{}")
    try:
        return await exec_command(executor, arguments["command"])
    except Exception as exc:
        return f"Ошибка выполнения инструмента: {exc}"


async def run_agent(
    llm: LLMClient,
    history: list[Message],
    tools: list[ToolSpec] | None = None,
    *,
    executor: CommandExecutor,
) -> str:
    """Агентный цикл: LLM → валидация хода → tool_calls → результаты в историю → повтор.

    Дополняет `history` на месте: assistant-ходы с вызовами инструментов,
    tool-сообщения с результатами и финальный ответ ассистента.
    Reasoning в историю не попадает (отбрасывается на уровне контракта LLM).

    Структурированные элементы хода валидируются до исполнителя (design D2):
    провал оставляет битый ответ в истории, рядом — сообщение с конкретной
    ошибкой, переделку делает очередной вызов LLM (design D1). Серия провалов
    ограничена отдельным счётчиком MAX_JSON_RETRIES, а обрыв по длине
    (finish_reason == "length" при невалидной структуре) терминалится сразу,
    без ретраев (design D4).

    Исполнитель команд передаётся готовым и переживает сообщения:
    жизненным циклом песочницы владеет main (design D5) — контейнер-жилец
    создаётся при первом exec и удаляется при завершении процесса бота.
    """
    specs_by_name: dict[str, ToolSpec] = {
        (spec.get("function") or {}).get("name"): spec for spec in tools or []
    }
    validation_failures = 0

    for _ in range(MAX_LLM_STEPS):
        turn = await llm.complete(history, tools)

        if turn["tool_calls"]:
            errors: list[str | None] = [
                _validate_call(call, specs_by_name) for call in turn["tool_calls"]
            ]
        else:
            errors = [validate_final(turn["content"] or "")]
        failed = any(error is not None for error in errors)

        # Сначала валидация, потом finish_reason (design D4): ретраи
        # не чинят обрыв генерации, валидный ответ при length принимается.
        if failed and turn["finish_reason"] == "length":
            return RESPONSE_TRUNCATED_MESSAGE
        if failed:
            validation_failures += 1
            if validation_failures >= MAX_JSON_RETRIES:
                return JSON_RETRIES_EXHAUSTED_MESSAGE
        else:
            validation_failures = 0

        if not turn["tool_calls"]:
            if failed:
                # Битой финальный ответ остаётся в истории, рядом — фидбек
                # user-сообщением (design D1).
                history.append({"role": "assistant", "content": turn["content"]})
                history.append({"role": "user", "content": errors[0]})
                continue
            reply = turn["content"] or ""
            history.append({"role": "assistant", "content": reply})
            return reply

        history.append(
            {
                "role": "assistant",
                "content": turn["content"],
                "tool_calls": turn["tool_calls"],
            }
        )
        for call, error in zip(turn["tool_calls"], errors):
            result = (
                error
                if error is not None
                else await execute_tool_call(call, executor)
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )
    return STEPS_EXHAUSTED_MESSAGE
