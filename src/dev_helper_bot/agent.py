from __future__ import annotations

import json
import logging
from typing import Any

from dev_helper_bot.llm import LLMClient, Message, ToolCall, ToolSpec
from dev_helper_bot.sandbox import SandboxExecutor
from dev_helper_bot.tools import (
    EXEC_TOOL_NAME,
    CommandExecutor,
    exec_command,
)

log = logging.getLogger("bot.agent")

MAX_LLM_STEPS = 8

STEPS_EXHAUSTED_MESSAGE = (
    "Не смог получить ответ за отведённое число шагов "
    f"({MAX_LLM_STEPS}). Попробуйте переформулировать запрос "
    "или начните новый диалог командой /new."
)


async def execute_tool_call(
    call: ToolCall, executor: CommandExecutor
) -> str:
    """Исполняет один вызов инструмента; ошибки возвращает как текст.

    Некорректные аргументы или неизвестный инструмент — не исключение,
    а tool-сообщение с ошибкой: модель видит её и корректирует вызов.
    """
    if call["name"] != EXEC_TOOL_NAME:
        return (
            f"Ошибка: неизвестный инструмент {call['name']!r}. "
            f"Доступен только {EXEC_TOOL_NAME!r}."
        )
    try:
        arguments = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError as exc:
        return (
            f"Ошибка разбора arguments (ожидается JSON-объект "
            f'со строковым полем "command"): {exc}'
        )
    if not isinstance(arguments, dict):
        return "Ошибка аргументов: arguments должен быть JSON-объектом."
    command = arguments.get("command")
    if not isinstance(command, str) or not command:
        return 'Ошибка аргументов: обязательный строковый параметр "command" отсутствует.'
    try:
        return await exec_command(executor, command)
    except Exception as exc:
        return f"Ошибка выполнения инструмента: {exc}"


async def run_agent(
    llm: LLMClient,
    history: list[Message],
    tools: list[ToolSpec] | None = None,
    executor: CommandExecutor | None = None,
) -> str:
    """Агентный цикл: LLM → tool_calls → результаты в историю → повтор.

    Дополняет `history` на месте: assistant-ходы с вызовами инструментов,
    tool-сообщения с результатами и финальный ответ ассистента.
    Reasoning в историю не попадает (отбрасывается на уровне контракта LLM).

    Песочница создаётся перед первым шагом и принудительно удаляется
    в `finally` независимо от исхода цикла (design D2).
    """
    if executor is None:
        executor = SandboxExecutor()
    try:
        await executor.start()
    except Exception as exc:
        # Не роняем цикл: exec вернёт модели текст ошибки, startup-проверка
        # уже гарантировала Docker при старте бота.
        log.warning("Не удалось создать песочницу: %s", exc)
    try:
        for _ in range(MAX_LLM_STEPS):
            turn = await llm.complete(history, tools)
            if not turn["tool_calls"]:
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
            for call in turn["tool_calls"]:
                result = await execute_tool_call(call, executor)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )
        return STEPS_EXHAUSTED_MESSAGE
    finally:
        await executor.stop()
