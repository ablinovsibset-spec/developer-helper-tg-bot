from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

EXEC_TOOL_NAME = "exec"
SEARCH_TOOL_NAME = "search_history"
EXEC_TIMEOUT_SECONDS = 30.0
OUTPUT_LIMIT = 3000
HEAD_CHARS = 1500
TAIL_CHARS = 1500

EXEC_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": EXEC_TOOL_NAME,
        "description": (
            "Выполнить консольную команду в изолированном Linux-контейнере (Alpine) "
            "и вернуть stdout, stderr и код выхода. Поддерживаются пайпы и &&. "
            "Файлы и установленные пакеты переживают сообщения (сброс — только "
            "пересозданием контейнера)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Команда для выполнения, например: echo hi",
                },
            },
            "required": ["command"],
        },
    },
}

SEARCH_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Поиск по завершённым беседам текущего чата (прошлым сессиям). "
            "Возвращает выдержки найденных сообщений с датой беседы и ролью "
            "автора. Используй, когда пользователь спрашивает о том, что "
            "обсуждалось раньше."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Поисковый запрос: слово или фраза, которые могли "
                        "встретиться в прошлой беседе"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass(frozen=True)
class ExecResult:
    """Сырой результат команды: контракт между исполнителем и форматированием."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandExecutor(Protocol):
    """Шов исполнителя команд: выполнение команды + уборка при завершении.

    Продакшн-реализация — Docker-песочница (sandbox.SandboxExecutor):
    один долгоживущий контейнер-жилец на процесс бота, создаётся лениво
    при первом `execute()`; в unit-тестах инъектируется двойник.
    """

    async def execute(
        self, command: str, timeout: float = EXEC_TIMEOUT_SECONDS
    ) -> ExecResult: ...

    async def stop(self) -> None: ...


class HistorySearcher(Protocol):
    """Шов поиска по прошлым беседам: читающий поиск по завершённым
    сессиям текущего чата.

    Продакшн-реализация — memory.ChatHistorySearcher (обёртка MemoryStore,
    привязанная к chat_id сообщения); в unit-тестах инъектируется двойник.
    """

    async def search(self, query: str) -> str: ...


def truncate_output(text: str) -> str:
    """Обрезает вывод ~до OUTPUT_LIMIT символов: голова + хвост + маркер."""
    if len(text) <= OUTPUT_LIMIT:
        return text
    omitted = len(text) - HEAD_CHARS - TAIL_CHARS
    return f"{text[:HEAD_CHARS]}\n… [обрезано {omitted} символов] …\n{text[-TAIL_CHARS:]}"


def format_result(
    exit_code: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    timeout: float,
) -> str:
    parts: list[str] = []
    if timed_out:
        parts.append(
            f"⏱ Таймаут {timeout:g}с: процесс принудительно завершён."
        )
    parts.append(f"exit_code: {exit_code}")
    parts.append(f"stdout:\n{stdout if stdout else '(пусто)'}")
    parts.append(f"stderr:\n{stderr if stderr else '(пусто)'}")
    return "\n".join(parts)


async def exec_command(
    executor: CommandExecutor,
    command: str,
    timeout: float = EXEC_TIMEOUT_SECONDS,
) -> str:
    """Выполняет команду через исполнитель и возвращает текст для модели.

    Ошибки выполнения не возбуждаются исключением — модель получает
    их описание как результат вызова инструмента.
    """
    try:
        result = await executor.execute(command, timeout)
    except Exception as exc:
        return f"Не удалось выполнить команду: {exc}"
    return truncate_output(
        format_result(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            timeout=timeout,
        )
    )
