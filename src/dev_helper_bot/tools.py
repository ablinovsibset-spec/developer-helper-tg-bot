from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Protocol

EXEC_TOOL_NAME = "exec"
READ_FILE_TOOL_NAME = "read_file"
SEARCH_TOOL_NAME = "search_history"
LIST_TOOL_NAME = "list_sessions"
EXEC_TIMEOUT_SECONDS = 30.0
OUTPUT_LIMIT = 3000
HEAD_CHARS = 1500
TAIL_CHARS = 1500

EXEC_INFRA_ERROR_PREFIX = "Не удалось выполнить команду"

EXEC_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": EXEC_TOOL_NAME,
        "description": (
            "Выполнить консольную команду в изолированном Linux-контейнере (Alpine) "
            "и вернуть stdout, stderr и код выхода. Поддерживаются пайпы и &&. "
            "Файлы и установленные пакеты переживают сообщения (сброс — только "
            "пересозданием контейнера). Файлы для чтения не открывай cat/head "
            "целиком — используй инструмент read_file: он возвращает строки "
            "с номерами и экономит контекст."
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

READ_FILE_OUTPUT_LIMIT = 1500
"""Жёсткий потолок ответа read_file в символах (design D1): ~1/2 OUTPUT_LIMIT."""

READ_FILE_DEFAULT_LIMIT = 100
"""Строк на страницу, когда модель не указала limit."""

READ_FILE_CONTINUATION_HINT = "[ответ обрезан: продолжай со строки {}]"

READ_FILE_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": READ_FILE_TOOL_NAME,
        "description": (
            "Постранично прочитать текстовый файл песочницы: строки с их "
            "номерами. Файлы читай этим инструментом, а не cat через exec. "
            "Ответ ограничен ~1500 символами; при обрезке в конце будет "
            "номер строки, с которой можно продолжить чтение."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу в песочнице, например /work/notes.md",
                },
                "offset": {
                    "type": "integer",
                    "description": "Номер первой строки (нумерация с 1); по умолчанию 1",
                },
                "limit": {
                    "type": "integer",
                    "description": "Максимум строк в ответе; по умолчанию 100",
                },
            },
            "required": ["path"],
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

LIST_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": LIST_TOOL_NAME,
        "description": (
            "Обзор завершённых бесед текущего чата (прошлых сессий): "
            "дата начала каждой беседы, число сообщений и превью её "
            "начала. Первый шаг при вопросах о прошлых беседах: по "
            "превью выбери ключевое слово для search_history. "
            "Параметров нет."
        ),
        "parameters": {"type": "object", "properties": {}},
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
    """Шов доступа к прошлым беседам: читающий поиск по завершённым
    сессиям текущего чата и их обзор.

    Продакшн-реализация — memory.ChatHistorySearcher (обёртка MemoryStore,
    привязанная к chat_id сообщения); в unit-тестах инъектируется двойник.
    """

    async def search(self, query: str) -> str: ...

    async def list_sessions(self) -> str: ...


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
        return f"{EXEC_INFRA_ERROR_PREFIX}: {exc}"
    return truncate_output(
        format_result(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            timeout=timeout,
        )
    )


def format_page(lines: list[str], first_number: int, has_more: bool) -> str:
    """Нумерует строки и укладывает страницу в потолок ответа read_file.

    Обрезка — по границам строк, чтобы «продолжай со строки N» был точным;
    если даже одна строка не влезает, она жёстко обрезается (потолок —
    инвариант ответа целиком, включая подсказку).
    """

    def build(budget: int) -> tuple[list[str], bool]:
        body: list[str] = []
        used = 0
        for number, line in enumerate(lines, start=first_number):
            entry = f"{number}: {line}"
            cost = len(entry) + (1 if body else 0)
            if used + cost > budget:
                return body, True
            body.append(entry)
            used += cost
        return body, has_more

    body, truncated = build(READ_FILE_OUTPUT_LIMIT)
    if not truncated:
        return "\n".join(body)

    hint = READ_FILE_CONTINUATION_HINT.format(first_number + len(body))
    body, _ = build(READ_FILE_OUTPUT_LIMIT - len(hint) - 1)
    if not body:
        # Единственная строка длиннее всего бюджета: жёсткая обрезка строки,
        # подсказка указывает на следующую (хвост обрезанной строки потерян).
        hint = READ_FILE_CONTINUATION_HINT.format(first_number + 1)
        budget = READ_FILE_OUTPUT_LIMIT - len(hint) - 1
        body = [f"{first_number}: {lines[0]}"[:budget]]
    return "\n".join(body) + "\n" + hint


async def read_file(
    executor: CommandExecutor,
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Читает диапазон строк файла в той же песочнице, что и exec (design D1).

    Потолок ответа и подсказка продолжения гарантируются кодом, а не
    дисциплиной модели. Ошибки (нет файла, каталог, доступ) возвращаются
    текстом и не прерывают агентный цикл.
    """
    try:
        first = 1 if offset is None else int(offset)
        page = READ_FILE_DEFAULT_LIMIT if limit is None else int(limit)
    except (TypeError, ValueError):
        return "Ошибка аргументов: offset и limit должны быть целыми числами."
    first = max(1, first)
    page = max(1, page)
    # Страница + 1 строка: лишняя — индикатор, что за диапазоном есть ещё.
    command = f"sed -n {first},{first + page}p {shlex.quote(path)}"
    try:
        result = await executor.execute(command)
    except Exception as exc:
        return f"{EXEC_INFRA_ERROR_PREFIX}: {exc}"
    if result.exit_code != 0:
        detail = result.stderr.strip() or f"exit_code: {result.exit_code}"
        return f"Ошибка чтения файла {path}: {detail}"
    lines = result.stdout.splitlines()
    has_more = len(lines) > page
    lines = lines[:page]
    if not lines:
        return f"(файл {path} пуст или строк с номером {first} в нём нет)"
    return format_page(lines, first, has_more)
