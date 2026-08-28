from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

EXEC_TOOL_NAME = "exec"
EXEC_TIMEOUT_SECONDS = 30.0
OUTPUT_LIMIT = 3000
HEAD_CHARS = 1500
TAIL_CHARS = 1500

EXEC_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": EXEC_TOOL_NAME,
        "description": (
            "Выполнить консольную команду в shell системы и вернуть "
            "stdout, stderr и код выхода. Поддерживаются пайпы и &&."
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


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """Принудительно завершает процесс и его группу (защита от зомби)."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


async def exec_command(command: str, timeout: float = EXEC_TIMEOUT_SECONDS) -> str:
    """Выполняет команду в shell и возвращает текст-результат для модели.

    Ошибки выполнения не возбуждаются исключением — модель получает
    их описание как результат вызова инструмента.
    """
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        return f"Не удалось запустить процесс: {exc}"

    timed_out = False
    stdout_bytes = b""
    stderr_bytes = b""
    exit_code = -1
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
        exit_code = process.returncode if process.returncode is not None else -1
    except asyncio.TimeoutError:
        timed_out = True
        _kill_process_group(process)
        await process.wait()

    stdout = stdout_bytes.decode(errors="replace")
    stderr = stderr_bytes.decode(errors="replace")
    return truncate_output(
        format_result(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            timeout=timeout,
        )
    )
