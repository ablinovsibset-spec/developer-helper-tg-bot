"""Docker-песочница для исполнения команд агента (change add-docker-sandbox).

Единственный продакшн-исполнитель команд: контейнер на один агентный цикл
(design D2), харднинг-флаги D5, busybox `timeout` внутри контейнера и
async-страховка снаружи (D4), ленивое самовосстановление (D6), а также
startup-последовательность: `docker info` → sweep → ensure image (D7/D8).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
from pathlib import Path

from dev_helper_bot.tools import (
    EXEC_TIMEOUT_SECONDS,
    CommandExecutor,
    ExecResult,
)

log = logging.getLogger("bot.sandbox")

SANDBOX_IMAGE = "dev-helper-bot-sandbox:latest"
SANDBOX_LABEL = "dev-helper-bot.sandbox"
MEMORY_LIMIT = "512m"
PIDS_LIMIT = "256"
CONTAINER_UID = "1000"
WORK_DIR = "/work"

# Запас поверх таймаута команды на накладные расходы docker CLI (design D4).
CLI_GRACE_SECONDS = 10.0
# Бюджет на служебные вызовы docker CLI (create/start/rm/ps).
CLI_TIMEOUT_SECONDS = 30.0

# busybox `timeout` завершает команду SIGTERM'ом → exit 143; 124 — семантика
# coreutils (на случай замены образа). Оба кода считаем признаком таймаута.
_TIMEOUT_EXIT_CODES = frozenset({124, 143})

# stderr-маркеры ошибок самого docker CLI/демона (не команды внутри).
# exit-коды 125/126/127 неоднозначны, различаем по тексту (design, Risks).
_DAEMON_ERROR_MARKERS = (
    "Error response from daemon",
    "Cannot connect to the Docker daemon",
    "error during connect",
    "No such container",
)
_NO_SUCH_CONTAINER = re.compile(r"No such container", re.IGNORECASE)

DOCKER_UNAVAILABLE_MESSAGE = (
    "Docker недоступен — бот не может стартовать. "
    "Команды агента выполняются в Docker-песочнице: установите и запустите "
    "Docker (https://docs.docker.com/get-docker/), затем запустите бота снова."
)


class SandboxError(RuntimeError):
    """Ошибка инфраструктуры песочницы (docker CLI или демон)."""


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """Принудительно завершает процесс и его группу (защита от зомби)."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


async def _run_cli(
    args: list[str], timeout: float | None = CLI_TIMEOUT_SECONDS
) -> tuple[int, str, str]:
    """Запускает docker CLI и ждёт завершения; (exit_code, stdout, stderr)."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        raise SandboxError(f"не удалось запустить docker CLI: {exc}") from exc
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        _kill_process_group(process)
        await process.wait()
        raise SandboxError(f"docker {' '.join(args)}: бюджет {timeout:g}с исчерпан")
    exit_code = process.returncode if process.returncode is not None else -1
    return (
        exit_code,
        stdout_b.decode(errors="replace").strip(),
        stderr_b.decode(errors="replace").strip(),
    )


def _format_timeout(timeout: float) -> str:
    return f"{timeout:g}"


def _is_daemon_error(stderr: str) -> bool:
    return any(marker in stderr for marker in _DAEMON_ERROR_MARKERS)


class SandboxExecutor:
    """Исполнитель команд в Docker-контейнере на один агентный цикл.

    `start()` создаёт и запускает контейнер с ограничениями; `execute()`
    выполняет команду через `docker exec` (busybox `timeout` внутри,
    async-страховка снаружи, пересоздание при смерти контейнера);
    `stop()` принудительно удаляет контейнер best-effort.
    """

    def __init__(self, image: str = SANDBOX_IMAGE) -> None:
        self._image = image
        self._container_id: str | None = None

    @property
    def container_id(self) -> str | None:
        return self._container_id

    async def start(self) -> None:
        await self._create_and_start()

    async def stop(self) -> None:
        container_id, self._container_id = self._container_id, None
        if container_id is None:
            return
        try:
            code, _, stderr = await _run_cli(["rm", "-f", container_id])
            if code != 0 and not _NO_SUCH_CONTAINER.search(stderr):
                log.warning(
                    "Не удалось удалить песочницу %s: %s",
                    container_id[:12],
                    stderr,
                )
        except SandboxError as exc:
            log.warning("Не удалось удалить песочницу: %s", exc)

    async def _create_and_start(self) -> None:
        code, stdout, stderr = await _run_cli(
            [
                "create",
                "--memory", MEMORY_LIMIT,
                "--pids-limit", PIDS_LIMIT,
                "--cap-drop", "ALL",
                "-u", CONTAINER_UID,
                "--rm",
                "--label", f"{SANDBOX_LABEL}=true",
                self._image,
                "sleep", "infinity",
            ]
        )
        if code != 0:
            raise SandboxError(f"docker create: {stderr or stdout}")
        container_id = stdout
        code, stdout, stderr = await _run_cli(["start", container_id])
        if code != 0:
            try:
                await _run_cli(["rm", "-f", container_id])
            except SandboxError:
                pass
            raise SandboxError(f"docker start: {stderr or stdout}")
        self._container_id = container_id

    async def execute(
        self, command: str, timeout: float = EXEC_TIMEOUT_SECONDS
    ) -> ExecResult:
        for attempt in (1, 2):
            if self._container_id is None:
                await self._create_and_start()
            try:
                return await self._exec_in_container(
                    self._container_id, command, timeout
                )
            except SandboxError as exc:
                # Ленивое самовосстановление (design D6): контейнер умер —
                # пересоздаём и повторяем команду ровно один раз.
                if attempt == 1 and _NO_SUCH_CONTAINER.search(str(exc)):
                    log.info("Песочница исчезла, пересоздаю и повторяю команду")
                    self._container_id = None
                    continue
                raise
        raise SandboxError("не удалось выполнить команду в песочнице")

    async def _exec_in_container(
        self, container_id: str, command: str, timeout: float
    ) -> ExecResult:
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                "-w", WORK_DIR,
                container_id,
                "timeout", _format_timeout(timeout),
                "sh", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as exc:
            raise SandboxError(f"не удалось запустить docker exec: {exc}") from exc
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout + CLI_GRACE_SECONDS
            )
        except asyncio.TimeoutError:
            # Страховка от зависания docker CLI: внутренний `timeout` должен
            # был сработать раньше; процесс внутри контейнера умрёт вместе
            # с контейнером в конце цикла (design D4).
            _kill_process_group(process)
            await process.wait()
            return ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        exit_code = process.returncode if process.returncode is not None else -1
        stderr = stderr_b.decode(errors="replace")
        if _is_daemon_error(stderr):
            raise SandboxError(stderr.strip())
        return ExecResult(
            exit_code=exit_code,
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr,
            timed_out=exit_code in _TIMEOUT_EXIT_CODES,
        )


def default_dockerfile_dir() -> Path:
    """Корень репозитория с Dockerfile образа песочницы."""
    return Path(__file__).resolve().parents[2]


async def assert_docker_running() -> None:
    """Fail fast при недоступном демоне, в стиле config.telegram_token()."""
    try:
        code, _, stderr = await _run_cli(["info"])
    except SandboxError as exc:
        raise SystemExit(f"{DOCKER_UNAVAILABLE_MESSAGE}\nПодробности: {exc}")
    if code != 0:
        raise SystemExit(f"{DOCKER_UNAVAILABLE_MESSAGE}\nПодробности: {stderr}")


async def sweep_orphaned_sandboxes() -> int:
    """Удаляет контейнеры с меткой бота, оставшиеся от прошлых запусков."""
    code, stdout, stderr = await _run_cli(
        ["ps", "-q", "--filter", f"label={SANDBOX_LABEL}=true"]
    )
    if code != 0:
        log.warning("Не удалось найти осиротевшие песочницы: %s", stderr)
        return 0
    container_ids = [line for line in stdout.splitlines() if line.strip()]
    for container_id in container_ids:
        code, _, stderr = await _run_cli(["rm", "-f", container_id])
        if code != 0:
            log.warning(
                "Не удалось удалить песочницу-сироту %s: %s",
                container_id[:12],
                stderr,
            )
    return len(container_ids)


async def ensure_image(
    image: str = SANDBOX_IMAGE, build_context: Path | None = None
) -> None:
    """Собирает образ песочницы, если его нет локально (design D8)."""
    code, _, _ = await _run_cli(["image", "inspect", image])
    if code == 0:
        return
    context = build_context or default_dockerfile_dir()
    log.info("Образ %s не найден — собираю из %s…", image, context)
    code, stdout, stderr = await _run_cli(
        ["build", "-t", image, str(context)], timeout=None
    )
    if code != 0:
        raise SandboxError(f"сборка образа не удалась: {stderr or stdout}")


async def prepare_sandbox_environment() -> None:
    """Startup-последовательность бота: docker info → sweep → ensure image."""
    await assert_docker_running()
    swept = await sweep_orphaned_sandboxes()
    if swept:
        log.info("Убрано осиротевших песочниц: %d", swept)
    try:
        await ensure_image()
    except SandboxError as exc:
        raise SystemExit(f"Не удалось подготовить образ песочницы: {exc}")
