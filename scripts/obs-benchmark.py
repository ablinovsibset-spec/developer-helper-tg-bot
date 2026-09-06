#!/usr/bin/env python3
"""Benchmark-харнесс: воспроизводимая генерация телеметрии прогонов
(change add-token-audit, design D3/D4/D5).

Гоняет фиксированный набор сценариев напрямую через агентный цикл, мимо
Telegram: тот же конвейер, что handle_text в main.py — system prompt из
скиллов, своя MemoryStore, RunRecorder + ObservingClient, run_agent с
SandboxExecutor. Отличие от handle_text только в отсутствии транспорта
и chunking; телеметрия по форме неотличима от боевой.

Изоляция (design D4): телеметрия — отдельный файл (--db / BENCHMARK_DB_PATH,
дефолт ~/.local/share/dev-helper-bot/benchmark.db), память — свой файл
(--memory-db / BENCHMARK_MEMORY_DB_PATH). Боевые observability.db и
memory.db не открываются вовсе. Метка прогонов — фиксированная константа
ниже (OBS_LABEL сознательно не читается, чтобы env не «перекрашивал»
бенчмарк). Файл памяти пересоздаётся на каждом запуске: сессии сценариев
всегда стартуют с чистого контекста, прогоны телеметрии накапливаются.

Фикстура-проект (design D5) доставляется в /work/sample_project перед
файловыми сценариями: служебный exec (гарантия контейнера-жильца),
rm -rf внутри и `docker cp` поверх — идемпотентность через пересоздание.

Недоступный LLM или Docker → понятная ошибка и ненулевой exit code
до начала прогонов (спека benchmark-harness).

Запуск:
    python scripts/obs-benchmark.py                     # полный набор
    python scripts/obs-benchmark.py --only files        # только сценарий files
    python scripts/obs-benchmark.py --db /tmp/bm.db --memory-db /tmp/bm-mem.db
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from dev_helper_bot.agent import run_agent  # noqa: E402
from dev_helper_bot.config import (  # noqa: E402
    llm_model_name,
    make_llm,
    obs_price_input_per_m,
    obs_price_output_per_m,
)
from dev_helper_bot.llm import LLMUnavailable, Message  # noqa: E402
from dev_helper_bot.memory import ChatHistorySearcher, MemoryStore  # noqa: E402
from dev_helper_bot.sandbox import (  # noqa: E402
    SandboxExecutor,
    prepare_sandbox_environment,
)
from dev_helper_bot.skills import build_system_prompt, default_skills_dir, load_skills  # noqa: E402
from dev_helper_bot.telemetry import (  # noqa: E402
    RUN_STATUS_LABELS,
    RUN_STATUS_LLM_ERROR,
    ObservingClient,
    RunRecorder,
    TelemetryStore,
)
from dev_helper_bot.tools import (  # noqa: E402
    EXEC_TOOL_SPEC,
    LIST_TOOL_SPEC,
    SEARCH_TOOL_SPEC,
)

BENCHMARK_LABEL = "benchmark"
"""Фиксированная метка прогонов харнесса (design D4): аудит фильтрует по ней."""

DEFAULT_DB_PATH = "~/.local/share/dev-helper-bot/benchmark.db"
DEFAULT_MEMORY_DB_PATH = "~/.local/share/dev-helper-bot/benchmark-memory.db"

FIXTURE_HOST_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_project"
FIXTURE_SANDBOX_PATH = "/work/sample_project"

AGENT_TOOLS = [EXEC_TOOL_SPEC, SEARCH_TOOL_SPEC, LIST_TOOL_SPEC]

CLOSE_SESSION = "close"
"""Маркер шага сценария: закрыть сессию памяти (аналог команды /new)."""


class BenchmarkError(RuntimeError):
    """Ошибка харнесса: понятное сообщение + ненулевой exit code."""


@dataclass(frozen=True)
class Scenario:
    name: str
    chat_id: int
    description: str
    steps: tuple[str, ...]
    needs_fixture: bool = False


@dataclass
class Harness:
    """Контекст прогона: владеет всеми ресурсами конвейера (design D3)."""

    llm: object
    memory: MemoryStore
    telemetry: TelemetryStore
    executor: SandboxExecutor
    skills: dict[str, str] = field(default_factory=dict)

    async def process_message(self, chat_id: int, text: str) -> str:
        """Один прогон = одно сообщение, конвейер handle_text без Telegram."""
        system_prompt = build_system_prompt(self.skills, datetime.now())
        history: list[Message] = [{"role": "system", "content": system_prompt}]
        history += await self.memory.load_open_history(chat_id)
        await self.memory.append_user(chat_id, text)
        history.append({"role": "user", "content": text})

        recorder = RunRecorder(
            self.telemetry,
            chat_id,
            BENCHMARK_LABEL,
            price_input_per_m=obs_price_input_per_m(),
            price_output_per_m=obs_price_output_per_m(),
        )
        await recorder.start()
        client = ObservingClient(self.llm, recorder, model=llm_model_name())
        try:
            reply = await run_agent(
                client,
                history,
                tools=AGENT_TOOLS,
                executor=self.executor,
                history_search=ChatHistorySearcher(self.memory, chat_id),
                recorder=recorder,
            )
        except LLMUnavailable as exc:
            # Ветка handle_text: прогон финализируется ошибкой LLM (design D2)
            await recorder.finish(RUN_STATUS_LLM_ERROR)
            raise BenchmarkError(
                f"LLM недоступен во время прогона (чат {chat_id}): {exc}"
            ) from exc
        await self.memory.append_assistant(chat_id, reply)
        return reply


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="files",
        chat_id=101,
        description="Файловая задача: изучить проект, найти функцию, запустить тесты",
        needs_fixture=True,
        steps=(
            "В песочнице есть учебный проект sample_project (каталог "
            "/work/sample_project). Изучи его структуру, найди функцию "
            "calculate_total, объясни коротко, что она делает и в чём её баг, "
            "и запусти тесты проекта. В ответе: назначение функции, её баг и "
            "результат тестов.",
        ),
    ),
    Scenario(
        name="verbose",
        chat_id=102,
        description="Команда с объёмным выводом: seq 1 3000",
        steps=(
            "Выполни в песочнице команду seq 1 3000 (вывод будет большим) и "
            "посчитай сумму выведенных чисел отдельной командой. Ответь "
            "коротко: чему равна сумма чисел от 1 до 3000?",
        ),
    ),
    Scenario(
        name="history",
        chat_id=103,
        description="Поиск по истории: завершённая сессия, затем вопрос о прошлом",
        steps=(
            "Расскажи двумя предложениями, чем плоха архитектура "
            "big ball of mud.",
            CLOSE_SESSION,
            "Мы уже обсуждали с тобой темы по архитектуре. Посмотри историю "
            "наших бесед (инструментами) и напомни: о чём именно был мой "
            "прошлый вопрос?",
        ),
    ),
    Scenario(
        name="multiturn",
        chat_id=104,
        description="Многоходовая сессия: 3 сообщения в одной сессии памяти",
        steps=(
            "Будем вести заметки по шагам. Шаг 1: создай в песочнице файл "
            "notes.md с нумерованным списком трёх языков программирования.",
            "Шаг 2: допиши в notes.md четвёртый язык и покажи итоговое "
            "содержимое файла.",
            "Шаг 3: не перечитывая файл, напомни, какие три языка мы "
            "записали на шаге 1, и проверь командой количество строк в "
            "notes.md.",
        ),
    ),
)


async def preflight_llm(llm: object) -> None:
    """Проверка доступности LLM до прогонов (спека: без частичных прогонов).

    Один пробный complete() без recorder'а — в телеметрии не отражается.
    """
    try:
        await llm.complete([{"role": "user", "content": "ping"}])
    except LLMUnavailable as exc:
        raise BenchmarkError(
            f"LLM-эндпоинт недоступен, бенчмарк не запущен: {exc}. "
            "Проверьте, что сервер запущен (LLM_BASE_URL)."
        ) from exc


async def preflight_docker() -> None:
    """Startup-последовательность песочницы как в main(): info → sweep → image."""
    try:
        await prepare_sandbox_environment()
    except SystemExit as exc:
        raise BenchmarkError(str(exc)) from exc


async def deploy_fixture(executor: SandboxExecutor) -> None:
    """Доставка фикстуры в /work (design D5): служебный exec → rm -rf → docker cp."""
    if not FIXTURE_HOST_DIR.is_dir():
        raise BenchmarkError(f"Фикстура не найдена: {FIXTURE_HOST_DIR}")
    # Служебный exec гарантирует контейнер-жильца и даёт его имя
    result = await executor.execute("true")
    if result.exit_code != 0:
        raise BenchmarkError(
            f"Служебный exec в песочнице не удался: {result.stderr.strip()}"
        )
    container_id = executor.container_id
    if not container_id:
        raise BenchmarkError("Не удалось узнать контейнер песочницы")
    # Идемпотентность через пересоздание: чистим цель и копируем поверх
    await executor.execute(f"rm -rf {FIXTURE_SANDBOX_PATH}")
    process = await asyncio.create_subprocess_exec(
        "docker",
        "cp",
        str(FIXTURE_HOST_DIR),
        f"{container_id}:{FIXTURE_SANDBOX_PATH}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr_b = await process.communicate()
    if process.returncode != 0:
        raise BenchmarkError(
            f"docker cp фикстуры не удался: {stderr_b.decode(errors='replace').strip()}"
        )


async def run_scenario(scenario: Scenario, harness: Harness) -> list[str]:
    """Исполняет шаги сценария; возвращает статусы созданных прогонов."""
    statuses: list[str] = []
    for step in scenario.steps:
        if step == CLOSE_SESSION:
            await harness.memory.close_session(scenario.chat_id)
            print(f"    сессия чата {scenario.chat_id} закрыта (аналог /new)")
            continue
        reply = await harness.process_message(scenario.chat_id, step)
        status = await _last_run_status(harness.telemetry)
        statuses.append(status)
        print(f"    прогон завершён: {RUN_STATUS_LABELS.get(status, status)}")
        print(f"    ответ: {_preview(reply)}")
    return statuses


async def _last_run_status(store: TelemetryStore) -> str | None:
    import sqlite3

    with sqlite3.connect(store.path) as conn:
        (status,) = conn.execute(
            "SELECT status FROM runs ORDER BY id DESC LIMIT 1"
        ).fetchone() or (None,)
    return status


def _preview(text: str, chars: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= chars else text[:chars] + "…"


def resolve_path(explicit: str | None, env_var: str, default: str) -> Path:
    return Path(
        os.path.expanduser(explicit or os.getenv(env_var) or default)
    )


async def run(args: argparse.Namespace) -> int:
    load_dotenv()

    db_path = resolve_path(args.db, "BENCHMARK_DB_PATH", DEFAULT_DB_PATH)
    memory_path = resolve_path(
        args.memory_db, "BENCHMARK_MEMORY_DB_PATH", DEFAULT_MEMORY_DB_PATH
    )
    scenarios = [
        s for s in SCENARIOS if not args.only or s.name in args.only
    ]
    unknown = sorted(set(args.only or ()) - {s.name for s in SCENARIOS})
    if unknown:
        raise BenchmarkError(
            f"Неизвестные сценарии: {', '.join(unknown)}. "
            f"Доступны: {', '.join(s.name for s in SCENARIOS)}."
        )
    if not scenarios:
        raise BenchmarkError("Набор сценариев пуст")

    await preflight_docker()
    llm = make_llm()
    await preflight_llm(llm)

    # Память бенчмарка — scratch: пересоздаём для одинакового контекста
    # сценариев на каждом запуске (телеметрия при этом накапливается).
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(memory_path) + suffix).unlink(missing_ok=True)

    memory = MemoryStore(memory_path)
    telemetry = TelemetryStore(db_path)
    executor = SandboxExecutor()
    await memory.open()
    try:
        await telemetry.open()
    except Exception as exc:
        await memory.close()
        raise BenchmarkError(f"Не удалось открыть БД телеметрии {db_path}: {exc}") from exc

    print(f"Бенчмарк: БД телеметрии {db_path}, метка {BENCHMARK_LABEL!r}")
    total_runs = 0
    try:
        if any(s.needs_fixture for s in scenarios):
            print("Доставка фикстуры sample_project в песочницу…")
            await deploy_fixture(executor)
            print(f"  фикстура в {FIXTURE_SANDBOX_PATH}")
        harness = Harness(
            llm=llm,
            memory=memory,
            telemetry=telemetry,
            executor=executor,
            skills=load_skills(default_skills_dir()),
        )
        for scenario in scenarios:
            print(f"\nСценарий {scenario.name} (чат {scenario.chat_id}): "
                  f"{scenario.description}")
            statuses = await run_scenario(scenario, harness)
            total_runs += len(statuses)
    finally:
        await executor.stop()
        await memory.close()
        await telemetry.close()

    print(f"\nГотово: прогонов создано {total_runs}, БД {db_path}, "
          f"метка {BENCHMARK_LABEL!r}.")
    print(f"Отчёт: python scripts/obs-audit.py --db {db_path} "
          f"--label {BENCHMARK_LABEL}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Генерация телеметрии benchmark-прогонов агента "
                    "(прямой прогон сценариев, мимо Telegram)."
    )
    parser.add_argument("--db", help="файл БД телеметрии бенчмарка "
                                     "(по умолчанию BENCHMARK_DB_PATH)")
    parser.add_argument("--memory-db", help="файл БД памяти бенчмарка "
                                            "(по умолчанию BENCHMARK_MEMORY_DB_PATH)")
    parser.add_argument("--only", action="append", default=[],
                        help="исполнить только сценарий с этим именем "
                             "(можно несколько раз)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        return asyncio.run(run(args))
    except BenchmarkError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
