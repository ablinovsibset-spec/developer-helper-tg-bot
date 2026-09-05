"""Хранилище телеметрии агентных прогонов (change add-agent-observability).

Отдельная SQLite-база по паттерну memory.db (design D4): aiosqlite, WAL,
файл на VM-локальном диске, жизненным циклом владеет main. Запись
best-effort: любой сбой логируется warning'ом и не проникает наружу —
наблюдение не может стать причиной отказа агента (спека agent-observability).

Схема толерантна к неполному usage (design D8): колонки токенов nullable,
рядом — raw usage JSON поставщика. Absence ≠ 0: null значит «поставщик
не отдал», ноль значит «поставщик сказал ноль».
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from dev_helper_bot.llm import (
    LLMClient,
    LLMUnavailable,
    Message,
    ResponseFormat,
    ToolSpec,
    Usage,
)

log = logging.getLogger(__name__)

RUN_STATUS_SUCCESS = "success"
RUN_STATUS_STEPS_EXHAUSTED = "steps_exhausted"
RUN_STATUS_VALIDATION_ERROR = "validation_error"
RUN_STATUS_LLM_ERROR = "llm_error"

RUN_STATUS_LABELS = {
    RUN_STATUS_SUCCESS: "успех",
    RUN_STATUS_STEPS_EXHAUSTED: "остановка по лимиту шагов",
    RUN_STATUS_VALIDATION_ERROR: "ошибка валидации ответа",
    RUN_STATUS_LLM_ERROR: "ошибка LLM",
    None: "не завершён",
}

TOOL_OUTPUT_CHARS_PER_TOKEN = 4
"""Оценка токенов tool-вывода (design D6): честного токенайзера нет,
~4 символа на токен достаточно для атрибуции «топ инструментов»."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    label TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT,
    llm_calls INTEGER,
    tool_calls INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    reasoning_tokens INTEGER,
    estimated_cost REAL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    created_at TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    model TEXT,
    latency_ms REAL NOT NULL,
    ok INTEGER NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    reasoning_tokens INTEGER,
    messages_count INTEGER NOT NULL,
    prompt_chars INTEGER NOT NULL,
    estimated_cost REAL,
    usage_raw TEXT
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    created_at TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    input_size INTEGER NOT NULL,
    output_size INTEGER NOT NULL,
    output_tokens INTEGER,
    duration_ms REAL NOT NULL,
    ok INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id, id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id, id);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def estimate_tool_tokens(output_size: int) -> int:
    """Оценка числа токенов результата инструмента: chars/4 (design D6)."""
    return output_size // TOOL_OUTPUT_CHARS_PER_TOKEN


def estimate_cost(
    usage: Usage | None, price_input_per_m: float, price_output_per_m: float
) -> float | None:
    """Виртуальная стоимость вызова, $ (design D5).

    Считается на момент записи по конфигурируемому прайсу ($ за 1M токенов);
    смена прайса историю не переписывает. Кэшированные токены учитываются
    по цене входных (в OpenAI-семантике они уже внутри input_tokens; если
    input отсутствует, считаем их отдельно). Нет токенов — null, а не 0:
    стоимость неизвестна, но нулевой прайс даёт честный 0.0.
    """
    if usage is None:
        return None
    input_effective = usage["input_tokens"]
    if input_effective is None:
        input_effective = usage["cached_tokens"]
    if input_effective is None and usage["output_tokens"] is None:
        return None
    return (
        (input_effective or 0) * price_input_per_m / 1_000_000
        + (usage["output_tokens"] or 0) * price_output_per_m / 1_000_000
    )


@dataclass(frozen=True)
class LLMCallRecord:
    """Одна строка llm_calls: попытка вызова LLM в прогоне."""

    run_id: int | None
    created_at: str
    turn_number: int
    model: str | None
    latency_ms: float
    ok: bool
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    messages_count: int
    prompt_chars: int
    estimated_cost: float | None
    usage_raw: str | None


@dataclass(frozen=True)
class ToolCallRecord:
    """Одна строка tool_calls: вызов инструмента в прогоне."""

    run_id: int | None
    created_at: str
    turn_number: int
    tool_name: str
    input_size: int
    output_size: int
    output_tokens: int | None
    duration_ms: float
    ok: bool


class TelemetryStore:
    """Хранилище телеметрии: runs/llm_calls/tool_calls.

    Паттерн MemoryStore (design D4): open при старте main, close в finally,
    WAL, FK на run. Все публичные методы best-effort: исключение записи
    логируется warning'ом и не проникает к вызывающему.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()
        self._db: aiosqlite.Connection | None = None

    @property
    def path(self) -> Path:
        return self._path

    async def open(self) -> None:
        """Открывает/создаёт файл БД, включает WAL и FK, создаёт схему."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        db, self._db = self._db, None
        if db is not None:
            await db.close()

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("TelemetryStore не открыт: сначала вызовите open()")
        return self._db

    async def _execute(self, sql: str, params: tuple) -> int | None:
        """Выполняет statement и коммитит; сбой — warning, не исключение.

        Возвращает lastrowid (для INSERT) или None при сбое.
        """
        try:
            cursor = await self._conn.execute(sql, params)
            await self._conn.commit()
            return cursor.lastrowid
        except Exception:
            log.warning(
                "telemetry write failed (%s)", sql.split("(")[0].strip(), exc_info=True
            )
            return None

    async def insert_run(self, chat_id: int, label: str | None) -> int | None:
        """Создаёт прогон; возвращает run_id или None при сбое записи."""
        return await self._execute(
            "INSERT INTO runs (chat_id, label, started_at) VALUES (?, ?, ?)",
            (chat_id, label, _utcnow_iso()),
        )

    async def insert_llm_call(self, record: LLMCallRecord) -> None:
        await self._execute(
            "INSERT INTO llm_calls (run_id, created_at, turn_number, model, "
            "latency_ms, ok, input_tokens, output_tokens, cached_tokens, "
            "reasoning_tokens, messages_count, prompt_chars, estimated_cost, "
            "usage_raw) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.run_id,
                record.created_at,
                record.turn_number,
                record.model,
                record.latency_ms,
                int(record.ok),
                record.input_tokens,
                record.output_tokens,
                record.cached_tokens,
                record.reasoning_tokens,
                record.messages_count,
                record.prompt_chars,
                record.estimated_cost,
                record.usage_raw,
            ),
        )

    async def insert_tool_call(self, record: ToolCallRecord) -> None:
        await self._execute(
            "INSERT INTO tool_calls (run_id, created_at, turn_number, tool_name, "
            "input_size, output_size, output_tokens, duration_ms, ok) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.run_id,
                record.created_at,
                record.turn_number,
                record.tool_name,
                record.input_size,
                record.output_size,
                record.output_tokens,
                record.duration_ms,
                int(record.ok),
            ),
        )

    async def finish_run(self, run_id: int | None, status: str) -> None:
        """Закрывает прогон: статус, время завершения, агрегаты по вызовам."""
        if run_id is None:
            return  # run не создался (сбой записи) — финализировать нечего
        await self._execute(
            "UPDATE runs SET finished_at = ?, status = ?, "
            "llm_calls = (SELECT COUNT(*) FROM llm_calls WHERE run_id = ?), "
            "tool_calls = (SELECT COUNT(*) FROM tool_calls WHERE run_id = ?), "
            "input_tokens = (SELECT SUM(input_tokens) FROM llm_calls WHERE run_id = ?), "
            "output_tokens = (SELECT SUM(output_tokens) FROM llm_calls WHERE run_id = ?), "
            "cached_tokens = (SELECT SUM(cached_tokens) FROM llm_calls WHERE run_id = ?), "
            "reasoning_tokens = "
            "  (SELECT SUM(reasoning_tokens) FROM llm_calls WHERE run_id = ?), "
            "estimated_cost = "
            "  (SELECT SUM(estimated_cost) FROM llm_calls WHERE run_id = ?) "
            "WHERE id = ?",
            (
                _utcnow_iso(),
                status,
                run_id,
                run_id,
                run_id,
                run_id,
                run_id,
                run_id,
                run_id,
                run_id,
            ),
        )


class RunRecorder:
    """Телеметрия одного прогона = одной обработки сообщения (design D2).

    Создаётся в handle_text на каждое сообщение: открывает run (chat_id,
    label), считает turn_number по вызовам complete() и финализирует прогон
    со статусом и агрегатами. Стоимость считается на записи по прайсу,
    переданному при создании (design D5).
    """

    def __init__(
        self,
        store: TelemetryStore,
        chat_id: int,
        label: str | None = None,
        *,
        price_input_per_m: float = 0.0,
        price_output_per_m: float = 0.0,
    ) -> None:
        self._store = store
        self._chat_id = chat_id
        self._label = label
        self._price_input_per_m = price_input_per_m
        self._price_output_per_m = price_output_per_m
        self.run_id: int | None = None
        self._turn = 0

    async def start(self) -> None:
        self.run_id = await self._store.insert_run(self._chat_id, self._label)

    def next_turn_number(self) -> int:
        """Порядковый номер вызова complete() в прогоне (с 1); ошибочные
        попытки тоже потребляют номер — фиксируется каждая."""
        self._turn += 1
        return self._turn

    async def record_llm_call(
        self,
        *,
        turn_number: int,
        model: str | None,
        latency_ms: float,
        ok: bool,
        usage: Usage | None,
        messages_count: int,
        prompt_chars: int,
        created_at: str | None = None,
    ) -> None:
        await self._store.insert_llm_call(
            LLMCallRecord(
                run_id=self.run_id,
                created_at=created_at or _utcnow_iso(),
                turn_number=turn_number,
                model=model,
                latency_ms=latency_ms,
                ok=ok,
                input_tokens=usage["input_tokens"] if usage else None,
                output_tokens=usage["output_tokens"] if usage else None,
                cached_tokens=usage["cached_tokens"] if usage else None,
                reasoning_tokens=usage["reasoning_tokens"] if usage else None,
                messages_count=messages_count,
                prompt_chars=prompt_chars,
                estimated_cost=(
                    estimate_cost(
                        usage, self._price_input_per_m, self._price_output_per_m
                    )
                    if ok
                    else None
                ),
                usage_raw=(
                    json.dumps(usage["raw"], ensure_ascii=False, sort_keys=True)
                    if usage
                    else None
                ),
            )
        )

    async def record_tool_call(
        self,
        *,
        turn_number: int,
        tool_name: str,
        input_size: int,
        output_size: int,
        duration_ms: float,
        ok: bool,
    ) -> None:
        await self._store.insert_tool_call(
            ToolCallRecord(
                run_id=self.run_id,
                created_at=_utcnow_iso(),
                turn_number=turn_number,
                tool_name=tool_name,
                input_size=input_size,
                output_size=output_size,
                output_tokens=estimate_tool_tokens(output_size),
                duration_ms=duration_ms,
                ok=ok,
            )
        )

    async def finish(self, status: str) -> None:
        await self._store.finish_run(self.run_id, status)


class ObservingClient:
    """Декоратор над LLMClient: пишет телеметрию каждого complete() (design D2).

    Прозрачен для вызывающего кода: тот же контракт, те же ответы и
    исключения. Latency меряется строго вокруг complete(); неуспешная
    попытка (LLMUnavailable) фиксируется с исходом «ошибка», фактической
    длительностью и без токенов, после чего исключение пробрасывается.
    """

    def __init__(self, inner: LLMClient, recorder: RunRecorder, model: str | None = None) -> None:
        self._inner = inner
        self._recorder = recorder
        self._model = model

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_format: ResponseFormat | None = None,
    ):
        turn_number = self._recorder.next_turn_number()
        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        started = time.perf_counter()
        created_at = _utcnow_iso()
        try:
            turn = await self._inner.complete(messages, tools, response_format)
        except LLMUnavailable:
            latency_ms = (time.perf_counter() - started) * 1000
            await self._recorder.record_llm_call(
                turn_number=turn_number,
                model=self._model,
                latency_ms=latency_ms,
                ok=False,
                usage=None,
                messages_count=len(messages),
                prompt_chars=prompt_chars,
                created_at=created_at,
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        await self._recorder.record_llm_call(
            turn_number=turn_number,
            model=self._model,
            latency_ms=latency_ms,
            ok=True,
            usage=turn.get("usage"),
            messages_count=len(messages),
            prompt_chars=prompt_chars,
            created_at=created_at,
        )
        return turn
