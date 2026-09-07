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

# Дефолты виртуального прайса аудита — те же, что у config.py и obs-audit.py.
# Определены в начале модуля: используются как default-значения параметров
# read_audit (вычисляются при определении класса, см. design D5).
DEFAULT_AUDIT_PRICE_INPUT_PER_M = 0.11
DEFAULT_AUDIT_PRICE_OUTPUT_PER_M = 0.60
DEFAULT_AUDIT_CACHED_PRICE_MULTIPLIER = 0.25
"""Дефолты виртуального прайса аудита: входные/выходные $ за 1M и доля
входной цены для кэшированных токенов в эффективной стоимости (design D5)."""

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
    estimated_cost REAL,
    billed_cost REAL
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
    prompt_roles TEXT,
    estimated_cost REAL,
    billed_cost REAL,
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


PROMPT_ROLE_NAMES = ("system", "user", "assistant", "tool")
"""Роли сообщений, по которым телеметрия хранит разбивку промпта (design D1)."""


def count_prompt_roles(messages: list[Message]) -> dict[str, int]:
    """Разбивка промпта в символах по ролям сообщений (design D1).

    Та же конвенция содержания, что у prompt_chars (len(content or ""));
    роли вне четырёх известных игнорируются. Все четыре ключа присутствуют
    всегда — 0 значит «роли в промпте не было», null в БД значит «разбивка
    не считалась» (запись до миграции).
    """
    roles = dict.fromkeys(PROMPT_ROLE_NAMES, 0)
    for message in messages:
        role = message.get("role")
        if role in roles:
            roles[role] += len(message.get("content") or "")
    return roles


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
    prompt_roles: str | None
    estimated_cost: float | None
    billed_cost: float | None
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
        await self._migrate()
        await self._db.commit()

    async def _migrate(self) -> None:
        """Idempotent-миграция старой базы (design D2): колонки, появившиеся
        после первого релиза схемы, добавляются `ALTER TABLE`, если их нет.
        Старые записи остаются null — absence ≠ 0 (как и для usage)."""
        cursor = await self._conn.execute("PRAGMA table_info(llm_calls)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "prompt_roles" not in columns:
            await self._conn.execute(
                "ALTER TABLE llm_calls ADD COLUMN prompt_roles TEXT"
            )
        if "billed_cost" not in columns:
            await self._conn.execute(
                "ALTER TABLE llm_calls ADD COLUMN billed_cost REAL"
            )
        cursor = await self._conn.execute("PRAGMA table_info(runs)")
        run_columns = {row[1] for row in await cursor.fetchall()}
        if "billed_cost" not in run_columns:
            await self._conn.execute(
                "ALTER TABLE runs ADD COLUMN billed_cost REAL"
            )

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
            "reasoning_tokens, messages_count, prompt_chars, prompt_roles, "
            "estimated_cost, billed_cost, usage_raw) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                record.prompt_roles,
                record.estimated_cost,
                record.billed_cost,
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
            "  (SELECT SUM(estimated_cost) FROM llm_calls WHERE run_id = ?), "
            "billed_cost = "
            "  (SELECT SUM(billed_cost) FROM llm_calls WHERE run_id = ?) "
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
                run_id,
            ),
        )

    # --- Read API (change add-obs-web-dashboard, design D3) -----------------

    async def _read_all(self, sql: str, params: tuple) -> list[tuple]:
        """Выполняет SELECT и возвращает строки; при сбое SQL — warning и []."""
        try:
            cursor = await self._conn.execute(sql, params)
            return list(await cursor.fetchall())
        except Exception:
            log.warning("telemetry read failed (%s)", sql.split("FROM")[0].strip(), exc_info=True)
            return []

    async def _read_one(self, sql: str, params: tuple) -> tuple | None:
        """Выполняет SELECT одной строки; при сбое SQL — warning и None."""
        try:
            cursor = await self._conn.execute(sql, params)
            return await cursor.fetchone()
        except Exception:
            log.warning("telemetry read failed (%s)", sql.split("FROM")[0].strip(), exc_info=True)
            return None

    async def read_summary(
        self, label: str | None = None, since: str | None = None
    ) -> Summary:
        """Агрегаты телеметрии по прогонам с фильтрами label/since (design D3).

        Семантика как у obs-dashboard.render_aggregates: COALESCE для сумм,
        которые осмысленны нулём (входные/выходные/стоимость/счётчики), и
        SUM без coalesce для nullable-полей поставщика (cached/reasoning —
        None, если ни одна строка не отдала). cache_hit_rate считается по
        строкам с известным cached_tokens. Пустая база — Summary с нулями.
        """
        params = (label, label, since, since)
        row = await self._read_one(
            "SELECT COUNT(*), COUNT(status), "
            "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
            "SUM(cached_tokens), SUM(reasoning_tokens), "
            "COALESCE(SUM(estimated_cost), 0), "
            "COALESCE(SUM(llm_calls), 0), COALESCE(SUM(tool_calls), 0) "
            "FROM runs WHERE (? IS NULL OR label = ?) "
            "AND (? IS NULL OR started_at >= ?)",
            params,
        )
        if row is None:
            return Summary(
                total_runs=0, finished_runs=0, input_tokens=0, output_tokens=0,
                cached_tokens=None, reasoning_tokens=None, reasoning_available=False,
                estimated_cost=0.0, llm_calls=0, tool_calls=0,
                status_breakdown=[], cache_hit_rate=None, top_tools=[],
            )
        (total, finished, input_t, output_t, cached_t, reasoning_t,
         cost, llm_total, tool_total) = row

        status_rows = await self._read_all(
            "SELECT status, COUNT(*) FROM runs "
            "WHERE (? IS NULL OR label = ?) AND (? IS NULL OR started_at >= ?) "
            "GROUP BY status ORDER BY COUNT(*) DESC",
            params,
        )
        status_breakdown = [(s, c) for s, c in status_rows]

        cache_row = await self._read_one(
            "SELECT SUM(c.cached_tokens), SUM(c.input_tokens) "
            "FROM llm_calls c JOIN runs r ON c.run_id = r.id "
            "WHERE (? IS NULL OR r.label = ?) AND (? IS NULL OR r.started_at >= ?) "
            "AND c.cached_tokens IS NOT NULL",
            params,
        )
        cached_sum, cached_input_sum = cache_row if cache_row is not None else (None, None)

        reasoning_row = await self._read_one(
            "SELECT EXISTS(SELECT 1 FROM llm_calls c JOIN runs r ON c.run_id = r.id "
            "WHERE (? IS NULL OR r.label = ?) AND (? IS NULL OR r.started_at >= ?) "
            "AND c.reasoning_tokens IS NOT NULL)",
            params,
        )
        reasoning_available = bool(reasoning_row[0]) if reasoning_row else False

        cache_hit_rate: float | None = None
        if cached_sum is not None and cached_input_sum:
            cache_hit_rate = cached_sum / cached_input_sum * 100

        top_rows = await self._read_all(
            "SELECT tc.tool_name, SUM(tc.output_tokens), COUNT(*) "
            "FROM tool_calls tc JOIN runs r ON tc.run_id = r.id "
            "WHERE (? IS NULL OR r.label = ?) AND (? IS NULL OR r.started_at >= ?) "
            "GROUP BY tc.tool_name ORDER BY SUM(tc.output_tokens) DESC LIMIT 5",
            params,
        )
        grand_row = await self._read_one(
            "SELECT COALESCE(SUM(tc.output_tokens), 0) "
            "FROM tool_calls tc JOIN runs r ON tc.run_id = r.id "
            "WHERE (? IS NULL OR r.label = ?) AND (? IS NULL OR r.started_at >= ?)",
            params,
        )
        grand_total = grand_row[0] if grand_row else 0
        top_tools = [
            ToolTotal(
                name=name, output_tokens=tokens or 0, calls=calls,
                share_percent=(tokens or 0) / grand_total * 100 if grand_total else 0.0,
            )
            for name, tokens, calls in top_rows
        ]

        return Summary(
            total_runs=total, finished_runs=finished,
            input_tokens=input_t, output_tokens=output_t,
            cached_tokens=cached_t, reasoning_tokens=reasoning_t,
            reasoning_available=reasoning_available,
            estimated_cost=cost, llm_calls=llm_total, tool_calls=tool_total,
            status_breakdown=status_breakdown,
            cache_hit_rate=cache_hit_rate, top_tools=top_tools,
        )

    async def read_runs(
        self, label: str | None = None, since: str | None = None
    ) -> list[RunSummary]:
        """Реестр прогонов с фильтрами label/since, упорядоченный по id (design D3)."""
        rows = await self._read_all(
            "SELECT id, chat_id, label, started_at, finished_at, status, "
            "llm_calls, tool_calls, input_tokens, output_tokens, "
            "cached_tokens, reasoning_tokens, estimated_cost "
            "FROM runs WHERE (? IS NULL OR label = ?) "
            "AND (? IS NULL OR started_at >= ?) ORDER BY id",
            (label, label, since, since),
        )
        return [
            RunSummary(
                id=rid, chat_id=chat_id, label=run_label, started_at=started,
                finished_at=finished, status=status, llm_calls=llm_n,
                tool_calls=tool_n, input_tokens=in_t, output_tokens=out_t,
                cached_tokens=cached, reasoning_tokens=reasoning,
                estimated_cost=cost,
            )
            for (rid, chat_id, run_label, started, finished, status, llm_n,
                 tool_n, in_t, out_t, cached, reasoning, cost) in rows
        ]

    async def read_run(self, run_id: int) -> RunDetail | None:
        """Прогон + его timeline (LLM-ходы и tool-вызовы); None, если run_id
        неизвестен (design D3, спека agent-observability «Timeline прогона»)."""
        row = await self._read_one(
            "SELECT id, chat_id, label, started_at, finished_at, status, "
            "llm_calls, tool_calls, input_tokens, output_tokens, "
            "cached_tokens, reasoning_tokens, estimated_cost "
            "FROM runs WHERE id = ?",
            (run_id,),
        )
        if row is None:
            return None
        (rid, chat_id, run_label, started, finished, status, llm_n,
         tool_n, in_t, out_t, cached, reasoning, cost) = row
        run = RunSummary(
            id=rid, chat_id=chat_id, label=run_label, started_at=started,
            finished_at=finished, status=status, llm_calls=llm_n,
            tool_calls=tool_n, input_tokens=in_t, output_tokens=out_t,
            cached_tokens=cached, reasoning_tokens=reasoning,
            estimated_cost=cost,
        )
        llm_rows = await self._read_all(
            "SELECT id, turn_number, ok, latency_ms, input_tokens, output_tokens, "
            "cached_tokens, reasoning_tokens, estimated_cost, model "
            "FROM llm_calls WHERE run_id = ? ORDER BY turn_number, id",
            (run_id,),
        )
        llm_calls = [
            LLMTurnRow(
                id=rid, turn_number=turn, ok=bool(ok), latency_ms=latency,
                input_tokens=in_t, output_tokens=out_t, cached_tokens=cached,
                reasoning_tokens=reasoning, estimated_cost=cost, model=model,
            )
            for (rid, turn, ok, latency, in_t, out_t, cached, reasoning, cost, model)
            in llm_rows
        ]
        tool_rows = await self._read_all(
            "SELECT id, turn_number, tool_name, ok, duration_ms, output_tokens, "
            "input_size, output_size FROM tool_calls WHERE run_id = ? "
            "ORDER BY turn_number, id",
            (run_id,),
        )
        tool_calls = [
            ToolCallRow(
                id=rid, turn_number=turn, tool_name=name, ok=bool(ok),
                duration_ms=duration, output_tokens=tokens, input_size=in_size,
                output_size=out_size,
            )
            for (rid, turn, name, ok, duration, tokens, in_size, out_size)
            in tool_rows
        ]
        return RunDetail(run=run, llm_calls=llm_calls, tool_calls=tool_calls)

    async def read_audit(
        self,
        label: str | None = None,
        since: str | None = None,
        *,
        price_input_per_m: float = DEFAULT_AUDIT_PRICE_INPUT_PER_M,
        price_output_per_m: float = DEFAULT_AUDIT_PRICE_OUTPUT_PER_M,
        cached_price_multiplier: float = DEFAULT_AUDIT_CACHED_PRICE_MULTIPLIER,
    ) -> Audit:
        """Аудит потребления токенов по отфильтрованным прогонам (design D5).

        Семантика Q1–Q5 идентична scripts/obs-audit.py: топ инструментов,
        самый дорогой ход, рост типов контекста, повторные токены, стоимость
        (фактический биллинг + сырая/эффективная оценки). Пустая база —
        Audit с runs_count=0 и пустыми секциями.
        """
        params = (label, label, since, since)
        run_rows = await self._read_all(
            "SELECT id FROM runs WHERE (? IS NULL OR label = ?) "
            "AND (? IS NULL OR started_at >= ?) ORDER BY id",
            params,
        )
        run_ids = [row[0] for row in run_rows]
        if not run_ids:
            return Audit(
                runs_count=0, label=label, since=since,
                top_tools=[], expensive_turns=[], most_expensive_turn=None,
                single_turn_only=False, context_types=None, repeats=None,
                cost=None,
            )

        placeholders = ",".join("?" * len(run_ids))
        llm_rows = await self._read_all(
            f"SELECT run_id, turn_number, input_tokens, output_tokens, "
            f"cached_tokens, prompt_roles, billed_cost "
            f"FROM llm_calls WHERE run_id IN ({placeholders}) "
            f"ORDER BY run_id, turn_number, id",
            tuple(run_ids),
        )
        llm_calls: list[dict] = []
        for (run_id, turn, input_tokens, output_tokens, cached_tokens,
             prompt_roles, billed_cost) in llm_rows:
            roles = None
            if prompt_roles is not None:
                try:
                    roles = json.loads(prompt_roles)
                except (ValueError, TypeError):
                    roles = None  # битый JSON трактуется как отсутствие разбивки
            llm_calls.append({
                "run_id": run_id, "turn": turn, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "cached_tokens": cached_tokens,
                "roles": roles, "billed_cost": billed_cost,
            })

        tool_rows = await self._read_all(
            f"SELECT tool_name, output_tokens FROM tool_calls "
            f"WHERE run_id IN ({placeholders})",
            tuple(run_ids),
        )
        tool_calls = [
            {"tool_name": name, "output_tokens": tokens or 0}
            for name, tokens in tool_rows
        ]

        top_tools = _compute_top_tools(tool_calls)
        expensive_turns, most_expensive_turn, single_turn_only = (
            _compute_expensive_turns(llm_calls)
        )
        context_types = _compute_context_types(llm_calls)
        repeats = _compute_repeats(llm_calls)
        cost = _compute_cost(
            llm_calls, price_input_per_m, price_output_per_m,
            cached_price_multiplier,
        )
        return Audit(
            runs_count=len(run_ids), label=label, since=since,
            top_tools=top_tools, expensive_turns=expensive_turns,
            most_expensive_turn=most_expensive_turn,
            single_turn_only=single_turn_only, context_types=context_types,
            repeats=repeats, cost=cost,
        )


# --- Read API (change add-obs-web-dashboard, design D3) ---------------------
# Хранилище получает async read-методы для агрегатов, реестра/timeline
# прогонов и данных аудита. Веб читает только через них. Null-tolerant:
# absence ≠ 0 — nullable-поля поставщика сохраняются как None при чтении.
# Сбой SQL логируется warning'ом и отдаёт пустой результат; store закрыт —
# _conn поднимает RuntimeError (веб трактует как «телеметрия недоступна»).


@dataclass(frozen=True)
class ToolTotal:
    """Строка «топ инструментов»: имя, оценочные токены вывода, число вызовов,
    доля от общего output всех инструментов (в %)."""

    name: str
    output_tokens: int
    calls: int
    share_percent: float


@dataclass(frozen=True)
class Summary:
    """Агрегаты телеметрии по отфильтрованным прогонам (design D3).

    cached_tokens/reasoning_tokens — SUM (None, если ни одна строка не
    отдала значение); reasoning_available — отдельный флаг, чтобы UI
    отличал «поставщик не отдаёт» от «нулевая сумма». cache_hit_rate —
    доля кэшированных во входных по строкам, где кэш известен (None, если
    таких строк нет).
    """

    total_runs: int
    finished_runs: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int | None
    reasoning_tokens: int | None
    reasoning_available: bool
    estimated_cost: float
    llm_calls: int
    tool_calls: int
    status_breakdown: list[tuple[str | None, int]]
    cache_hit_rate: float | None
    top_tools: list[ToolTotal]


@dataclass(frozen=True)
class RunSummary:
    """Краткая запись прогона для реестра и заголовка timeline."""

    id: int
    chat_id: int
    label: str | None
    started_at: str
    finished_at: str | None
    status: str | None
    llm_calls: int | None
    tool_calls: int | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    estimated_cost: float | None


@dataclass(frozen=True)
class LLMTurnRow:
    """Один LLM-ход из timeline прогона (design D3)."""

    id: int
    turn_number: int
    ok: bool
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    estimated_cost: float | None
    model: str | None


@dataclass(frozen=True)
class ToolCallRow:
    """Один вызов инструмента из timeline прогона (design D3)."""

    id: int
    turn_number: int
    tool_name: str
    ok: bool
    duration_ms: float
    output_tokens: int | None
    input_size: int
    output_size: int


@dataclass(frozen=True)
class RunDetail:
    """Прогон + его timeline: LLM-ходы и tool-вызовы отдельными списками
    (порядок хода сохраняется внутри каждого; UI мерджит по turn_number)."""

    run: RunSummary
    llm_calls: list[LLMTurnRow]
    tool_calls: list[ToolCallRow]


# --- Аудит потребления токенов (change add-obs-web-dashboard, design D5) -----
# Семантика Q1–Q5 перенесена из scripts/obs-audit.py в чистые функции на
# загруженных данных, чтобы веб и CLI не расходились. CLI в этом change
# остаётся файл-ориентированным; сверка — тестами на тех же фикстурах.

TYPE_SYSTEM = "System prompt"
TYPE_SESSION_HISTORY = "История сессии (ход 1: user+assistant)"
TYPE_DIALOG_HISTORY = "История диалога (assistant)"
TYPE_USER = "Текущая задача (user сверх хода 1)"
TYPE_TOOLS = "Выводы инструментов (вкл. файлы)"
CONTEXT_TYPES = (
    TYPE_SYSTEM, TYPE_SESSION_HISTORY, TYPE_DIALOG_HISTORY, TYPE_USER, TYPE_TOOLS,
)


@dataclass(frozen=True)
class AuditTopTool:
    """Q1: инструмент с оценочными токенами вывода, числом вызовов и долей."""

    name: str
    output_tokens: int
    calls: int
    share_percent: float


@dataclass(frozen=True)
class AuditTurn:
    """Q2: ход со средним input_tokens, числом прогонов и ростом к предыдущему."""

    turn: int
    avg_input_tokens: float
    runs: int
    growth_percent: float | None  # None для первого хода


@dataclass(frozen=True)
class AuditContextTypes:
    """Q3: распределение символов контекста по типам и тренд долей по ходам."""

    totals: dict[str, int]  # type -> chars (ключи из CONTEXT_TYPES)
    total_chars: int
    shares: dict[str, float]  # type -> percent
    skipped_no_roles: int  # вызовы без разбивки, не учтённые
    trend_by_turn: dict[int, dict[str, float]]  # turn -> type -> avg share %


@dataclass(frozen=True)
class AuditRepeats:
    """Q4: повторно отправляемые токены (append-only история + межпрогонный слой)."""

    total_input: int
    new_tokens: int
    repeated_tokens: int
    repeated_share_percent: float
    inter_run_tokens: int  # оценка сверху ре-отправки истории сессии


@dataclass(frozen=True)
class AuditCost:
    """Q5: фактический биллинг поставщика + оценки по прайсу (сырая/эффективная).

    billed_total — сумма billed_cost (None, если ни один вызов его не отдал);
    raw_cost/effective_cost — None, если нет вызовов с input_tokens. has_cache_data
    отличает «поставщик не отдаёт кэш» (эффективная = сырая) от расчёта со скидкой.
    """

    billed_total: float | None
    billed_calls: int
    raw_cost: float | None
    effective_cost: float | None
    cached_share_percent: float | None
    cache_savings: float | None
    has_cache_data: bool
    total_input: int
    total_output: int
    total_cached: int
    price_input_per_m: float
    price_output_per_m: float
    cached_price_per_m: float


@dataclass(frozen=True)
class Audit:
    """Полный снимок аудита потребления токенов по отфильтрованным прогонам."""

    runs_count: int
    label: str | None
    since: str | None
    top_tools: list[AuditTopTool]
    expensive_turns: list[AuditTurn]
    most_expensive_turn: int | None
    single_turn_only: bool  # все прогоны по одному ходу — динамики роста нет
    context_types: AuditContextTypes | None  # None, если ни одной разбивки
    repeats: AuditRepeats | None  # None, если нет прогонов с input_tokens
    cost: AuditCost | None  # None, если нет input_tokens и нет billed_cost


def _group_by_run(calls: list[dict]) -> dict[int, list[dict]]:
    runs: dict[int, list[dict]] = {}
    for call in calls:
        runs.setdefault(call["run_id"], []).append(call)
    return runs


def _run_type_chars(run_calls: list[dict]) -> dict[str, int] | None:
    """Символы по типам контекста для одного прогона (design D8, как obs-audit).

    system → System prompt; user+assistant хода 1 → загруженная история
    сессии; assistant сверх хода 1 → история диалога; user сверх хода 1 →
    текущая задача; tool → выводы инструментов. None, если ни один ход
    прогона не имеет разбивки prompt_roles.
    """
    first = min((c["turn"] for c in run_calls), default=None)
    chars = dict.fromkeys(CONTEXT_TYPES, 0)
    has_roles = False
    for call in run_calls:
        roles = call["roles"]
        if roles is None:
            continue
        has_roles = True
        turn = call["turn"]
        chars[TYPE_SYSTEM] += roles.get("system", 0) or 0
        chars[TYPE_TOOLS] += roles.get("tool", 0) or 0
        user = roles.get("user", 0) or 0
        assistant = roles.get("assistant", 0) or 0
        if turn == first:
            chars[TYPE_SESSION_HISTORY] += user + assistant
        else:
            chars[TYPE_USER] += user
            chars[TYPE_DIALOG_HISTORY] += assistant
    return chars if has_roles else None


def _compute_top_tools(tool_calls: list[dict]) -> list[AuditTopTool]:
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for call in tool_calls:
        totals[call["tool_name"]] = totals.get(call["tool_name"], 0) + call["output_tokens"]
        counts[call["tool_name"]] = counts.get(call["tool_name"], 0) + 1
    grand_total = sum(totals.values())
    return [
        AuditTopTool(
            name=name, output_tokens=tokens, calls=counts[name],
            share_percent=tokens / grand_total * 100 if grand_total else 0.0,
        )
        for name, tokens in sorted(totals.items(), key=lambda kv: -kv[1])
    ]


def _compute_expensive_turns(calls: list[dict]) -> tuple[list[AuditTurn], int | None, bool]:
    """Q2: средние input_tokens по ходам, рост, самый дорогой ход.

    Возвращает (список ходов, самый_дорогой_ход, single_turn_only).
    """
    with_input = [c for c in calls if c["input_tokens"] is not None]
    if not with_input:
        return [], None, False
    by_turn: dict[int, list[int]] = {}
    for call in with_input:
        by_turn.setdefault(call["turn"], []).append(call["input_tokens"])
    averages = {turn: sum(vals) / len(vals) for turn, vals in by_turn.items()}
    single_turn_only = len(averages) == 1
    turns: list[AuditTurn] = []
    previous: float | None = None
    for turn in sorted(averages):
        avg = averages[turn]
        growth = (avg / previous * 100 - 100) if previous is not None else None
        turns.append(AuditTurn(
            turn=turn, avg_input_tokens=avg, runs=len(by_turn[turn]),
            growth_percent=growth,
        ))
        previous = avg
    most_expensive = max(averages, key=lambda t: averages[t]) if averages else None
    return turns, most_expensive, single_turn_only


def _compute_context_types(calls: list[dict]) -> AuditContextTypes | None:
    with_roles = [c for c in calls if c["roles"] is not None]
    if not with_roles:
        return None
    skipped = len(calls) - len(with_roles)
    totals = dict.fromkeys(CONTEXT_TYPES, 0)
    for run_calls in _group_by_run(calls).values():
        run_chars = _run_type_chars(run_calls)
        if run_chars is None:
            continue
        for type_name, value in run_chars.items():
            totals[type_name] += value
    grand_total = sum(totals.values())
    shares = {
        type_name: (totals[type_name] / grand_total * 100 if grand_total else 0.0)
        for type_name in CONTEXT_TYPES
    }
    # Тренд долей по ходам, усреднённый по прогонам (design D8)
    turn_shares: dict[int, dict[str, list[int]]] = {}
    for run_calls in _group_by_run(calls).values():
        first = min((c["turn"] for c in run_calls), default=None)
        for call in run_calls:
            roles = call["roles"]
            if roles is None:
                continue
            chars = {
                TYPE_SYSTEM: roles.get("system", 0) or 0,
                TYPE_TOOLS: roles.get("tool", 0) or 0,
                TYPE_SESSION_HISTORY: (
                    (roles.get("user", 0) or 0) + (roles.get("assistant", 0) or 0)
                    if call["turn"] == first else 0
                ),
                TYPE_DIALOG_HISTORY: (
                    roles.get("assistant", 0) or 0 if call["turn"] != first else 0
                ),
                TYPE_USER: (
                    roles.get("user", 0) or 0 if call["turn"] != first else 0
                ),
            }
            call_total = sum(chars.values())
            if not call_total:
                continue
            shares_dict = turn_shares.setdefault(call["turn"], {})
            for type_name, value in chars.items():
                shares_dict.setdefault(type_name, []).append(value / call_total * 100)
    trend_by_turn: dict[int, dict[str, float]] = {}
    for turn, shares_dict in turn_shares.items():
        trend_by_turn[turn] = {
            type_name: sum(values) / len(values)
            for type_name, values in shares_dict.items()
        }
    return AuditContextTypes(
        totals=totals, total_chars=grand_total, shares=shares,
        skipped_no_roles=skipped, trend_by_turn=trend_by_turn,
    )


def _compute_repeats(calls: list[dict]) -> AuditRepeats | None:
    by_run = _group_by_run(calls)
    total_input = 0
    repeated_total = 0
    inter_run_tokens = 0
    runs_counted = 0
    for run_calls in by_run.values():
        with_input = [c for c in run_calls if c["input_tokens"] is not None]
        if not with_input:
            continue
        runs_counted += 1
        inputs = [c["input_tokens"] for c in with_input]
        total_input += sum(inputs)
        # Append-only история: повторными считаются входы всех вызовов, кроме последнего.
        repeated_total += sum(inputs[:-1])
        first = run_calls[0]
        if first["roles"] is not None:
            inter_run_tokens += (
                (first["roles"].get("user", 0) or 0)
                + (first["roles"].get("assistant", 0) or 0)
            ) // TOOL_OUTPUT_CHARS_PER_TOKEN
    if not runs_counted:
        return None
    new_total = total_input - repeated_total
    repeated_share = repeated_total / total_input * 100 if total_input else 0.0
    return AuditRepeats(
        total_input=total_input, new_tokens=new_total,
        repeated_tokens=repeated_total, repeated_share_percent=repeated_share,
        inter_run_tokens=inter_run_tokens,
    )


def _compute_cost(
    calls: list[dict],
    price_input_per_m: float,
    price_output_per_m: float,
    cached_price_multiplier: float,
) -> AuditCost | None:
    with_input = [c for c in calls if c["input_tokens"] is not None]
    billed_values = [c["billed_cost"] for c in calls if c["billed_cost"] is not None]
    if not with_input and not billed_values:
        return None
    billed_total = sum(billed_values) if billed_values else None
    billed_calls = len(billed_values)
    if not with_input:
        return AuditCost(
            billed_total=billed_total, billed_calls=billed_calls,
            raw_cost=None, effective_cost=None, cached_share_percent=None,
            cache_savings=None, has_cache_data=False,
            total_input=0, total_output=0, total_cached=0,
            price_input_per_m=price_input_per_m,
            price_output_per_m=price_output_per_m,
            cached_price_per_m=price_input_per_m * cached_price_multiplier,
        )
    total_input = sum(c["input_tokens"] or 0 for c in calls)
    total_output = sum(c["output_tokens"] or 0 for c in calls)
    total_cached = sum(c["cached_tokens"] or 0 for c in calls)
    has_cache_data = any(c["cached_tokens"] is not None for c in calls)
    cached_price = price_input_per_m * cached_price_multiplier
    raw_cost = (
        total_input * price_input_per_m + total_output * price_output_per_m
    ) / 1_000_000
    effective_cost = (
        (total_input - total_cached) * price_input_per_m
        + total_cached * cached_price
        + total_output * price_output_per_m
    ) / 1_000_000
    cached_share = total_cached / total_input * 100 if total_input else 0.0
    cache_savings = raw_cost - effective_cost if has_cache_data else None
    return AuditCost(
        billed_total=billed_total, billed_calls=billed_calls,
        raw_cost=raw_cost, effective_cost=effective_cost,
        cached_share_percent=cached_share if has_cache_data else None,
        cache_savings=cache_savings, has_cache_data=has_cache_data,
        total_input=total_input, total_output=total_output, total_cached=total_cached,
        price_input_per_m=price_input_per_m,
        price_output_per_m=price_output_per_m,
        cached_price_per_m=cached_price,
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
        prompt_roles: dict[str, int] | None = None,
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
                prompt_roles=(
                    json.dumps(prompt_roles, ensure_ascii=False, sort_keys=True)
                    if prompt_roles is not None
                    else None
                ),
                estimated_cost=(
                    estimate_cost(
                        usage, self._price_input_per_m, self._price_output_per_m
                    )
                    if ok
                    else None
                ),
                # Ошибочный вызов не оплачивается: стоимость не фиксируется
                # (спека agent-observability, «Стоимость ошибочного вызова»)
                billed_cost=(
                    usage.get("billed_cost")
                    if ok and usage is not None
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
        prompt_roles = count_prompt_roles(messages)
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
                prompt_roles=prompt_roles,
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
            prompt_roles=prompt_roles,
            created_at=created_at,
        )
        return turn
