from __future__ import annotations

import logging
import sqlite3

import pytest

from dev_helper_bot.llm import LLMUnavailable
from dev_helper_bot.telemetry import (
    RUN_STATUS_LLM_ERROR,
    RUN_STATUS_SUCCESS,
    ObservingClient,
    RunRecorder,
    TelemetryStore,
    count_prompt_roles,
    estimate_cost,
    estimate_tool_tokens,
)

from tests.conftest import (
    FakeLLM,
    assistant_turn,
)

CHAT_ID = 42
TELEMETRY_LOGGER_NAME = "dev_helper_bot.telemetry"


@pytest.fixture
async def store(tmp_path):
    store = TelemetryStore(tmp_path / "obs.db")
    await store.open()
    yield store
    await store.close()


def telemetry_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == TELEMETRY_LOGGER_NAME and r.levelname == "WARNING"
    ]


def fetch_all(db_path, sql: str, params: tuple = ()) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def make_usage(
    input_tokens: int | None = 100,
    output_tokens: int | None = 40,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "raw": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
    }


# --- Хранилище: схема и вставки (task 2.3) ---


async def test_open_creates_db_file_with_schema(tmp_path):
    db_path = tmp_path / "nested" / "obs.db"

    fresh = TelemetryStore(db_path)
    assert not db_path.exists()
    await fresh.open()
    try:
        assert db_path.exists()
        with sqlite3.connect(db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert {"runs", "llm_calls", "tool_calls"} <= tables
        assert mode.lower() == "wal"
    finally:
        await fresh.close()


async def test_insert_run_and_finish_store_status_and_aggregates(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    recorder = RunRecorder(store, CHAT_ID, label="baseline")
    await recorder.start()
    run_id = recorder.run_id
    assert run_id is not None

    await recorder.record_llm_call(
        turn_number=1,
        model="test-model",
        latency_ms=120.5,
        ok=True,
        usage=make_usage(input_tokens=100, output_tokens=40),
        messages_count=3,
        prompt_chars=500,
    )
    await recorder.record_tool_call(
        turn_number=1,
        tool_name="exec",
        input_size=25,
        output_size=400,
        duration_ms=30.0,
        ok=True,
    )
    await recorder.finish(RUN_STATUS_SUCCESS)
    await store.close()

    run = fetch_all(
        db_path, "SELECT chat_id, label, status, llm_calls, tool_calls, input_tokens, output_tokens FROM runs"
    )
    assert run == [(CHAT_ID, "baseline", RUN_STATUS_SUCCESS, 1, 1, 100, 40)]


async def test_llm_call_row_tolerates_null_usage(store):
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()

    await recorder.record_llm_call(
        turn_number=1,
        model="m",
        latency_ms=5.0,
        ok=True,
        usage=None,
        messages_count=1,
        prompt_chars=10,
    )

    row = fetch_all(
        store.path,
        "SELECT input_tokens, output_tokens, cached_tokens, reasoning_tokens, "
        "estimated_cost, usage_raw FROM llm_calls",
    )[0]
    assert row == (None, None, None, None, None, None)


async def test_failed_llm_call_row_has_no_tokens(store):
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()

    await recorder.record_llm_call(
        turn_number=1,
        model="m",
        latency_ms=7.0,
        ok=False,
        usage=None,
        messages_count=1,
        prompt_chars=10,
    )

    ok, latency = fetch_all(store.path, "SELECT ok, latency_ms FROM llm_calls")[0]
    assert ok == 0
    assert latency == 7.0


async def test_cost_computed_on_write_with_env_like_prices(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    recorder = RunRecorder(
        store,
        CHAT_ID,
        price_input_per_m=0.11,
        price_output_per_m=0.60,
    )
    await recorder.start()

    await recorder.record_llm_call(
        turn_number=1,
        model="m",
        latency_ms=1.0,
        ok=True,
        usage=make_usage(input_tokens=1_000_000, output_tokens=500_000),
        messages_count=1,
        prompt_chars=1,
    )
    await store.close()

    ((cost,),) = fetch_all(db_path, "SELECT estimated_cost FROM llm_calls")
    assert cost == pytest.approx(0.11 + 0.5 * 0.60)


def test_estimate_cost_rules():
    usage = make_usage(input_tokens=200, output_tokens=100, cached_tokens=50)
    assert estimate_cost(usage, 0.10, 0.50) == pytest.approx(
        200 * 0.10 / 1e6 + 100 * 0.50 / 1e6
    )
    # Нет usage — стоимость неизвестна (null), нулевой прайс — честный 0.0
    assert estimate_cost(None, 1.0, 1.0) is None
    assert estimate_cost(usage, 0.0, 0.0) == 0.0
    # input отсутствует, но есть cached — считаем cached по цене входных
    only_cached = make_usage(input_tokens=None, output_tokens=None, cached_tokens=100)
    assert estimate_cost(only_cached, 0.10, 0.50) == pytest.approx(100 * 0.10 / 1e6)
    empty = make_usage(input_tokens=None, output_tokens=None, cached_tokens=None)
    assert estimate_cost(empty, 0.10, 0.50) is None


def test_estimate_tool_tokens_is_chars_over_four():
    assert estimate_tool_tokens(399) == 99
    assert estimate_tool_tokens(0) == 0


# --- Best-effort: сбой записи не возбуждает исключение (task 2.3) ---


async def test_write_failure_on_closed_store_is_swallowed(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=TELEMETRY_LOGGER_NAME)
    store = TelemetryStore(tmp_path / "obs.db")
    await store.open()
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    await store.close()

    # Публичные методы записи не поднимают исключение, а пишут warning
    await recorder.record_llm_call(
        turn_number=1,
        model="m",
        latency_ms=1.0,
        ok=True,
        usage=make_usage(),
        messages_count=1,
        prompt_chars=1,
    )
    await recorder.record_tool_call(
        turn_number=1,
        tool_name="exec",
        input_size=1,
        output_size=1,
        duration_ms=0.0,
        ok=True,
    )
    await recorder.finish(RUN_STATUS_SUCCESS)

    assert telemetry_warnings(caplog)


async def test_fk_violation_is_swallowed_with_warning(store, caplog):
    caplog.set_level(logging.WARNING, logger=TELEMETRY_LOGGER_NAME)
    broken = RunRecorder(store, CHAT_ID)
    await broken.start()
    broken.run_id = 999_999  # несуществующий run

    await broken.record_llm_call(
        turn_number=1,
        model="m",
        latency_ms=1.0,
        ok=True,
        usage=None,
        messages_count=1,
        prompt_chars=1,
    )
    await broken.record_tool_call(
        turn_number=1,
        tool_name="exec",
        input_size=1,
        output_size=4,
        duration_ms=1.0,
        ok=True,
    )

    assert telemetry_warnings(caplog)
    assert fetch_all(store.path, "SELECT COUNT(*) FROM llm_calls") == [(0,)]
    assert fetch_all(store.path, "SELECT COUNT(*) FROM tool_calls") == [(0,)]


async def test_null_run_id_insert_is_swallowed(store, caplog):
    """run не создался (сбой) — последующие записи тоже падают, но не наружу."""
    caplog.set_level(logging.WARNING, logger=TELEMETRY_LOGGER_NAME)
    recorder = RunRecorder(store, CHAT_ID)
    recorder.run_id = None  # имитация провала insert_run

    await recorder.record_tool_call(
        turn_number=1,
        tool_name="exec",
        input_size=1,
        output_size=4,
        duration_ms=1.0,
        ok=True,
    )
    await recorder.finish(RUN_STATUS_SUCCESS)  # no-op, без исключения

    assert telemetry_warnings(caplog)


# --- Recorder и обёртка (tasks 3.1–3.3) ---


def _observing(llm: FakeLLM, recorder: RunRecorder, model: str = "test-model"):
    return ObservingClient(llm, recorder, model=model)


async def test_turn_numbers_ascending_across_calls(store):
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    client = _observing(FakeLLM(), recorder)

    await client.complete([{"role": "user", "content": "раз"}])
    await client.complete([{"role": "user", "content": "два"}])
    await client.complete([{"role": "user", "content": "три"}])

    turns = [row[0] for row in fetch_all(store.path, "SELECT turn_number FROM llm_calls ORDER BY id")]
    assert turns == [1, 2, 3]


async def test_observing_client_records_usage_messages_prompt_chars(store):
    llm = FakeLLM(
        turns=[
            assistant_turn(
                "ok",
                usage={
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cached_tokens": None,
                    "reasoning_tokens": 2,
                    "raw": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            )
        ]
    )
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    client = _observing(llm, recorder)

    await client.complete(
        [
            {"role": "system", "content": "правила"},
            {"role": "user", "content": "привет"},
        ]
    )

    row = fetch_all(
        store.path,
        "SELECT model, ok, input_tokens, output_tokens, reasoning_tokens, "
        "messages_count, prompt_chars, latency_ms, usage_raw FROM llm_calls",
    )[0]
    assert row[0] == "test-model"
    assert row[1] == 1
    assert row[2:5] == (10, 4, 2)
    assert row[5] == 2  # system + user
    assert row[6] == len("правила") + len("привет")
    assert row[7] >= 0.0
    assert '"prompt_tokens"' in row[8]


async def test_failed_llm_call_is_recorded_and_exception_propagates(store):
    llm = FakeLLM(error=LLMUnavailable("connection refused"))
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    client = _observing(llm, recorder)

    with pytest.raises(LLMUnavailable):
        await client.complete([{"role": "user", "content": "привет"}])

    rows = fetch_all(
        store.path, "SELECT ok, input_tokens, latency_ms FROM llm_calls"
    )
    assert rows == [(0, None, rows[0][2])]
    assert rows[0][2] >= 0.0


async def test_repeated_llm_errors_are_all_recorded(store):
    """Повторная ошибка LLM фиксируется — каждая попытка отдельной записью."""
    llm = FakeLLM(error=LLMUnavailable("timeout"))
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    client = _observing(llm, recorder)

    for _ in range(3):
        with pytest.raises(LLMUnavailable):
            await client.complete([{"role": "user", "content": "ещё раз"}])

    rows = fetch_all(store.path, "SELECT turn_number, ok FROM llm_calls ORDER BY id")
    assert rows == [(1, 0), (2, 0), (3, 0)]
    await recorder.finish(RUN_STATUS_LLM_ERROR)
    ((status,),) = fetch_all(store.path, "SELECT status FROM runs")
    assert status == RUN_STATUS_LLM_ERROR


async def test_wrapper_is_transparent_for_caller(store):
    """Обёртка ведёт себя идентично голому клиенту: те же ходы и исключения,
    аргументы пробрасываются как есть."""
    usage = {
        "input_tokens": 5,
        "output_tokens": 2,
        "cached_tokens": None,
        "reasoning_tokens": None,
        "raw": {"prompt_tokens": 5},
    }
    turn = assistant_turn("ответ", usage=usage)
    llm = FakeLLM(turns=[turn])
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    client = _observing(llm, recorder)

    tools = [{"type": "function", "function": {"name": "exec"}}]
    result = await client.complete(
        [{"role": "user", "content": "привет"}], tools=tools, response_format=None
    )

    assert result == turn
    assert llm.requests == [[{"role": "user", "content": "привет"}]]
    assert llm.tools_per_request == [tools]
    assert llm.formats_per_request == [None]


async def test_wrapper_passes_through_unexpected_exception(store):
    class Boom(Exception):
        pass

    llm = FakeLLM(error=Boom("unexpected"))
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    client = _observing(llm, recorder)

    with pytest.raises(Boom):
        await client.complete([{"role": "user", "content": "x"}])


# --- Разбивка промпта по ролям: схема и миграция (tasks 1.1–1.3) ---

# Схема llm_calls до появления prompt_roles (первый релиз add-agent-observability)
_LEGACY_LLM_CALLS_DDL = """
CREATE TABLE llm_calls (
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
"""


def _make_legacy_db(db_path) -> None:
    """База «как до миграции»: старая схема + одна живая запись."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            "CREATE TABLE runs ("
            " id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, label TEXT,"
            " started_at TEXT NOT NULL, finished_at TEXT, status TEXT,"
            " llm_calls INTEGER, tool_calls INTEGER, input_tokens INTEGER,"
            " output_tokens INTEGER, cached_tokens INTEGER,"
            " reasoning_tokens INTEGER, estimated_cost REAL);"
            + _LEGACY_LLM_CALLS_DDL
        )
        conn.execute(
            "INSERT INTO runs (id, chat_id, started_at, status) "
            "VALUES (1, ?, '2026-09-01T00:00:00+00:00', 'success')",
            (CHAT_ID,),
        )
        conn.execute(
            "INSERT INTO llm_calls (run_id, created_at, turn_number, model,"
            " latency_ms, ok, input_tokens, output_tokens, messages_count,"
            " prompt_chars) VALUES (1, '2026-09-01T00:00:01+00:00', 1, 'm',"
            " 10.0, 1, 100, 40, 2, 500)"
        )


def _llm_calls_columns(db_path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(llm_calls)")}


async def test_migration_adds_column_and_keeps_old_records(tmp_path):
    db_path = tmp_path / "obs.db"
    _make_legacy_db(db_path)
    assert "prompt_roles" not in _llm_calls_columns(db_path)

    store = TelemetryStore(db_path)
    await store.open()
    try:
        assert "prompt_roles" in _llm_calls_columns(db_path)
    finally:
        await store.close()

    # Старая запись не тронута: разбивка null (absence ≠ 0), остальное как было
    row = fetch_all(
        db_path,
        "SELECT prompt_roles, input_tokens, output_tokens, prompt_chars, "
        "turn_number FROM llm_calls",
    )[0]
    assert row == (None, 100, 40, 500, 1)

    # Повторное открытие — идемпотентно (без ошибки duplicate column)
    store = TelemetryStore(db_path)
    await store.open()
    await store.close()
    assert "prompt_roles" in _llm_calls_columns(db_path)


async def test_fresh_db_created_with_prompt_roles_column(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    await store.close()

    assert "prompt_roles" in _llm_calls_columns(db_path)


async def test_observing_client_records_prompt_roles_breakdown(store):
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    client = _observing(FakeLLM(), recorder)

    await client.complete(
        [
            {"role": "system", "content": "правила"},
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "exit_code: 0",
            },
        ]
    )

    import json as _json

    raw = fetch_all(store.path, "SELECT prompt_roles FROM llm_calls")[0][0]
    assert _json.loads(raw) == {
        "system": len("правила"),
        "user": len("привет"),
        "assistant": 0,  # content=None считается нулём
        "tool": len("exit_code: 0"),
    }


async def test_prompt_roles_null_when_not_passed(store):
    """Запись без разбивки (прямая запись через recorder) — null, не нули."""
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()

    await recorder.record_llm_call(
        turn_number=1,
        model="m",
        latency_ms=5.0,
        ok=True,
        usage=None,
        messages_count=1,
        prompt_chars=10,
    )

    raw = fetch_all(store.path, "SELECT prompt_roles FROM llm_calls")[0][0]
    assert raw is None


async def test_failed_call_also_carries_prompt_roles(store):
    """Ошибочная попытка тоже видела промпт — разбивка фиксируется."""
    llm = FakeLLM(error=LLMUnavailable("down"))
    recorder = RunRecorder(store, CHAT_ID)
    await recorder.start()
    client = _observing(llm, recorder)

    with pytest.raises(LLMUnavailable):
        await client.complete([{"role": "system", "content": "s"},
                               {"role": "user", "content": "u"}])

    import json as _json

    raw = fetch_all(store.path, "SELECT prompt_roles FROM llm_calls")[0][0]
    assert _json.loads(raw)["system"] == 1
    assert _json.loads(raw)["user"] == 1


def test_count_prompt_roles_rules():
    roles = count_prompt_roles(
        [
            {"role": "system", "content": "ab"},
            {"role": "user", "content": "cde"},
            {"role": "user", "content": None},
            {"role": "tool", "content": "f"},
            {"role": "unknown", "content": "ignored"},
        ]
    )
    # Все четыре ключа всегда, неизвестные роли игнорируются
    assert roles == {"system": 2, "user": 3, "assistant": 0, "tool": 1}
    assert count_prompt_roles([]) == {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool": 0,
    }
