"""Тесты dashboard телеметрии на засеянной и пустой БД (task 5.3)."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

from dev_helper_bot.llm import LLMUnavailable
from dev_helper_bot.telemetry import (
    ObservingClient,
    RunRecorder,
    TelemetryStore,
)

from tests.conftest import FakeLLM, assistant_turn, tool_call

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "obs-dashboard.py"


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("obs_dashboard", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dashboard = _load_dashboard()

CHAT_ID = 42
PRICE_IN = 0.11
PRICE_OUT = 0.60


def usage(input_tokens=10, output_tokens=4, cached=None, reasoning=None) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
        "raw": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
    }


async def _record_llm(recorder, turn, ok=True, usage_dict=None, model="m"):
    await recorder.record_llm_call(
        turn_number=turn,
        model=model,
        latency_ms=100.0 * turn,
        ok=ok,
        usage=usage_dict,
        messages_count=turn + 1,
        prompt_chars=100 * turn,
    )


@pytest.fixture
async def seeded_db(tmp_path):
    """Два прогона: успешный с инструментом и кэшем, и оборванный ошибкой LLM."""
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        # Прогон 1: label=baseline, exec-инструмент, полный usage с кэшем
        run1 = RunRecorder(
            store, CHAT_ID, "baseline",
            price_input_per_m=PRICE_IN, price_output_per_m=PRICE_OUT,
        )
        await run1.start()
        await _record_llm(run1, 1, usage_dict=usage(100, 40, cached=60))
        await run1.record_tool_call(
            turn_number=1, tool_name="exec",
            input_size=25, output_size=800, duration_ms=250.0, ok=True,
        )
        await _record_llm(run1, 2, usage_dict=usage(200, 50))
        await run1.finish("success")

        # Прогон 2: без метки; успешный ход, затем ошибка недоступности LLM
        run2 = RunRecorder(
            store, CHAT_ID + 1,
            price_input_per_m=PRICE_IN, price_output_per_m=PRICE_OUT,
        )
        await run2.start()
        await _record_llm(run2, 1, usage_dict=usage(10, 5))
        await _record_llm(run2, 2, ok=False)
        await run2.finish("llm_error")
    finally:
        await store.close()
    return db_path


@pytest.fixture
async def empty_db(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    await store.close()
    return db_path


def open_conn(db_path):
    conn = sqlite3.connect(db_path)
    return conn


# --- Агрегаты (task 5.3) ---


async def test_aggregates_on_seeded_db(seeded_db):
    conn = open_conn(seeded_db)
    try:
        out = dashboard.render_aggregates(conn, None, None)
    finally:
        conn.close()

    assert "Прогонов: 2 (завершено 2)" in out
    assert "успех: 1" in out
    assert "ошибка LLM: 1" in out
    assert f"Входные       {dashboard.human_count(310)}" in out
    assert f"Выходные      {dashboard.human_count(95)}" in out
    assert f"Кэшированные  {dashboard.human_count(60)}" in out
    assert "Reasoning     недоступно" in out  # поставщик не отдавал
    assert "$0.000091" in out  # (310*0.11 + 95*0.60)/1M
    assert "Токены        202.5" in out  # (310+95)/2
    assert "LLM-ходы      2" in out  # (2 успешных хода + успех и ошибка)/2
    assert "Tool-вызовы   0.5" in out  # 1/2
    assert "Cache hit rate: 60%" in out  # 60 кэш / 100 input (где кэш известен)
    assert "exec" in out
    assert "200 ток." in out  # exec: 800 симв. / 4


async def test_aggregates_label_filter(seeded_db):
    conn = open_conn(seeded_db)
    try:
        out = dashboard.render_aggregates(conn, "baseline", None)
    finally:
        conn.close()

    assert "Прогонов: 1 (завершено 1)" in out
    assert f"Входные       {dashboard.human_count(300)}" in out
    assert f"Выходные      {dashboard.human_count(90)}" in out


async def test_aggregates_since_future_is_empty(seeded_db):
    conn = open_conn(seeded_db)
    try:
        out = dashboard.render_aggregates(conn, None, "2100-01-01")
    finally:
        conn.close()

    assert "нет прогонов" in out
    assert "--since" in out  # подсказка про фильтры


async def test_aggregates_empty_db(empty_db):
    conn = open_conn(empty_db)
    try:
        out = dashboard.render_aggregates(conn, None, None)
    finally:
        conn.close()

    assert out == "В базе телеметрии нет прогонов."


async def test_aggregates_without_cached_data_reports_na(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        run = RunRecorder(store, CHAT_ID, price_input_per_m=0.0, price_output_per_m=0.0)
        await run.start()
        await _record_llm(run, 1, usage_dict=usage(30, 8))
        await run.finish("success")
    finally:
        await store.close()

    conn = open_conn(db_path)
    try:
        out = dashboard.render_aggregates(conn, None, None)
    finally:
        conn.close()

    assert "Кэшированные  недоступно" in out
    assert "Cache hit rate: недоступно" in out
    assert "Стоимость (виртуальная): $0.00" in out  # нулевой прайс — честный 0


# --- Timeline (task 5.3) ---


async def test_timeline_of_seeded_run(seeded_db):
    conn = open_conn(seeded_db)
    try:
        out = dashboard.render_timeline(conn, 1)
    finally:
        conn.close()

    assert "Прогон #1" in out
    assert "чат 42" in out
    assert "метка 'baseline'" in out
    assert "успех" in out
    assert "Turn 1" in out
    assert "Turn 2" in out
    # Ход LLM: токены и длительность (числа выровнены по правому краю)
    assert f"in {dashboard.human_count(100):>8}" in out
    assert f"out {dashboard.human_count(40):>8}" in out
    assert "кэш 60" in out
    assert "100мс" in out
    assert "200мс" in out
    # Инструмент на ходу 1 с оценкой токенов и исходом
    assert "exec" in out
    assert "~200 ток." in out
    assert "успех" in out


async def test_timeline_of_llm_error_run_shows_failed_attempt(seeded_db):
    conn = open_conn(seeded_db)
    try:
        out = dashboard.render_timeline(conn, 2)
    finally:
        conn.close()

    assert "ошибка LLM" in out
    assert "ошибка недоступности" in out


async def test_timeline_missing_run_is_friendly(seeded_db):
    conn = open_conn(seeded_db)
    try:
        out = dashboard.render_timeline(conn, 999)
    finally:
        conn.close()

    assert "не найден" in out


# --- CLI main: коды возврата и сообщения (task 5.2) ---


async def test_main_missing_db_file_is_friendly(tmp_path, capsys):
    code = dashboard.main(["--db", str(tmp_path / "nope.db")])

    assert code == 1
    assert "не найдена" in capsys.readouterr().out


async def test_main_empty_db_returns_message(tmp_path, empty_db, capsys):
    code = dashboard.main(["--db", str(empty_db)])

    assert code == 0
    assert "нет прогонов" in capsys.readouterr().out


async def test_main_run_flag_prints_timeline(seeded_db, capsys):
    code = dashboard.main(["--db", str(seeded_db), "--run", "1"])

    assert code == 0
    out = capsys.readouterr().out
    assert "Прогон #1" in out
    assert "Turn 1" in out


async def test_main_db_defaults_to_env(monkeypatch, seeded_db, capsys):
    monkeypatch.setenv("OBS_DB_PATH", str(seeded_db))

    code = dashboard.main([])

    assert code == 0
    assert "Прогонов: 2" in capsys.readouterr().out


# Смоук: обёртка + recorder + dashboard вместе (герметичный мини-прогон)


async def test_end_to_end_wrapper_records_readable_by_dashboard(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        recorder = RunRecorder(
            store, CHAT_ID, "smoke",
            price_input_per_m=PRICE_IN, price_output_per_m=PRICE_OUT,
        )
        await recorder.start()
        client = ObservingClient(
            FakeLLM([assistant_turn("ok", usage=usage(500, 120))]), recorder, "t"
        )
        await client.complete([{"role": "user", "content": "привет"}])
        await recorder.finish("success")
    finally:
        await store.close()

    conn = open_conn(db_path)
    try:
        out = dashboard.render_aggregates(conn, "smoke", None)
    finally:
        conn.close()

    assert "Прогонов: 1" in out
    assert f"Входные       {dashboard.human_count(500)}" in out


def test_human_count_and_duration_formats():
    assert dashboard.human_count(8_200) == "8,200"
    assert dashboard.human_count(39_400) == "39.4k"
    assert dashboard.human_count(4_200_000) == "4.2M"
    assert dashboard.human_count(None) == "недоступно"
    assert dashboard.human_duration(250.0) == "250мс"
    assert dashboard.human_duration(1500.0) == "1.5с"
