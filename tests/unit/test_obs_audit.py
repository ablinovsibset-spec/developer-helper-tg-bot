"""Смоук-тесты аудита потребления токенов на временной БД с синтетическими
данными (task 3.6): все четыре секции, фильтры, толерантность к null."""
from __future__ import annotations

import importlib.util
import re
import sqlite3
from pathlib import Path

import pytest

from dev_helper_bot.telemetry import RunRecorder, TelemetryStore

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "obs-audit.py"


def _load_audit():
    spec = importlib.util.spec_from_file_location("obs_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def usage(input_tokens, output_tokens=4, cached_tokens=None, billed_cost=None) -> dict:
    u = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": None,
        "raw": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "prompt_tokens_details": (
                {"cached_tokens": cached_tokens}
                if cached_tokens is not None
                else None
            ),
        },
    }
    if billed_cost is not None:
        u["billed_cost"] = billed_cost
        u["raw"]["cost"] = billed_cost
    return u


async def _llm(recorder, turn, input_tokens, roles, ok=True, cached_tokens=None,
               billed_cost=None):
    await recorder.record_llm_call(
        turn_number=turn,
        model="m",
        latency_ms=10.0 * turn,
        ok=ok,
        usage=(
            usage(input_tokens, cached_tokens=cached_tokens,
                  billed_cost=billed_cost)
            if ok else None
        ),
        messages_count=turn + 1,
        prompt_chars=100 * turn,
        prompt_roles=roles,
    )


@pytest.fixture
async def seeded_db(tmp_path):
    """Три прогона: дваходовый с инструментами и фидбеком, одноходовый
    с крупной историей сессии, легаси-прогон без разбивки ролей."""
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        # Прогон A (label=bench, чат 1): ход 1 с историей сессии, ход 2 с
        # tool-выводом и фидбеком, ход 3 — ошибочная попытка без токенов
        run_a = RunRecorder(store, 1, "bench")
        await run_a.start()
        await _llm(run_a, 1, 100, {"system": 800, "user": 100,
                                   "assistant": 40, "tool": 0})
        await run_a.record_tool_call(
            turn_number=1, tool_name="exec",
            input_size=25, output_size=400, duration_ms=100.0, ok=True,
        )
        await run_a.record_tool_call(
            turn_number=1, tool_name="search_history",
            input_size=15, output_size=80, duration_ms=50.0, ok=True,
        )
        await _llm(run_a, 2, 200, {"system": 800, "user": 20,
                                   "assistant": 50, "tool": 60})
        await _llm(run_a, 3, None, {"system": 800, "user": 20,
                                    "assistant": 50, "tool": 60}, ok=False)
        await run_a.finish("success")

        # Прогон B (label=bench, чат 1): одноходовый, ход 1 почти весь —
        # загруженная история сессии
        run_b = RunRecorder(store, 1, "bench")
        await run_b.start()
        await _llm(run_b, 1, 100, {"system": 800, "user": 240,
                                   "assistant": 0, "tool": 0})
        await run_b.finish("success")

        # Прогон C (без метки, чат 2): легаси-запись без разбивки
        run_c = RunRecorder(store, 2)
        await run_c.start()
        await _llm(run_c, 1, 50, None)
        await run_c.finish("success")
    finally:
        await store.close()
    return db_path


def render(db_path, label=None, since=None) -> str:
    conn = sqlite3.connect(db_path)
    try:
        return audit.render_audit(conn, label, since)
    finally:
        conn.close()


# --- Q1: топ инструментов ---


async def test_q1_shares_and_bars(seeded_db):
    out = render(seeded_db)
    q1 = out.split("Q2.")[0]

    # exec: 400/4=100 ток., search_history: 80/4=20 ток. → 83%/17%
    assert "exec" in q1
    assert "search_history" in q1
    assert "83.3%" in q1
    assert "16.7%" in q1
    assert "█" in q1
    assert "120" in q1  # всего оценочных токенов вывода


async def test_q1_no_tool_calls(seeded_db):
    db_path = seeded_db
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM tool_calls")
        conn.commit()
        out = audit.render_audit(conn, "bench", None)
    finally:
        conn.close()

    assert "Нет вызовов инструментов" in out


# --- Q2: самый дорогой ход ---


async def test_q2_most_expensive_turn_and_growth(seeded_db):
    out = render(seeded_db, label="bench")
    q2 = out.split("Q2.")[1].split("Q3.")[0]

    # ход 1: (100+100)/2=100; ход 2: 200 → самый дорогой — ход 2 (+100%)
    assert "Ход 1" in q2
    assert "Ход 2" in q2
    assert "Самый дорогой ход: 2" in q2
    assert "рост к предыдущему +100%" in q2


async def test_q2_single_turn_runs_report_no_dynamics(seeded_db):
    db_path = seeded_db
    conn = sqlite3.connect(db_path)
    try:
        # Оставляем только одноходовые прогоны B и C
        conn.execute("DELETE FROM llm_calls WHERE run_id = 1")
        conn.execute("UPDATE runs SET status = 'success'")
        conn.commit()
        out = audit.render_audit(conn, None, None)
    finally:
        conn.close()

    q2 = out.split("Q2.")[1].split("Q3.")[0]
    assert "по одному LLM-ходу — динамики роста нет" in q2


# --- Q3: рост типов контекста ---


async def test_q3_type_shares(seeded_db):
    out = render(seeded_db)
    q3 = out.split("Q3.")[1].split("Q4.")[0]

    # Символы: system 3200, сессия 380, диалог 100, user 40, tools 120
    # (итого 3840): легаси-прогон C без ролей не подставляет нули
    assert "(без разбивки, не учтено: 1 вызов.)" in q3
    assert "83.3%" in q3  # system: 3200/3840
    assert "9.9%" in q3  # история сессии: 380/3840
    assert "2.6%" in q3  # история диалога: 100/3840
    assert "1.0%" in q3  # текущая задача: 40/3840
    assert "3.1%" in q3  # выводы инструментов: 120/3840


async def test_q3_trend_by_turn(seeded_db):
    out = render(seeded_db, label="bench")
    q3 = out.split("Q3.")[1].split("Q4.")[0]

    assert "Тренд долей по ходам" in q3
    assert "ход 1:" in q3
    assert "ход 2:" in q3


async def test_q3_unavailable_without_roles(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        run = RunRecorder(store, 1, None)
        await run.start()
        await _llm(run, 1, 10, None)  # легаси-запись без разбивки
        await run.finish("success")
    finally:
        await store.close()

    out = render(db_path)
    q3 = out.split("Q3.")[1].split("Q4.")[0]
    assert "Недоступно" in q3
    assert "0%" not in q3  # нули не подставляются


# --- Q4: повторные токены ---


async def test_q4_repeated_new_and_inter_run_layer(seeded_db):
    out = render(seeded_db)
    q4 = out.split("Q4.")[1]

    # Входы: A [100, 200] → повторно 100; B [100] → 0; C [50] → 0.
    # Итого 450, повторно 100 (22%), новая 350.
    assert re.search(r"Всего входных:\s+450", q4)
    assert re.search(r"Новая информация:\s+350", q4)
    assert re.search(r"Повторная информация:\s+100", q4)
    assert "(22% входных — модель уже видела)" in q4
    # Межпрогонный слой: A (100+40)//4=35, B (240+0)//4=60, C без ролей → 95
    assert "~95" in q4
    assert "оценка сверху" in q4


async def test_q4_no_inputs_reports_no_data(seeded_db):
    db_path = seeded_db
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE llm_calls SET input_tokens = NULL")
        conn.commit()
        out = audit.render_audit(conn, None, None)
    finally:
        conn.close()

    q4 = out.split("Q4.")[1].split("Q5.")[0]
    assert "Нет прогонов с входными токенами" in q4


# --- Q5: стоимость с кэш-скидкой (task 6.2) ---


def q5(out: str) -> str:
    return out.split("Q5.")[1]


async def test_q5_without_cache_data_effective_equals_raw_with_marker(seeded_db):
    out = render(seeded_db)

    section = q5(out)
    raw = re.search(r"Сырая стоимость:\s+\$([\d.]+)", section)
    effective = re.search(r"Эффективная стоимость:\s+\$([\d.]+)", section)
    assert raw and effective
    assert raw.group(1) == effective.group(1)  # равна сырой
    assert "данных о кэше нет" in section
    assert "Экономия кэша" not in section


async def test_q5_with_cache_data_shows_both_costs(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        run = RunRecorder(store, 1, "bench")
        await run.start()
        await _llm(run, 1, 1000, {"system": 100}, cached_tokens=600)
        await run.finish("success")
    finally:
        await store.close()

    out = render(db_path, label="bench")

    section = q5(out)
    # raw = 1000×0.11 + 4×0.60 = $0.0001124/1M-масштаб… считаем точно:
    raw = (1000 * 0.11 + 4 * 0.60) / 1_000_000
    effective = (400 * 0.11 + 600 * 0.0275 + 4 * 0.60) / 1_000_000
    assert f"${raw:.6f}" in section
    assert f"${effective:.6f}" in section
    assert "кэшировано 60.0% входных" in section
    assert f"Экономия кэша:         ${raw - effective:.6f}" in section


async def test_q5_multiplier_is_configurable(tmp_path):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        run = RunRecorder(store, 1, "bench")
        await run.start()
        await _llm(run, 1, 1000, None, cached_tokens=1000)
        await run.finish("success")
    finally:
        await store.close()

    conn = sqlite3.connect(db_path)
    try:
        out = audit.render_audit(
            conn, "bench", None, cached_price_multiplier=0.5
        )
    finally:
        conn.close()

    section = q5(out)
    # Полностью кэшированный ввод при 0.5×: эффективная вдвое дешевле сырой
    raw = (1000 * 0.11 + 4 * 0.60) / 1_000_000
    effective = (1000 * 0.055 + 4 * 0.60) / 1_000_000
    assert f"${effective:.6f}" in section
    assert f"${raw:.6f}" in section
    assert "0.5× входного" in section


async def test_q5_no_inputs_reports_no_data(seeded_db):
    db_path = seeded_db
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE llm_calls SET input_tokens = NULL")
        conn.commit()
        out = audit.render_audit(conn, None, None)
    finally:
        conn.close()

    assert "Нет LLM-вызовов с входными токенами" in q5(out)


# --- Q5: фактический биллинг поставщика (pin-routerai-flex, task 3.2) ---
# Три сценараря спеки token-audit: billed есть; billed нет, кеш есть;
# ни того ни другого.


async def test_q5_billed_present_actual_cost_is_main_line(tmp_path):
    """Сценарий «Фактическая стоимость доступна»: сумма billed_cost (₽) —
    основной строкой, оценки по прайсу — следом с пометкой «оценка»."""
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        run = RunRecorder(store, 1, "flex")
        await run.start()
        await _llm(run, 1, 1000, {"system": 100}, billed_cost=1.25,
                   cached_tokens=600)
        await _llm(run, 2, 500, {"system": 100}, billed_cost=0.75)
        await run.finish("success")
    finally:
        await store.close()

    out = render(db_path, label="flex")

    section = q5(out)
    # Основная строка — фактическая сумма (2.00 ₽), до строк оценок
    assert "Фактический биллинг поставщика: 2.0000 ₽" in section
    assert "сумма usage.cost по 2 вызовам" in section
    # Оценки — следом, с явной пометкой
    raw = (1500 * 0.11 + 8 * 0.60) / 1_000_000
    assert section.index("Фактический биллинг") < section.index("Сырая оценка")
    assert re.search(rf"Сырая оценка:\s+\${raw:.6f}", section)
    assert "оценка: входные $0.11/1M" in section
    assert "Эффективная оценка:" in section
    # Прежние подписи основных строк больше не основа
    assert "Сырая стоимость:" not in section


async def test_q5_billed_absent_cache_present_keeps_estimate_behavior(tmp_path):
    """Сценарий «Есть данные о кэше»: сырая и эффективная стоимости с долей
    кэшированных токенов (LM Studio: billed_cost нет)."""
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        run = RunRecorder(store, 1, "bench")
        await run.start()
        await _llm(run, 1, 1000, {"system": 100}, cached_tokens=600)
        await run.finish("success")
    finally:
        await store.close()

    out = render(db_path, label="bench")

    section = q5(out)
    assert "Фактический биллинг" not in section
    assert "Сырая стоимость:" in section
    assert "кэшировано 60.0% входных" in section
    assert "Экономия кэша" in section


async def test_q5_neither_billed_nor_cache_effective_equals_raw(seeded_db):
    """Сценарий «Данных о кэше нет»: эффективная равна сырой с пометкой."""
    out = render(seeded_db)

    section = q5(out)
    assert "Фактический биллинг" not in section
    assert "данных о кэше нет" in section
    assert "Экономия кэша" not in section


# --- Фильтры и CLI-поведение ---


async def test_label_filter_excludes_other_runs(seeded_db):
    out = render(seeded_db, label="bench")

    # Прогоны A+B: всего входных 100+200+100=400; прогон C (50) не участвует
    assert "прогонов 2" in out
    q4 = out.split("Q4.")[1]
    assert re.search(r"Всего входных:\s+400", q4)
    assert not re.search(r"Всего входных:\s+450", q4)


async def test_since_filter_future_is_empty(seeded_db):
    out = render(seeded_db, since="2100-01-01")

    assert "нет прогонов" in out
    assert "--label/--since" in out


async def test_main_missing_db_file_is_friendly(tmp_path, capsys):
    code = audit.main(["--db", str(tmp_path / "nope.db")])

    assert code == 1
    assert "не найдена" in capsys.readouterr().out


async def test_main_empty_db_returns_message(tmp_path, capsys):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    await store.close()

    code = audit.main(["--db", str(db_path)])

    assert code == 0
    assert "нет прогонов" in capsys.readouterr().out


async def test_main_prints_all_sections(seeded_db, capsys):
    code = audit.main(["--db", str(seeded_db), "--label", "bench"])

    out = capsys.readouterr().out
    assert code == 0
    for section in ("Q1.", "Q2.", "Q3.", "Q4.", "Q5."):
        assert section in out


async def test_main_cached_price_multiplier_flag(tmp_path, capsys):
    db_path = tmp_path / "obs.db"
    store = TelemetryStore(db_path)
    await store.open()
    try:
        run = RunRecorder(store, 1, "bench")
        await run.start()
        await _llm(run, 1, 100, None, cached_tokens=50)
        await run.finish("success")
    finally:
        await store.close()

    code = audit.main(["--db", str(db_path), "--label", "bench",
                       "--cached-price-multiplier", "0.1"])

    out = capsys.readouterr().out
    assert code == 0
    assert "0.1× входного" in q5(out)


def test_bar_and_human_count():
    assert audit.bar(50.0) == "█" * 12
    assert audit.bar(100.0) == "█" * 24
    assert audit.bar(0.0) == ""
    assert audit.human_count(None) == "недоступно"
