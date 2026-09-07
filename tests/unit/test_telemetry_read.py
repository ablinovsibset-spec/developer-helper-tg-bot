"""Unit-тесты read-API TelemetryStore (change add-obs-web-dashboard, task 1.3).

Покрывает: агрегаты с фильтрами, реестр прогонов, timeline, аудит Q1–Q5,
пустую БД, неизвестный run_id, null ≠ 0 для cached/reasoning. Сверка
ключевых цифр аудита с той же фикстурой, что test_obs_audit.py — семантика
read_audit идентична scripts/obs-audit.py (design D5).
"""
from __future__ import annotations

import pytest

from dev_helper_bot.telemetry import (
    RunRecorder,
    TelemetryStore,
)

CHAT_ID = 42


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
                {"cached_tokens": cached_tokens} if cached_tokens is not None else None
            ),
        },
    }
    if billed_cost is not None:
        u["billed_cost"] = billed_cost
        u["raw"]["cost"] = billed_cost
    return u


async def _llm(recorder, turn, input_tokens, roles, ok=True, cached_tokens=None,
               billed_cost=None, output_tokens=4):
    await recorder.record_llm_call(
        turn_number=turn,
        model="m",
        latency_ms=10.0 * turn,
        ok=ok,
        usage=(
            usage(input_tokens, output_tokens=output_tokens,
                  cached_tokens=cached_tokens, billed_cost=billed_cost)
            if ok else None
        ),
        messages_count=turn + 1,
        prompt_chars=100 * turn,
        prompt_roles=roles,
    )


@pytest.fixture
async def empty_store(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    yield s
    await s.close()


@pytest.fixture
async def seeded_store(tmp_path):
    """Та же фикстура, что test_obs_audit.py: три прогона (двухходовый с
    инструментами, одноходовый с крупной историей сессии, легаси без ролей)."""
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    try:
        run_a = RunRecorder(s, 1, "bench")
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

        run_b = RunRecorder(s, 1, "bench")
        await run_b.start()
        await _llm(run_b, 1, 100, {"system": 800, "user": 240,
                                  "assistant": 0, "tool": 0})
        await run_b.finish("success")

        run_c = RunRecorder(s, 2)
        await run_c.start()
        await _llm(run_c, 1, 50, None)
        await run_c.finish("success")
    except Exception:
        await s.close()
        raise
    yield s
    await s.close()


# --- read_summary ---


async def test_summary_empty_db_has_zeros_and_no_cached(empty_store):
    summary = await empty_store.read_summary()

    assert summary.total_runs == 0
    assert summary.input_tokens == 0
    assert summary.cached_tokens is None  # absence ≠ 0
    assert summary.reasoning_tokens is None
    assert summary.reasoning_available is False
    assert summary.estimated_cost == 0.0
    assert summary.status_breakdown == []
    assert summary.cache_hit_rate is None
    assert summary.top_tools == []


async def test_summary_seeded_aggregates_and_status_breakdown(seeded_store):
    summary = await seeded_store.read_summary()

    assert summary.total_runs == 3
    assert summary.finished_runs == 3
    assert summary.input_tokens == 450  # 100+200+100+50
    assert summary.output_tokens == 16
    assert summary.cached_tokens is None
    assert summary.reasoning_tokens is None
    assert summary.reasoning_available is False
    assert summary.llm_calls == 5
    assert summary.tool_calls == 2
    assert summary.status_breakdown == [("success", 3)]
    assert summary.cache_hit_rate is None
    names = [t.name for t in summary.top_tools]
    assert names == ["exec", "search_history"]
    assert summary.top_tools[0].output_tokens == 100  # 400/4
    assert summary.top_tools[0].calls == 1
    assert summary.top_tools[0].share_percent == pytest.approx(83.3, abs=0.1)
    assert summary.top_tools[1].output_tokens == 20  # 80/4
    assert summary.top_tools[1].share_percent == pytest.approx(16.7, abs=0.1)


async def test_summary_label_filter_excludes_other_runs(seeded_store):
    summary = await seeded_store.read_summary(label="bench")

    assert summary.total_runs == 2
    assert summary.input_tokens == 400
    assert summary.tool_calls == 2


async def test_summary_since_future_is_empty(seeded_store):
    assert (await seeded_store.read_summary(since="2100-01-01")).total_runs == 0


async def test_summary_null_cached_preserved_not_zero(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    try:
        run = RunRecorder(s, CHAT_ID)
        await run.start()
        await _llm(run, 1, 100, {"system": 1}, cached_tokens=60)
        await _llm(run, 2, 200, {"system": 1})  # cached=None
        await run.finish("success")

        summary = await s.read_summary()
        assert summary.cached_tokens == 60
        assert summary.cache_hit_rate == pytest.approx(60.0)
    finally:
        await s.close()


async def test_summary_reasoning_available_distinguishes_absent_from_zero(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    try:
        run = RunRecorder(s, CHAT_ID)
        await run.start()
        await _llm(run, 1, 100, {"system": 1})
        await run.finish("success")

        summary = await s.read_summary()
        assert summary.reasoning_tokens is None
        assert summary.reasoning_available is False
    finally:
        await s.close()


# --- read_runs ---


async def test_read_runs_empty_db_returns_empty_list(empty_store):
    assert await empty_store.read_runs() == []


async def test_read_runs_ordered_by_id_with_filters(seeded_store):
    runs = await seeded_store.read_runs()
    assert [r.id for r in runs] == [1, 2, 3]
    assert runs[0].label == "bench"
    assert runs[2].label is None

    bench_only = await seeded_store.read_runs(label="bench")
    assert [r.id for r in bench_only] == [1, 2]

    assert await seeded_store.read_runs(since="2100-01-01") == []


# --- read_run ---


async def test_read_run_unknown_id_returns_none(seeded_store):
    assert await seeded_store.read_run(999) is None


async def test_read_run_unknown_id_empty_db_returns_none(empty_store):
    assert await empty_store.read_run(1) is None


async def test_read_run_timeline_has_llm_and_tool_calls(seeded_store):
    detail = await seeded_store.read_run(1)

    assert detail is not None
    assert detail.run.id == 1
    assert detail.run.label == "bench"
    assert detail.run.status == "success"
    assert [c.turn_number for c in detail.llm_calls] == [1, 2, 3]
    assert detail.llm_calls[0].ok is True
    assert detail.llm_calls[0].input_tokens == 100
    assert detail.llm_calls[0].cached_tokens is None  # absence ≠ 0
    assert detail.llm_calls[2].ok is False
    assert detail.llm_calls[2].input_tokens is None
    assert [c.turn_number for c in detail.tool_calls] == [1, 1]
    assert detail.tool_calls[0].tool_name == "exec"
    assert detail.tool_calls[0].output_tokens == 100  # 400/4
    assert detail.tool_calls[1].tool_name == "search_history"


# --- read_audit: сверка ключевых цифр с obs-audit (design D5) ---


async def test_audit_empty_db_returns_empty_sections(empty_store):
    audit = await empty_store.read_audit()

    assert audit.runs_count == 0
    assert audit.top_tools == []
    assert audit.expensive_turns == []
    assert audit.most_expensive_turn is None
    assert audit.context_types is None
    assert audit.repeats is None
    assert audit.cost is None


async def test_audit_q1_top_tools_shares_match_cli(seeded_store):
    audit = await seeded_store.read_audit()

    # exec: 100 ток. (83.3%), search_history: 20 ток. (16.7%)
    assert audit.top_tools[0].name == "exec"
    assert audit.top_tools[0].output_tokens == 100
    assert audit.top_tools[0].calls == 1
    assert audit.top_tools[0].share_percent == pytest.approx(83.3, abs=0.1)
    assert audit.top_tools[1].name == "search_history"
    assert audit.top_tools[1].output_tokens == 20
    assert audit.top_tools[1].share_percent == pytest.approx(16.7, abs=0.1)


async def test_audit_q2_most_expensive_turn_and_growth_match_cli(seeded_store):
    audit = await seeded_store.read_audit(label="bench")

    turns = {t.turn: t for t in audit.expensive_turns}
    assert turns[1].avg_input_tokens == pytest.approx(100.0)
    assert turns[1].runs == 2
    assert turns[1].growth_percent is None
    assert turns[2].avg_input_tokens == pytest.approx(200.0)
    assert turns[2].runs == 1
    assert turns[2].growth_percent == pytest.approx(100.0)
    assert audit.most_expensive_turn == 2
    assert audit.single_turn_only is False


async def test_audit_q2_single_turn_reports_no_dynamics(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    try:
        run = RunRecorder(s, 1, None)
        await run.start()
        await _llm(run, 1, 10, {"system": 1})
        await run.finish("success")
    finally:
        await s.close()

    s2 = TelemetryStore(tmp_path / "obs.db")
    await s2.open()
    try:
        audit = await s2.read_audit()
    finally:
        await s2.close()

    assert audit.single_turn_only is True
    assert audit.most_expensive_turn == 1


async def test_audit_q3_context_type_shares_match_cli(seeded_store):
    audit = await seeded_store.read_audit()

    ct = audit.context_types
    assert ct is not None
    # Символы: system 3200, сессия 380, диалог 100, user 40, tools 120 (итого 3840)
    assert ct.total_chars == 3840
    assert ct.totals["System prompt"] == 3200
    assert ct.totals["История сессии (ход 1: user+assistant)"] == 380
    assert ct.totals["История диалога (assistant)"] == 100
    assert ct.totals["Текущая задача (user сверх хода 1)"] == 40
    assert ct.totals["Выводы инструментов (вкл. файлы)"] == 120
    assert ct.skipped_no_roles == 1  # легаси-прогон C без ролей
    assert ct.shares["System prompt"] == pytest.approx(83.3, abs=0.1)
    assert ct.shares["История сессии (ход 1: user+assistant)"] == pytest.approx(9.9, abs=0.1)
    assert ct.shares["Выводы инструментов (вкл. файлы)"] == pytest.approx(3.1, abs=0.1)
    assert 1 in ct.trend_by_turn
    assert 2 in ct.trend_by_turn


async def test_audit_q3_unavailable_without_roles(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    try:
        run = RunRecorder(s, 1, None)
        await run.start()
        await _llm(run, 1, 10, None)
        await run.finish("success")
    finally:
        await s.close()

    s2 = TelemetryStore(tmp_path / "obs.db")
    await s2.open()
    try:
        audit = await s2.read_audit()
    finally:
        await s2.close()

    assert audit.context_types is None


async def test_audit_q4_repeats_match_cli(seeded_store):
    audit = await seeded_store.read_audit()

    # Входы: A [100, 200] → повторно 100; B [100] → 0; C [50] → 0.
    # Итого 450, повторно 100 (22%), новая 350. Межпрогонный слой: 95.
    r = audit.repeats
    assert r is not None
    assert r.total_input == 450
    assert r.new_tokens == 350
    assert r.repeated_tokens == 100
    assert r.repeated_share_percent == pytest.approx(22.0, abs=1.0)
    assert r.inter_run_tokens == 95


async def test_audit_q4_no_inputs_returns_none(seeded_store, tmp_path):
    # Мутируем: все input_tokens → NULL
    import sqlite3
    db_path = seeded_store.path
    await seeded_store.close()
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE llm_calls SET input_tokens = NULL")
        conn.commit()
    s2 = TelemetryStore(db_path)
    await s2.open()
    try:
        audit = await s2.read_audit()
    finally:
        await s2.close()

    assert audit.repeats is None


async def test_audit_q5_without_cache_effective_equals_raw(seeded_store):
    audit = await seeded_store.read_audit()

    cost = audit.cost
    assert cost is not None
    assert cost.billed_total is None  # billed_cost не отдаётся
    assert cost.billed_calls == 0
    assert cost.has_cache_data is False
    # raw = (450*0.11 + 16*0.60)/1M
    expected_raw = (450 * 0.11 + 16 * 0.60) / 1_000_000
    assert cost.raw_cost == pytest.approx(expected_raw)
    assert cost.effective_cost == pytest.approx(expected_raw)  # равна сырой
    assert cost.cached_share_percent is None
    assert cost.cache_savings is None


async def test_audit_q5_with_cache_data_shows_both_costs(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    try:
        run = RunRecorder(s, 1, "bench")
        await run.start()
        await _llm(run, 1, 1000, {"system": 100}, cached_tokens=600)
        await run.finish("success")
    finally:
        await s.close()

    s2 = TelemetryStore(tmp_path / "obs.db")
    await s2.open()
    try:
        audit = await s2.read_audit(label="bench")
    finally:
        await s2.close()

    cost = audit.cost
    assert cost is not None
    raw = (1000 * 0.11 + 4 * 0.60) / 1_000_000
    effective = (400 * 0.11 + 600 * 0.0275 + 4 * 0.60) / 1_000_000
    assert cost.raw_cost == pytest.approx(raw)
    assert cost.effective_cost == pytest.approx(effective)
    assert cost.cached_share_percent == pytest.approx(60.0)
    assert cost.cache_savings == pytest.approx(raw - effective)
    assert cost.has_cache_data is True


async def test_audit_q5_billed_present_is_main_value(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    try:
        run = RunRecorder(s, 1, "flex")
        await run.start()
        await _llm(run, 1, 1000, {"system": 100}, billed_cost=1.25, cached_tokens=600)
        await _llm(run, 2, 500, {"system": 100}, billed_cost=0.75)
        await run.finish("success")
    finally:
        await s.close()

    s2 = TelemetryStore(tmp_path / "obs.db")
    await s2.open()
    try:
        audit = await s2.read_audit(label="flex")
    finally:
        await s2.close()

    cost = audit.cost
    assert cost is not None
    assert cost.billed_total == pytest.approx(2.0)
    assert cost.billed_calls == 2
    raw = (1500 * 0.11 + 8 * 0.60) / 1_000_000
    assert cost.raw_cost == pytest.approx(raw)


async def test_audit_q5_multiplier_is_configurable(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    try:
        run = RunRecorder(s, 1, "bench")
        await run.start()
        await _llm(run, 1, 1000, None, cached_tokens=1000)
        await run.finish("success")
    finally:
        await s.close()

    s2 = TelemetryStore(tmp_path / "obs.db")
    await s2.open()
    try:
        audit = await s2.read_audit(label="bench", cached_price_multiplier=0.5)
    finally:
        await s2.close()

    cost = audit.cost
    raw = (1000 * 0.11 + 4 * 0.60) / 1_000_000
    effective = (1000 * 0.055 + 4 * 0.60) / 1_000_000
    assert cost.raw_cost == pytest.approx(raw)
    assert cost.effective_cost == pytest.approx(effective)
    assert cost.cached_price_per_m == pytest.approx(0.055)


async def test_audit_label_filter_excludes_other_runs(seeded_store):
    audit = await seeded_store.read_audit(label="bench")

    # Прогоны A+B: всего входных 100+200+100=400; прогон C (50) не участвует
    assert audit.runs_count == 2
    assert audit.repeats.total_input == 400

