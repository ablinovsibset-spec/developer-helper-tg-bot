"""Тесты HTTP-хендлеров дашборда (change add-obs-web-dashboard, task 2.4).

aiohttp test utilities: сводка, реестр, timeline, аудит, заглушка без store,
фильтры label/since, null ≠ 0 в JSON. Сверка ключевых цифр с той же фикстурой,
что test_obs_audit.py / test_telemetry_read.py.
"""
from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from dev_helper_bot.obs_web import build_app, start_app, stop_app
from dev_helper_bot.telemetry import RunRecorder, TelemetryStore

CHAT_ID = 42


def usage(input_tokens, output_tokens=4, cached_tokens=None, billed_cost=None) -> dict:
    u = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": None,
        "raw": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
    }
    if billed_cost is not None:
        u["billed_cost"] = billed_cost
        u["raw"]["cost"] = billed_cost
    return u


async def _llm(recorder, turn, input_tokens, roles, ok=True, cached_tokens=None,
               billed_cost=None):
    await recorder.record_llm_call(
        turn_number=turn, model="m", latency_ms=10.0 * turn, ok=ok,
        usage=(usage(input_tokens, cached_tokens=cached_tokens,
                     billed_cost=billed_cost) if ok else None),
        messages_count=turn + 1, prompt_chars=100 * turn, prompt_roles=roles,
    )


@pytest.fixture
async def seeded_store(tmp_path):
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


@pytest.fixture
async def empty_store(tmp_path):
    s = TelemetryStore(tmp_path / "obs.db")
    await s.open()
    yield s
    await s.close()


def _client(store):
    return TestClient(TestServer(build_app(store)))


# --- Заглушка без store (task 2.3) ---


async def test_no_store_html_pages_show_stub():
    client = TestClient(TestServer(build_app(None)))
    await client.start_server()
    try:
        for path in ("/", "/runs", "/runs/1", "/audit"):
            r = await client.get(path)
            assert r.status == 200
            text = await r.text()
            assert "Телеметрия недоступна" in text
    finally:
        await client.close()


async def test_no_store_json_api_reports_unavailable():
    client = TestClient(TestServer(build_app(None)))
    await client.start_server()
    try:
        for path in ("/api/summary", "/api/runs", "/api/runs/1", "/api/audit"):
            r = await client.get(path)
            assert r.status == 200
            j = await r.json()
            assert j["available"] is False
    finally:
        await client.close()


# --- Сводка ---


async def test_summary_page_renders_when_store_present(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/")
        assert r.status == 200
        text = await r.text()
        assert "Сводка телеметрии" in text
        assert "renderSummary" in text  # JS poll на месте
    finally:
        await client.close()


async def test_api_summary_seeded_aggregates(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/summary")
        j = await r.json()
        assert j["available"] is True
        s = j["summary"]
        assert s["total_runs"] == 3
        assert s["input_tokens"] == 450
        assert s["cached_tokens"] is None  # absence ≠ 0
        assert s["reasoning_tokens"] is None
        assert s["reasoning_available"] is False
        assert s["top_tools"][0]["name"] == "exec"
        assert s["top_tools"][0]["output_tokens"] == 100
    finally:
        await client.close()


async def test_api_summary_label_filter(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/summary?label=bench")
        s = (await r.json())["summary"]
        assert s["total_runs"] == 2
        assert s["input_tokens"] == 400
    finally:
        await client.close()


async def test_api_summary_empty_db(empty_store):
    client = _client(empty_store)
    await client.start_server()
    try:
        r = await client.get("/api/summary")
        s = (await r.json())["summary"]
        assert s["total_runs"] == 0
        assert s["cached_tokens"] is None
    finally:
        await client.close()


# --- Реестр прогонов ---


async def test_api_runs_list_ordered(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/runs")
        j = await r.json()
        assert [run["id"] for run in j["runs"]] == [1, 2, 3]
        assert j["runs"][0]["label"] == "bench"
        assert j["runs"][2]["label"] is None
    finally:
        await client.close()


async def test_api_runs_label_filter(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/runs?label=bench")
        j = await r.json()
        assert [run["id"] for run in j["runs"]] == [1, 2]
    finally:
        await client.close()


async def test_runs_page_html_has_links(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/runs")
        text = await r.text()
        assert "renderRuns" in text
        assert "/runs/" in text  # шаблон ссылок
    finally:
        await client.close()


# --- Timeline прогона ---


async def test_api_run_timeline(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/runs/1")
        j = await r.json()
        assert j["available"] is True
        run = j["run"]
        assert run["run"]["id"] == 1
        assert run["run"]["label"] == "bench"
        assert [c["turn_number"] for c in run["llm_calls"]] == [1, 2]
        assert run["llm_calls"][0]["input_tokens"] == 100
        assert run["llm_calls"][0]["cached_tokens"] is None  # absence ≠ 0
        assert [c["turn_number"] for c in run["tool_calls"]] == [1, 1]
        assert run["tool_calls"][0]["tool_name"] == "exec"
    finally:
        await client.close()


async def test_api_run_unknown_id_returns_null(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/runs/999")
        j = await r.json()
        assert j["available"] is True
        assert j["run"] is None
    finally:
        await client.close()


async def test_run_page_html(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/runs/1")
        text = await r.text()
        assert "renderRun" in text
        assert "Прогон #1" in text
    finally:
        await client.close()


# --- Аудит ---


async def test_api_audit_q1_top_tools(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/audit")
        a = (await r.json())["audit"]
        assert a["runs_count"] == 3
        assert a["top_tools"][0]["name"] == "exec"
        assert a["top_tools"][0]["share_percent"] == pytest.approx(83.3, abs=0.1)
    finally:
        await client.close()


async def test_api_audit_q2_most_expensive_turn(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/audit?label=bench")
        a = (await r.json())["audit"]
        turns = {t["turn"]: t for t in a["expensive_turns"]}
        assert turns[2]["avg_input_tokens"] == pytest.approx(200.0)
        assert a["most_expensive_turn"] == 2
    finally:
        await client.close()


async def test_api_audit_q4_repeats(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/audit")
        rep = (await r.json())["audit"]["repeats"]
        assert rep["total_input"] == 450
        assert rep["repeated_tokens"] == 100
        assert rep["inter_run_tokens"] == 95
    finally:
        await client.close()


async def test_api_audit_q5_without_cache(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/api/audit")
        cost = (await r.json())["audit"]["cost"]
        assert cost["billed_total"] is None
        assert cost["has_cache_data"] is False
        assert cost["raw_cost"] == pytest.approx(cost["effective_cost"])
    finally:
        await client.close()


async def test_audit_page_html(seeded_store):
    client = _client(seeded_store)
    await client.start_server()
    try:
        r = await client.get("/audit")
        text = await r.text()
        assert "renderAudit" in text
        assert "Аудит" in text
    finally:
        await client.close()


# --- Lifecycle: start_app / stop_app (design D1, task 3.2) ---


async def test_start_app_binds_and_stop_cleans_up(seeded_store, unused_tcp_port):
    app = build_app(seeded_store)
    runner = await start_app(app, port=unused_tcp_port)
    try:
        # Дашборд доступен по loopback
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{unused_tcp_port}/api/summary") as r:
                assert r.status == 200
                j = await r.json()
                assert j["available"] is True
    finally:
        await stop_app(runner)


async def test_start_app_busy_port_raises(seeded_store, unused_tcp_port):
    app1 = build_app(seeded_store)
    runner1 = await start_app(app1, port=unused_tcp_port)
    try:
        app2 = build_app(seeded_store)
        with pytest.raises(OSError, match="Не удалось занять порт"):
            await start_app(app2, port=unused_tcp_port)
    finally:
        await stop_app(runner1)
