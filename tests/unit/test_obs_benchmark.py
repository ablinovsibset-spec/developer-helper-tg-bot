"""Unit-тесты benchmark-харнесса: метка CLI и автопроверки ответов
(change optimize-token-usage, tasks 5.1–5.2)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dev_helper_bot.memory import MemoryStore
from dev_helper_bot.telemetry import TelemetryStore

from tests.conftest import FakeCommandExecutor

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "obs-benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("obs_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["obs_benchmark"] = module  # нужно для dataclass-аннотаций
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()


def make_harness(executor: FakeCommandExecutor) -> benchmark.Harness:
    return benchmark.Harness(
        llm=object(),
        memory=object(),
        telemetry=object(),
        executor=executor,
    )


# --- Метка прогона (task 5.1) ---


def test_parser_label_defaults_to_benchmark_constant():
    assert benchmark.build_parser().parse_args([]).label == "benchmark"


def test_parser_label_override():
    args = benchmark.build_parser().parse_args(["--label", "optimized"])
    assert args.label == "optimized"


async def test_harness_passes_label_to_recorder(tmp_path):
    """Метка харнесса попадает в телеметрию прогона (спека benchmark-harness)."""
    import sqlite3

    from tests.conftest import assistant_turn, make_scripted_llm

    telemetry = TelemetryStore(tmp_path / "bench.db")
    memory = MemoryStore(tmp_path / "mem.db")
    await telemetry.open()
    await memory.open()
    harness = benchmark.Harness(
        llm=make_scripted_llm([assistant_turn(content="готово")]),
        memory=memory,
        telemetry=telemetry,
        executor=FakeCommandExecutor(),
        skills={},
        label="optimized",
    )
    try:
        await harness.process_message(7, "привет")
    finally:
        await memory.close()
        await telemetry.close()

    with sqlite3.connect(tmp_path / "bench.db") as conn:
        (label,) = conn.execute("SELECT label FROM runs").fetchone()
    assert label == "optimized"


# --- number_in_text ---


def test_number_in_text_matches_plain_and_grouped():
    assert benchmark.number_in_text("сумма 4501500", 4_501_500)
    assert benchmark.number_in_text("сумма 4 501 500", 4_501_500)
    assert benchmark.number_in_text("сумма 4,501,500", 4_501_500)
    assert benchmark.number_in_text("сумма 4\u00a0501\u00a0500", 4_501_500)
    assert not benchmark.number_in_text("сумма 123456", 4_501_500)


# --- Автопроверки сценариев (task 5.2) ---


async def test_check_verbose_sum():
    harness = make_harness(FakeCommandExecutor())
    ok = benchmark.StepCheck(description="", verify=benchmark.check_verbose_sum)
    assert await ok.verify(harness, "Сумма равна 4 501 500.")
    assert not await ok.verify(harness, "Сумма равна 123.")


async def test_check_files_bug_mentions_discount():
    harness = make_harness(FakeCommandExecutor())
    assert await benchmark.check_files_bug(harness, "Баг: скидка не применяется")
    assert await benchmark.check_files_bug(harness, "Функция не учитывает СКИДКУ")
    assert not await benchmark.check_files_bug(harness, "Функция суммирует цены")


async def test_check_multiturn_languages_against_notes_file():
    executor = FakeCommandExecutor()
    executor.files["notes.md"] = "1. Python\n2. JavaScript\n3. Go\n4. Rust"
    harness = make_harness(executor)

    assert await benchmark.check_multiturn_languages(
        harness, "На шаге 1 записали python, JavaScript и GO."
    )
    assert not await benchmark.check_multiturn_languages(
        harness, "Записали python и javascript."  # не все три
    )


async def test_check_multiturn_languages_without_file_fails():
    harness = make_harness(FakeCommandExecutor())  # notes.md нет

    assert not await benchmark.check_multiturn_languages(harness, "что-то")


def test_scenarios_declare_checks():
    by_name = {s.name: s for s in benchmark.SCENARIOS}
    assert dict(by_name["verbose"].checks)[0].verify is benchmark.check_verbose_sum
    assert dict(by_name["files"].checks)[0].verify is benchmark.check_files_bug
    assert dict(by_name["multiturn"].checks)[2].verify is (
        benchmark.check_multiturn_languages
    )
    assert by_name["history"].checks == ()
    # Индексы проверок указывают на существующие шаги
    for scenario in benchmark.SCENARIOS:
        for index, _ in scenario.checks:
            assert 0 <= index < len(scenario.steps)


# --- Маркировка провала отдельно от статуса (task 5.3, unit-часть) ---


async def test_run_scenario_prints_check_verdicts_separately_from_status(tmp_path, capsys):
    """Вывод шага содержит и статус прогона, и отдельную строку проверки."""
    from tests.conftest import assistant_turn, make_scripted_llm

    telemetry = TelemetryStore(tmp_path / "bench.db")
    memory = MemoryStore(tmp_path / "mem.db")
    await telemetry.open()
    await memory.open()
    harness = benchmark.Harness(
        llm=make_scripted_llm([assistant_turn(content="какие-то языки")]),
        memory=memory,
        telemetry=telemetry,
        executor=FakeCommandExecutor(),  # notes.md нет → проверка провалится
        skills={},
        label="test",
    )
    scenario = {s.name: s for s in benchmark.SCENARIOS}["verbose"]
    try:
        statuses, passed, failed = await benchmark.run_scenario(scenario, harness)
    finally:
        await memory.close()
        await telemetry.close()

    out = capsys.readouterr().out
    assert statuses == ["success"]
    assert (passed, failed) == (0, 1)
    assert "прогон завершён: успех" in out
    assert "✗ ПРОВАЛЕНА" in out
