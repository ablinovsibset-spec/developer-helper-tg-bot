"""Инструментация run_agent и проводка телеметрии: статусы прогонов
и неизменность поведения агента (task 4.3, спека agent-observability)."""
from __future__ import annotations

import sqlite3

import pytest

from dev_helper_bot.agent import (
    JSON_RETRIES_EXHAUSTED_MESSAGE,
    RESPONSE_TRUNCATED_MESSAGE,
    STEPS_EXHAUSTED_MESSAGE,
    run_agent,
)
from dev_helper_bot.llm import LLMUnavailable
from dev_helper_bot.main import handle_text
from dev_helper_bot.memory import MemoryStore
from dev_helper_bot.telemetry import (
    RUN_STATUS_LLM_ERROR,
    RUN_STATUS_STEPS_EXHAUSTED,
    RUN_STATUS_SUCCESS,
    RUN_STATUS_VALIDATION_ERROR,
    ObservingClient,
    RunRecorder,
    TelemetryStore,
)
from dev_helper_bot.tools import EXEC_TOOL_SPEC, LIST_TOOL_SPEC, SEARCH_TOOL_SPEC

from tests.conftest import (
    FakeBot,
    FakeCommandExecutor,
    FakeLLM,
    FakeMessage,
    assistant_turn,
    make_scripted_llm,
    tool_call,
)

TOOLS = [EXEC_TOOL_SPEC, SEARCH_TOOL_SPEC, LIST_TOOL_SPEC]
CHAT_ID = 42


@pytest.fixture
async def obs(tmp_path):
    store = TelemetryStore(tmp_path / "obs.db")
    await store.open()
    yield store
    await store.close()


@pytest.fixture
async def memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.open()
    yield store
    await store.close()


def rows(db_path, sql: str, params: tuple = ()) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def run_row(store: TelemetryStore) -> tuple:
    (row,) = rows(store.path, "SELECT status, llm_calls, tool_calls FROM runs")
    return row


def new_recorder(store: TelemetryStore, **kwargs) -> RunRecorder:
    return RunRecorder(store, CHAT_ID, **kwargs)


async def run_with_telemetry(store: TelemetryStore, llm: FakeLLM):
    """Герметичный прогон: recorder + наблюдающая обёртка над фейковым LLM."""
    recorder = new_recorder(store)
    await recorder.start()
    client = ObservingClient(llm, recorder, model="fake-model")
    reply = await run_agent(
        client,
        [{"role": "user", "content": "запрос"}],
        tools=TOOLS,
        executor=FakeCommandExecutor(),
        recorder=recorder,
    )
    return reply, recorder


def usage(input_tokens=10, output_tokens=4, cached=None, reasoning=None) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
        "raw": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
    }


def echo_turn(command: str) -> dict:
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "%s"}' % command)],
        finish_reason="tool_calls",
    )


def broken_json_turn(finish_reason: str = "tool_calls") -> dict:
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "echo')],
        finish_reason=finish_reason,
    )


# --- Статусы прогона для терминальных возвратов run_agent ---


async def test_final_reply_finalizes_run_as_success(obs):
    llm = make_scripted_llm(
        [
            assistant_turn("готово", usage=usage(input_tokens=100, output_tokens=20)),
        ]
    )

    reply, _ = await run_with_telemetry(obs, llm)

    assert reply == "готово"
    assert run_row(obs) == (RUN_STATUS_SUCCESS, 1, 0)
    ((input_tokens, output_tokens),) = rows(
        obs.path, "SELECT input_tokens, output_tokens FROM runs"
    )
    assert (input_tokens, output_tokens) == (100, 20)


async def test_steps_exhausted_finalizes_run_with_eight_tool_calls(obs):
    llm = make_scripted_llm([echo_turn("echo again")])

    reply, _ = await run_with_telemetry(obs, llm)

    assert reply == STEPS_EXHAUSTED_MESSAGE
    assert run_row(obs) == (RUN_STATUS_STEPS_EXHAUSTED, 8, 8)


async def test_validation_retries_exhausted_finalizes_validation_error(obs):
    llm = make_scripted_llm([broken_json_turn()])

    reply, _ = await run_with_telemetry(obs, llm)

    assert reply == JSON_RETRIES_EXHAUSTED_MESSAGE
    # 3 LLM-хода; отказы валидации первых двух зафиксированы как ошибочные
    # tool-вызовы (design D3), третий — терминал до исполнения
    assert run_row(obs) == (RUN_STATUS_VALIDATION_ERROR, 3, 2)


async def test_truncated_response_finalizes_validation_error(obs):
    llm = make_scripted_llm([broken_json_turn(finish_reason="length")])

    reply, _ = await run_with_telemetry(obs, llm)

    assert reply == RESPONSE_TRUNCATED_MESSAGE
    assert run_row(obs) == (RUN_STATUS_VALIDATION_ERROR, 1, 0)


# --- Телеметрия tool-вызовов (task 4.1) ---


async def test_tool_call_recorded_with_sizes_duration_and_success(obs):
    llm = make_scripted_llm(
        [echo_turn("echo hi"), assistant_turn("готово", usage=usage())]
    )

    await run_with_telemetry(obs, llm)

    (tool_name, input_size, output_size, output_tokens, ok, turn_number) = rows(
        obs.path,
        "SELECT tool_name, input_size, output_size, output_tokens, ok, turn_number "
        "FROM tool_calls",
    )[0]
    assert tool_name == "exec"
    assert input_size == len('{"command": "echo hi"}')
    assert output_size > 0
    assert output_tokens == output_size // 4
    assert ok == 1
    assert turn_number == 1  # инструмент приписан ходу LLM, который его родил
    (duration_ms,) = rows(obs.path, "SELECT duration_ms FROM tool_calls")[0]
    assert duration_ms >= 0.0


async def test_validation_rejection_recorded_as_error_with_zero_duration(obs):
    llm = make_scripted_llm([broken_json_turn(), assistant_turn("исправился")])

    await run_with_telemetry(obs, llm)

    (ok, duration_ms) = rows(obs.path, "SELECT ok, duration_ms FROM tool_calls")[0]
    assert ok == 0
    assert duration_ms == 0.0


async def test_execution_error_recorded_as_failed_tool_call(obs):
    from tests.conftest import BrokenHistorySearcher

    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[
                    tool_call(name="search_history", arguments='{"query": "деплой"}')
                ],
                finish_reason="tool_calls",
            ),
            assistant_turn("повторю позже"),
        ]
    )
    recorder = new_recorder(obs)
    await recorder.start()
    client = ObservingClient(llm, recorder, model="fake-model")

    await run_agent(
        client,
        [{"role": "user", "content": "запрос"}],
        tools=TOOLS,
        executor=FakeCommandExecutor(),
        history_search=BrokenHistorySearcher(),
        recorder=recorder,
    )

    (tool_name, ok) = rows(obs.path, "SELECT tool_name, ok FROM tool_calls")[0]
    assert tool_name == "search_history"
    assert ok == 0


# --- Ветка LLMUnavailable в handle_text (task 4.3) ---


async def test_llm_unavailable_branch_finalizes_run_as_llm_error(obs, memory):
    llm = FakeLLM(error=LLMUnavailable("connection refused"))

    await handle_text(
        FakeMessage("привет", chat_id=CHAT_ID),
        FakeBot(),
        llm,
        memory,
        {},
        FakeCommandExecutor(),
        telemetry=obs,
    )

    status = run_row(obs)[0]
    assert status == RUN_STATUS_LLM_ERROR
    # Неудачная попытка зафиксирована: ok=0, без токенов, с длительностью
    (ok, input_tokens, latency_ms) = rows(
        obs.path, "SELECT ok, input_tokens, latency_ms FROM llm_calls"
    )[0]
    assert ok == 0
    assert input_tokens is None
    assert latency_ms >= 0.0


# --- Герметичный прогон: телеметрия не меняет ответы агента ---


async def test_handle_text_with_telemetry_records_full_run(obs, memory, monkeypatch):
    monkeypatch.delenv("OBS_PRICE_INPUT_PER_M", raising=False)
    monkeypatch.delenv("OBS_PRICE_OUTPUT_PER_M", raising=False)
    llm = make_scripted_llm(
        [
            assistant_turn(None, [tool_call()], "tool_calls", usage=usage(50, 30)),
            assistant_turn("готово", usage=usage(80, 10)),
        ]
    )
    bot = FakeBot()

    await handle_text(
        FakeMessage("сделай что-нибудь", chat_id=CHAT_ID),
        bot,
        llm,
        memory,
        {},
        FakeCommandExecutor(),
        telemetry=obs,
    )

    assert bot.sent[-1]["text"] == "готово"
    assert await memory.load_open_history(CHAT_ID) == [
        {"role": "user", "content": "сделай что-нибудь"},
        {"role": "assistant", "content": "готово"},
    ]
    assert run_row(obs) == (RUN_STATUS_SUCCESS, 2, 1)
    # Прайс из config по умолчанию (0.11/0.60) — стоимость прогона на записи
    (input_tokens, output_tokens, cost) = rows(
        obs.path, "SELECT input_tokens, output_tokens, estimated_cost FROM runs"
    )[0]
    assert (input_tokens, output_tokens) == (130, 40)
    assert cost == pytest.approx((130 * 0.11 + 40 * 0.60) / 1_000_000)


async def test_telemetry_does_not_change_agent_behavior(obs):
    """Одинаковый сценарий с телеметрией и без: ответы и запросы к LLM идентичны."""
    def make_llm() -> FakeLLM:
        return make_scripted_llm(
            [
                assistant_turn(None, [tool_call()], "tool_calls", usage=usage(5, 2)),
                assistant_turn("готово", usage=usage(7, 3)),
            ]
        )

    plain_llm = make_llm()
    reply_plain = await run_agent(
        plain_llm,
        [{"role": "user", "content": "запрос"}],
        tools=TOOLS,
        executor=FakeCommandExecutor(),
    )

    observed_llm = make_llm()
    reply_observed, _ = await run_with_telemetry(obs, observed_llm)

    assert reply_observed == reply_plain == "готово"
    assert observed_llm.requests == plain_llm.requests
    assert observed_llm.tools_per_request == plain_llm.tools_per_request


async def test_run_without_recorder_behaves_as_before():
    """recorder=None — прежнее поведение run_agent без телеметрии."""
    llm = make_scripted_llm([echo_turn("echo hi"), assistant_turn("готово")])

    reply = await run_agent(
        llm,
        [{"role": "user", "content": "запрос"}],
        tools=TOOLS,
        executor=FakeCommandExecutor(),
    )

    assert reply == "готово"
    assert len(llm.requests) == 2
