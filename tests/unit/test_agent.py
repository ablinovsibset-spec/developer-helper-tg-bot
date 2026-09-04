from __future__ import annotations

import pytest

from dev_helper_bot.agent import (
    JSON_RETRIES_EXHAUSTED_MESSAGE,
    MAX_LLM_STEPS,
    RESPONSE_TRUNCATED_MESSAGE,
    STEPS_EXHAUSTED_MESSAGE,
    run_agent,
    validate_final,
    validate_tool_call,
)
from dev_helper_bot.llm import LLMUnavailable, Message
from dev_helper_bot.tools import EXEC_TOOL_SPEC

from tests.conftest import (
    FakeCommandExecutor,
    assistant_turn,
    make_scripted_llm,
    tool_call,
)

TOOLS = [EXEC_TOOL_SPEC]


def new_history() -> list[Message]:
    return [{"role": "user", "content": "сделай что-нибудь"}]


def echo_turn(command: str) -> dict:
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "%s"}' % command)],
        finish_reason="tool_calls",
    )


def test_validate_tool_call_correct_json_returns_none():
    assert (
        validate_tool_call(
            tool_call(arguments='{"command": "echo hi"}'), EXEC_TOOL_SPEC
        )
        is None
    )


def test_validate_tool_call_broken_json_error_contains_parse_details():
    error = validate_tool_call(tool_call(arguments="не json"), EXEC_TOOL_SPEC)

    assert error is not None
    assert "Ошибка разбора arguments" in error
    assert "Expecting" in error  # конкретная ошибка разбора
    assert '"command"' in error  # ожидаемая форма


def test_validate_tool_call_missing_required_param():
    error = validate_tool_call(
        tool_call(arguments='{"cmd": "echo hi"}'), EXEC_TOOL_SPEC
    )

    assert error is not None
    assert "отсутствует" in error
    assert '"command"' in error


def test_validate_tool_call_non_string_param():
    error = validate_tool_call(
        tool_call(arguments='{"command": 5}'), EXEC_TOOL_SPEC
    )

    assert error is not None
    assert '"command"' in error
    assert "должен быть строкой" in error


def test_validate_tool_call_non_object_json():
    error = validate_tool_call(tool_call(arguments="[1, 2]"), EXEC_TOOL_SPEC)

    assert error is not None
    assert "JSON-объектом" in error


def test_validate_final_is_always_valid_for_now():
    assert validate_final("любой текст") is None
    assert validate_final("") is None


async def test_tool_call_then_final_returns_final_text():
    llm = make_scripted_llm(
        [echo_turn("echo agent-test"), assistant_turn(content="готово")]
    )
    history = new_history()
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, history, tools=TOOLS, executor=executor)

    assert reply == "готово"
    assert len(llm.requests) == 2

    second_request = llm.requests[1]
    assert second_request[0] == {"role": "user", "content": "сделай что-нибудь"}
    assistant_msg = second_request[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    tool_msg = second_request[2]
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": tool_msg["content"],
    }
    assert "agent-test" in tool_msg["content"]
    assert "exit_code: 0" in tool_msg["content"]
    assert history[-1] == {"role": "assistant", "content": "готово"}


async def test_final_without_tools_ends_loop_immediately():
    llm = make_scripted_llm([assistant_turn(content="просто ответ")])
    history = new_history()
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, history, tools=TOOLS, executor=executor)

    assert reply == "просто ответ"
    assert len(llm.requests) == 1
    assert not any(m["role"] == "tool" for m in history)
    assert executor.commands == []


async def test_endless_tool_calls_stop_at_step_limit():
    endless = echo_turn("echo again")
    llm = make_scripted_llm([endless])
    history = new_history()
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, history, tools=TOOLS, executor=executor)

    assert reply == STEPS_EXHAUSTED_MESSAGE
    assert len(llm.requests) == MAX_LLM_STEPS == 8
    assert len([m for m in history if m["role"] == "tool"]) == 8


async def test_broken_arguments_returned_as_tool_error_and_loop_continues():
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(arguments="не json")],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="исправился"),
        ]
    )

    reply = await run_agent(
        llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor()
    )

    assert reply == "исправился"
    tool_msg = llm.requests[1][2]
    assert tool_msg["role"] == "tool"
    assert "Ошибка разбора arguments" in tool_msg["content"]


async def test_missing_command_argument_returned_as_tool_error():
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(arguments='{"cmd": "echo hi"}')],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="ок"),
        ]
    )

    reply = await run_agent(
        llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor()
    )

    assert reply == "ок"
    tool_msg = llm.requests[1][2]
    assert '"command"' in tool_msg["content"]


async def test_unknown_tool_returned_as_tool_error():
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(name="destroy_everything")],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="ладно"),
        ]
    )

    reply = await run_agent(
        llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor()
    )

    assert reply == "ладно"
    tool_msg = llm.requests[1][2]
    assert "неизвестный инструмент" in tool_msg["content"]


async def test_nonzero_exit_is_returned_to_model_not_raised():
    llm = make_scripted_llm([echo_turn("exit 7"), assistant_turn(content="понял ошибку")])
    executor = FakeCommandExecutor()

    reply = await run_agent(
        llm, new_history(), tools=TOOLS, executor=executor
    )

    assert reply == "понял ошибку"
    tool_msg = llm.requests[1][2]
    assert "exit_code: 7" in tool_msg["content"]


async def test_run_agent_does_not_stop_executor_after_final_reply():
    """Lifecycle жителя принадлежит main: цикл его не останавливает."""
    llm = make_scripted_llm([assistant_turn(content="готово")])
    executor = FakeCommandExecutor()

    await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert executor.stop_calls == 0


async def test_run_agent_does_not_stop_executor_after_steps_exhausted():
    llm = make_scripted_llm([echo_turn("echo again")])
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == STEPS_EXHAUSTED_MESSAGE
    assert executor.stop_calls == 0


async def test_run_agent_does_not_stop_executor_on_llm_error():
    llm = make_scripted_llm([])
    llm.error = LLMUnavailable("connection refused")
    executor = FakeCommandExecutor()

    with pytest.raises(LLMUnavailable):
        await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert executor.stop_calls == 0


async def test_file_created_on_one_step_is_visible_on_next_step():
    llm = make_scripted_llm(
        [
            echo_turn("echo data > note"),
            echo_turn("cat note"),
            assistant_turn(content="прочитал"),
        ]
    )
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == "прочитал"
    assert executor.commands == ["echo data > note", "cat note"]
    tool_msg = llm.requests[2][4]
    assert tool_msg["role"] == "tool"
    assert "data" in tool_msg["content"]


def broken_json_turn(finish_reason: str = "tool_calls") -> dict:
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "echo')],
        finish_reason=finish_reason,
    )


async def test_three_consecutive_validation_failures_return_distinct_terminal():
    llm = make_scripted_llm([broken_json_turn()])
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == JSON_RETRIES_EXHAUSTED_MESSAGE
    assert reply != STEPS_EXHAUSTED_MESSAGE
    assert len(llm.requests) == 3
    assert executor.commands == []


async def test_success_between_failures_resets_counter():
    llm = make_scripted_llm(
        [
            broken_json_turn(),
            echo_turn("echo hi"),
            broken_json_turn(),
            broken_json_turn(),
            assistant_turn(content="готово"),
        ]
    )
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == "готово"
    assert len(llm.requests) == 5
    assert executor.commands == ["echo hi"]


async def test_truncated_invalid_response_is_terminal_without_retries():
    llm = make_scripted_llm([broken_json_turn(finish_reason="length")])
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == RESPONSE_TRUNCATED_MESSAGE
    assert len(llm.requests) == 1
    assert executor.commands == []


async def test_valid_tool_call_with_length_finish_reason_is_accepted():
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(arguments='{"command": "echo hi"}')],
                finish_reason="length",
            ),
            assistant_turn(content="готово"),
        ]
    )
    executor = FakeCommandExecutor()

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == "готово"
    assert executor.commands == ["echo hi"]


async def test_failed_turn_stays_in_history_with_specific_error_feedback():
    llm = make_scripted_llm(
        [broken_json_turn(), assistant_turn(content="исправился")]
    )

    await run_agent(
        llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor()
    )

    second_request = llm.requests[1]
    broken_reply = second_request[1]
    assert broken_reply["role"] == "assistant"
    assert broken_reply["tool_calls"][0]["arguments"] == '{"command": "echo'
    feedback = second_request[2]
    assert feedback["role"] == "tool"
    assert feedback["tool_call_id"] == "call_1"
    assert "Ошибка разбора arguments" in feedback["content"]
