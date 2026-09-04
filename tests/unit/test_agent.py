from __future__ import annotations

import re

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
from dev_helper_bot.memory import (
    SEARCH_EXCERPT_LIMIT,
    SEARCH_NOT_FOUND_TEMPLATE,
    ChatHistorySearcher,
    MemoryStore,
)
from dev_helper_bot.tools import EXEC_TOOL_SPEC, SEARCH_TOOL_SPEC

from tests.conftest import (
    FakeCommandExecutor,
    assistant_turn,
    make_scripted_llm,
    tool_call,
)

TOOLS = [EXEC_TOOL_SPEC, SEARCH_TOOL_SPEC]
CHAT_ID = 42
OTHER_CHAT_ID = 4242
DATE_IN_BRACKETS = re.compile(r"\[\d{4}-\d{2}-\d{2}\]")


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


def search_turn(query: str) -> dict:
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(name="search_history", arguments=f'{{"query": "{query}"}}')],
        finish_reason="tool_calls",
    )


def tool_result(llm, request_index: int = 1) -> str:
    tool_msgs = [m for m in llm.requests[request_index] if m["role"] == "tool"]
    assert tool_msgs, "tool-сообщение не найдено в запросе"
    return tool_msgs[-1]["content"]


@pytest.fixture
async def store(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.open()
    yield store
    await store.close()


async def run_search_agent(store, llm) -> str:
    return await run_agent(
        llm,
        new_history(),
        tools=TOOLS,
        executor=FakeCommandExecutor(),
        history_search=ChatHistorySearcher(store, CHAT_ID),
    )


async def test_search_history_returns_excerpts_from_completed_session(store):
    await store.append_user(CHAT_ID, "обсудили деплой базы")
    await store.append_assistant(CHAT_ID, "итог: используем sqlite")
    await store.close_session(CHAT_ID)
    llm = make_scripted_llm(
        [search_turn("деплой"), assistant_turn(content="вот что было")]
    )

    reply = await run_search_agent(store, llm)

    assert reply == "вот что было"
    result = tool_result(llm)
    assert "обсудили деплой базы" in result
    assert "user" in result
    assert DATE_IN_BRACKETS.search(result)


async def test_search_history_not_found_returns_explicit_marker(store):
    await store.append_user(CHAT_ID, "разговор о погоде")
    await store.close_session(CHAT_ID)
    llm = make_scripted_llm(
        [search_turn("квантовая физика"), assistant_turn(content="не помню такого")]
    )

    await run_search_agent(store, llm)

    assert tool_result(llm) == SEARCH_NOT_FOUND_TEMPLATE.format(
        query="квантовая физика"
    )


async def test_search_history_volume_is_limited(store):
    await store.append_user(CHAT_ID, "начало")
    for i in range(SEARCH_EXCERPT_LIMIT + 2):
        await store.append_user(CHAT_ID, f"запись маркер {i:02d}")
    await store.close_session(CHAT_ID)
    llm = make_scripted_llm(
        [search_turn("маркер"), assistant_turn(content="нашёл выдержки")]
    )

    await run_search_agent(store, llm)

    assert tool_result(llm).count("маркер") == SEARCH_EXCERPT_LIMIT


async def test_search_history_isolates_chats(store):
    await store.append_user(OTHER_CHAT_ID, "чужой секретный маркер")
    await store.close_session(OTHER_CHAT_ID)
    llm = make_scripted_llm(
        [search_turn("маркер"), assistant_turn(content="пусто")]
    )

    await run_search_agent(store, llm)

    assert tool_result(llm) == SEARCH_NOT_FOUND_TEMPLATE.format(query="маркер")


async def test_search_history_skips_open_session(store):
    await store.append_user(CHAT_ID, "открытая сессия про деплой")
    llm = make_scripted_llm(
        [search_turn("деплой"), assistant_turn(content="в текущем контексте")]
    )

    await run_search_agent(store, llm)

    assert tool_result(llm) == SEARCH_NOT_FOUND_TEMPLATE.format(query="деплой")


async def test_search_history_is_read_only(store):
    await store.append_user(CHAT_ID, "запись про маркер")
    await store.append_assistant(CHAT_ID, "итог сессии")
    await store.close_session(CHAT_ID)
    before = await store.load_open_history(CHAT_ID)
    llm = make_scripted_llm(
        [search_turn("маркер"), assistant_turn(content="нашёл")]
    )

    await run_search_agent(store, llm)

    assert await store.load_open_history(CHAT_ID) == before == []


async def test_search_history_without_searcher_reports_tool_error():
    llm = make_scripted_llm(
        [search_turn("что угодно"), assistant_turn(content="понял")]
    )

    reply = await run_agent(
        llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor()
    )

    assert reply == "понял"
    assert "недоступен" in tool_result(llm)


async def test_search_history_missing_query_validated_before_executor(store):
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(name="search_history", arguments="{}")],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="исправился"),
        ]
    )

    reply = await run_search_agent(store, llm)

    assert reply == "исправился"
    assert '"query"' in tool_result(llm)
