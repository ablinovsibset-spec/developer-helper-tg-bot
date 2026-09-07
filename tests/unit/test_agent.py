from __future__ import annotations

import logging
import re

import pytest

from dev_helper_bot.agent import (
    JSON_RETRIES_EXHAUSTED_MESSAGE,
    MAX_LLM_STEPS,
    RESPONSE_TRUNCATED_MESSAGE,
    STEPS_EXHAUSTED_MESSAGE,
    TOOL_OUTPUT_BUDGET_CHARS,
    compact_tool_outputs,
    compacted_tool_stub,
    run_agent,
    validate_final,
    validate_tool_call,
)
from dev_helper_bot.llm import LLMUnavailable, Message
from dev_helper_bot.memory import (
    LIST_SESSIONS_EMPTY_MARKER,
    SEARCH_EXCERPT_LIMIT,
    SEARCH_NOT_FOUND_TEMPLATE,
    ChatHistorySearcher,
    MemoryStore,
)
from dev_helper_bot.tools import (
    EXEC_TOOL_SPEC,
    GET_SKILL_TOOL_SPEC,
    LIST_TOOL_SPEC,
    READ_FILE_TOOL_SPEC,
    SEARCH_TOOL_SPEC,
    ExecResult,
)

from tests.conftest import (
    BrokenHistorySearcher,
    FakeCommandExecutor,
    assistant_turn,
    make_scripted_llm,
    tool_call,
)

TOOLS = [
    EXEC_TOOL_SPEC,
    READ_FILE_TOOL_SPEC,
    GET_SKILL_TOOL_SPEC,
    SEARCH_TOOL_SPEC,
    LIST_TOOL_SPEC,
]
CHAT_ID = 42
OTHER_CHAT_ID = 4242
DATE_IN_BRACKETS = re.compile(r"\[\d{4}-\d{2}-\d{2}\]")
AGENT_LOGGER_NAME = "dev_helper_bot.agent"


def agent_log_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == AGENT_LOGGER_NAME]


def new_history() -> list[Message]:
    return [{"role": "user", "content": "сделай что-нибудь"}]


def echo_turn(command: str) -> dict:
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "%s"}' % command)],
        finish_reason="tool_calls",
    )


def read_file_turn(
    path: str = "/work/notes.md",
    arguments: str | None = None,
) -> dict:
    if arguments is None:
        arguments = f'{{"path": "{path}"}}'
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(name="read_file", arguments=arguments)],
        finish_reason="tool_calls",
    )


async def test_read_file_call_returns_numbered_lines_to_model():
    executor = FakeCommandExecutor()
    executor.files["/work/notes.md"] = "первая\nвторая\nтретья"
    llm = make_scripted_llm(
        [
            read_file_turn(arguments='{"path": "/work/notes.md", "offset": 2, "limit": 1}'),
            assistant_turn(content="прочитал"),
        ]
    )

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == "прочитал"
    tool_msg = llm.requests[1][2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"].splitlines() == [
        "2: вторая",
        "[ответ обрезан: продолжай со строки 3]",
    ]


async def test_file_created_via_exec_is_readable_by_read_file():
    """Одна песочница: файл из exec виден read_file согласованно."""
    executor = FakeCommandExecutor()
    llm = make_scripted_llm(
        [
            echo_turn("echo data > note"),
            read_file_turn(path="note"),
            assistant_turn(content="прочитал"),
        ]
    )

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == "прочитал"
    assert executor.commands == ["echo data > note", "sed -n 1,101p note"]
    tool_msg = llm.requests[2][4]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"].splitlines() == ["1: data"]


async def test_read_file_missing_file_error_does_not_break_loop():
    executor = FakeCommandExecutor()  # файла нет
    llm = make_scripted_llm(
        [
            read_file_turn(path="/work/nope.md"),
            assistant_turn(content="файла нет, но я справился"),
        ]
    )

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == "файла нет, но я справился"
    tool_msg = llm.requests[1][2]
    assert "Ошибка чтения файла" in tool_msg["content"]


async def test_get_skill_call_returns_body_to_model():
    from dev_helper_bot.skills import Skill

    skills = {
        "morning": Skill(
            name="morning",
            description="Утро",
            body="Шаг 1. Погода в Минске\nШаг 2. Habr",
        )
    }
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[
                    tool_call(
                        name="get_skill",
                        arguments='{"name": "morning"}',
                    )
                ],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="сводка готова"),
        ]
    )

    reply = await run_agent(
        llm,
        new_history(),
        tools=TOOLS,
        executor=FakeCommandExecutor(),
        skills=skills,
    )

    assert reply == "сводка готова"
    tool_msg = llm.requests[1][2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == skills["morning"].body
    assert "name:" not in tool_msg["content"]
    assert "---" not in tool_msg["content"]


def test_validate_tool_call_get_skill_requires_name():
    error = validate_tool_call(
        tool_call(name="get_skill", arguments="{}"), GET_SKILL_TOOL_SPEC
    )

    assert error is not None
    assert '"name"' in error


def test_validate_tool_call_read_file_requires_path():
    error = validate_tool_call(
        tool_call(name="read_file", arguments="{}"), READ_FILE_TOOL_SPEC
    )

    assert error is not None
    assert '"path"' in error

    assert (
        validate_tool_call(
            tool_call(
                name="read_file",
                arguments='{"path": "/work/a.md", "offset": 2}',
            ),
            READ_FILE_TOOL_SPEC,
        )
        is None
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


def list_sessions_turn() -> dict:
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(name="list_sessions", arguments="{}")],
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


def test_validate_tool_call_list_sessions_accepts_empty_arguments():
    assert (
        validate_tool_call(
            tool_call(name="list_sessions", arguments="{}"), LIST_TOOL_SPEC
        )
        is None
    )
    assert (
        validate_tool_call(tool_call(name="list_sessions", arguments=""), LIST_TOOL_SPEC)
        is None
    )


def test_validate_tool_call_list_sessions_broken_json_reports_error():
    error = validate_tool_call(
        tool_call(name="list_sessions", arguments="не json"), LIST_TOOL_SPEC
    )

    assert error is not None
    assert "Ошибка разбора arguments" in error


async def test_list_sessions_delegates_to_seam_and_returns_its_result(store):
    await store.append_user(CHAT_ID, "беседа про деплой базы")
    await store.append_assistant(CHAT_ID, "итог: используем sqlite")
    await store.close_session(CHAT_ID)
    llm = make_scripted_llm(
        [list_sessions_turn(), assistant_turn(content="вот что было раньше")]
    )

    reply = await run_search_agent(store, llm)

    assert reply == "вот что было раньше"
    result = tool_result(llm)
    assert "Завершённые беседы" in result
    assert DATE_IN_BRACKETS.search(result)
    assert "беседа про деплой базы" in result


async def test_list_sessions_without_completed_sessions_returns_marker(store):
    llm = make_scripted_llm(
        [list_sessions_turn(), assistant_turn(content="прошлых бесед нет")]
    )

    await run_search_agent(store, llm)

    assert tool_result(llm) == LIST_SESSIONS_EMPTY_MARKER


async def test_list_sessions_without_searcher_reports_tool_error():
    llm = make_scripted_llm(
        [list_sessions_turn(), assistant_turn(content="понял")]
    )

    reply = await run_agent(
        llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor()
    )

    assert reply == "понял"
    assert "недоступен" in tool_result(llm)


async def test_list_sessions_execution_error_returned_as_tool_text():
    llm = make_scripted_llm(
        [list_sessions_turn(), assistant_turn(content="повторю позже")]
    )

    reply = await run_agent(
        llm,
        new_history(),
        tools=TOOLS,
        executor=FakeCommandExecutor(),
        history_search=BrokenHistorySearcher(),
    )

    assert reply == "повторю позже"
    assert "Ошибка выполнения инструмента" in tool_result(llm)


async def test_successful_tool_call_is_logged_with_name_and_outcome(caplog):
    caplog.set_level(logging.INFO, logger=AGENT_LOGGER_NAME)
    llm = make_scripted_llm(
        [echo_turn("echo hi"), assistant_turn(content="готово")]
    )

    await run_agent(llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor())

    records = agent_log_records(caplog)
    assert len(records) == 1
    assert records[0].levelname == "INFO"
    assert "exec" in records[0].getMessage()
    assert "success" in records[0].getMessage()
    # Аргументы и содержимое результатов в журнал не попадают (design D4)
    assert "echo hi" not in records[0].getMessage()


async def test_execution_error_is_logged_with_tool_name(caplog):
    caplog.set_level(logging.INFO, logger=AGENT_LOGGER_NAME)
    llm = make_scripted_llm(
        [search_turn("деплой"), assistant_turn(content="повторю позже")]
    )

    await run_agent(
        llm,
        new_history(),
        tools=TOOLS,
        executor=FakeCommandExecutor(),
        history_search=BrokenHistorySearcher(),
    )

    records = agent_log_records(caplog)
    assert len(records) == 1
    assert records[0].levelname == "INFO"
    message = records[0].getMessage()
    assert "search_history" in message
    assert "error" in message


async def test_validation_rejected_call_is_logged_with_reason(caplog):
    caplog.set_level(logging.INFO, logger=AGENT_LOGGER_NAME)
    llm = make_scripted_llm([broken_json_turn(), assistant_turn(content="исправился")])

    await run_agent(llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor())

    records = agent_log_records(caplog)
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    message = records[0].getMessage()
    assert "exec" in message
    assert "Ошибка разбора arguments" in message


async def test_no_tool_log_records_without_tool_calls(caplog):
    caplog.set_level(logging.INFO, logger=AGENT_LOGGER_NAME)
    llm = make_scripted_llm([assistant_turn(content="просто ответ")])

    await run_agent(llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor())

    assert agent_log_records(caplog) == []


# --- Компакция tool-выводов (specs/agent-loop, design D2) ---


def big_output_history() -> list[Message]:
    """Прогон с двумя большими tool-выводами: сумма заметно выше бюджета."""
    return [
        {"role": "user", "content": "вопрос"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call(id="call_1", arguments='{"command": "first"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "A" * 5000},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call(id="call_2", arguments='{"command": "second"}')],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "B" * 5000},
    ]


def tool_total(history: list[Message]) -> int:
    return sum(len(m["content"] or "") for m in history if m["role"] == "tool")


def test_compact_over_budget_collapses_oldest_first():
    history = big_output_history()

    compact_tool_outputs(history)

    assert tool_total(history) <= TOOL_OUTPUT_BUDGET_CHARS
    assert history[2]["content"] == compacted_tool_stub(5000)
    assert "5000" in history[2]["content"]
    assert history[4]["content"] == "B" * 5000  # младший вывод цел


def test_compact_within_budget_changes_nothing():
    history = big_output_history()
    history[2]["content"] = "короткий вывод"
    history[4]["content"] = "ещё короче"
    before = [dict(m) for m in history]

    compact_tool_outputs(history)

    assert history == before


def test_compact_never_touches_dialog_messages():
    history = big_output_history()

    compact_tool_outputs(history)

    assert history[0] == {"role": "user", "content": "вопрос"}
    assert history[1]["tool_calls"][0]["arguments"] == '{"command": "first"}'
    assert history[3]["tool_calls"][0]["arguments"] == '{"command": "second"}'
    assert all(m["role"] != "tool" for i, m in enumerate(history) if i in (0, 1, 3))


async def test_run_agent_compacts_oldest_tool_output_in_context():
    """Три exec с выводом ~3000 симв.: на третьем сумма превышает бюджет,
    старейший вывод в контексте следующего хода — заглушка."""
    executor = FakeCommandExecutor(
        default=ExecResult(exit_code=0, stdout="y" * 3000, stderr="")
    )
    llm = make_scripted_llm(
        [
            echo_turn("big one"),
            echo_turn("big two"),
            echo_turn("big three"),
            assistant_turn(content="готово"),
        ]
    )

    reply = await run_agent(llm, new_history(), tools=TOOLS, executor=executor)

    assert reply == "готово"
    request = llm.requests[3]
    tool_msgs = [m for m in request if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    assert tool_msgs[0]["content"].startswith("[компакция:")
    assert tool_msgs[1]["content"] != "[компакция:"
    assert "y" * 100 in tool_msgs[2]["content"]  # самый свежий вывод цел
    # Команды остались в assistant-сообщениях — повторный вызов возможен
    assistant_calls = [m for m in request if m.get("tool_calls")]
    assert len(assistant_calls) == 3


async def test_run_agent_keeps_small_tool_outputs_intact():
    llm = make_scripted_llm(
        [echo_turn("echo hi"), assistant_turn(content="готово")]
    )

    await run_agent(llm, new_history(), tools=TOOLS, executor=FakeCommandExecutor())

    tool_msg = llm.requests[1][2]
    assert "hi" in tool_msg["content"]
    assert "[компакция:" not in tool_msg["content"]
