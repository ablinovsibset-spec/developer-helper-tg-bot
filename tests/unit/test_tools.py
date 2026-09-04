from __future__ import annotations

from dev_helper_bot.tools import (
    HEAD_CHARS,
    LIST_TOOL_NAME,
    LIST_TOOL_SPEC,
    OUTPUT_LIMIT,
    TAIL_CHARS,
    ExecResult,
    exec_command,
    truncate_output,
)

from tests.conftest import BrokenExecutor, FakeCommandExecutor


async def test_exec_successful_command_returns_stdout_and_zero_exit():
    fake = FakeCommandExecutor(
        scripted={"echo hello": ExecResult(exit_code=0, stdout="hello\n", stderr="")}
    )

    result = await exec_command(fake, "echo hello")

    assert "exit_code: 0" in result
    assert "hello" in result
    assert "(пусто)" in result  # stderr пуст


async def test_exec_nonzero_exit_returns_stderr_and_exit_code():
    fake = FakeCommandExecutor(
        scripted={
            "echo oops >&2; exit 3": ExecResult(
                exit_code=3, stdout="", stderr="oops\n"
            )
        }
    )

    result = await exec_command(fake, "echo oops >&2; exit 3")

    assert "exit_code: 3" in result
    assert "oops" in result


async def test_exec_long_output_is_truncated_head_and_tail():
    fake = FakeCommandExecutor(
        default=ExecResult(exit_code=0, stdout="y\n" * 5000, stderr="")
    )

    result = await exec_command(fake, "yes | head -c 10000")

    assert len(result) < OUTPUT_LIMIT + 100
    assert "обрезано" in result
    assert result.startswith("exit_code:")  # голова: начало форматированного вывода
    assert result.endswith("(пусто)")  # хвост: конец форматированного вывода
    assert "y\ny" in result


async def test_exec_timeout_reports_marker_and_exit_code():
    fake = FakeCommandExecutor(
        default=ExecResult(exit_code=124, stdout="", stderr="", timed_out=True)
    )

    result = await exec_command(fake, "sleep 5", timeout=0.2)

    assert "Таймаут 0.2с" in result
    assert "exit_code: 124" in result


async def test_exec_infrastructure_error_returned_as_text_not_raised():
    result = await exec_command(BrokenExecutor(), "echo hi")

    assert "Не удалось выполнить команду" in result
    assert "docker daemon is down" in result


def test_truncate_output_keeps_short_text_intact():
    assert truncate_output("короткий") == "короткий"
    assert len(truncate_output("a" * OUTPUT_LIMIT)) == OUTPUT_LIMIT


def test_truncate_output_cuts_middle_with_marker():
    text = "a" * 2000 + "B" * 2000

    truncated = truncate_output(text)

    assert len(truncated) < len(text)
    assert truncated.startswith("a" * HEAD_CHARS)
    assert truncated.endswith("B" * TAIL_CHARS)
    assert "обрезано 1000 символов" in truncated


def test_list_tool_spec_declares_no_parameters():
    function = LIST_TOOL_SPEC["function"]

    assert function["name"] == LIST_TOOL_NAME == "list_sessions"
    assert function["parameters"] == {"type": "object", "properties": {}}
    assert not (function["parameters"].get("required") or [])
