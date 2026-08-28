from __future__ import annotations

import time

from dev_helper_bot.tools import (
    HEAD_CHARS,
    OUTPUT_LIMIT,
    TAIL_CHARS,
    exec_command,
    truncate_output,
)


async def test_exec_successful_command_returns_stdout_and_zero_exit():
    result = await exec_command("echo hello")

    assert "exit_code: 0" in result
    assert "hello" in result
    assert "(пусто)" in result  # stderr пуст


async def test_exec_nonzero_exit_returns_stderr_and_exit_code():
    result = await exec_command("echo oops >&2; exit 3")

    assert "exit_code: 3" in result
    assert "oops" in result


async def test_exec_long_output_is_truncated_head_and_tail():
    result = await exec_command("yes | head -c 10000")

    assert len(result) < OUTPUT_LIMIT + 100
    assert "обрезано" in result
    assert result.startswith("exit_code:")  # голова: начало форматированного вывода
    assert result.endswith("(пусто)")  # хвост: конец форматированного вывода
    assert "y\ny" in result


async def test_exec_timeout_kills_process_and_reports():
    start = time.monotonic()

    result = await exec_command("sleep 5", timeout=0.2)

    elapsed = time.monotonic() - start
    assert elapsed < 3
    assert "Таймаут 0.2с" in result


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
