from __future__ import annotations

import re

from dev_helper_bot.tools import (
    EXEC_TOOL_SPEC,
    HEAD_CHARS,
    LIST_TOOL_NAME,
    LIST_TOOL_SPEC,
    OUTPUT_LIMIT,
    READ_FILE_DEFAULT_LIMIT,
    READ_FILE_OUTPUT_LIMIT,
    READ_FILE_TOOL_NAME,
    READ_FILE_TOOL_SPEC,
    TAIL_CHARS,
    ExecResult,
    exec_command,
    format_page,
    read_file,
    truncate_output,
)

from tests.conftest import BrokenExecutor, FakeCommandExecutor


def notes_executor(content: str, path: str = "/work/notes.md") -> FakeCommandExecutor:
    fake = FakeCommandExecutor()
    fake.files[path] = content
    return fake


def continuation_line(page: str) -> int:
    """Номер строки продолжения из подсказки страницы read_file."""
    match = re.search(r"продолжай со строки (\d+)", page)
    assert match, f"подсказка продолжения не найдена: {page!r}"
    return int(match.group(1))


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


# --- read_file (specs/read-file-tool, design D1) ---


async def test_read_file_range_with_offset_and_limit():
    fake = notes_executor("\n".join(f"строка {i:02d}" for i in range(1, 11)))

    result = await read_file(fake, "/work/notes.md", offset=3, limit=2)

    # За диапазоном есть строка 5 (детектор +1) → подсказка продолжения
    assert result.splitlines() == [
        "3: строка 03",
        "4: строка 04",
        "[ответ обрезан: продолжай со строки 5]",
    ]
    assert fake.commands == ["sed -n 3,5p /work/notes.md"]


async def test_read_file_range_reaching_eof_has_no_continuation_hint():
    fake = notes_executor("\n".join(f"строка {i:02d}" for i in range(1, 11)))

    result = await read_file(fake, "/work/notes.md", offset=9, limit=2)

    assert result.splitlines() == ["9: строка 09", "10: строка 10"]


async def test_read_file_without_offset_and_limit_reads_first_lines():
    fake = notes_executor("\n".join(f"line {i}" for i in range(1, 6)))

    result = await read_file(fake, "/work/notes.md")

    assert result.splitlines() == [f"{i}: line {i}" for i in range(1, 6)]
    assert fake.commands == [
        f"sed -n 1,{1 + READ_FILE_DEFAULT_LIMIT}p /work/notes.md"
    ]


async def test_read_file_cut_by_limit_reports_continuation():
    fake = notes_executor("\n".join(f"line {i}" for i in range(1, 11)))

    result = await read_file(fake, "/work/notes.md", offset=2, limit=3)

    body = result.splitlines()
    assert body[:3] == ["2: line 2", "3: line 3", "4: line 4"]
    assert continuation_line(result) == 5  # за диапазоном есть ещё строки


async def test_read_file_cut_by_ceiling_reports_continuation():
    # 40 строк по ~100 символов: суммарно много больше потолка 1500
    fake = notes_executor("\n".join(f"{i:03d}-" + "x" * 96 for i in range(1, 41)))

    result = await read_file(fake, "/work/notes.md")

    assert len(result) <= READ_FILE_OUTPUT_LIMIT  # жёсткий потолок всего ответа
    body = result.splitlines()
    assert body[0].startswith("1: ")
    assert body[-1] == f"[ответ обрезан: продолжай со строки {len(body)}]"
    assert continuation_line(result) == len(body)  # нумерация без пропусков
    assert body[-2].startswith(f"{len(body) - 1}: ")


async def test_read_file_missing_file_returns_error_text_not_exception():
    fake = FakeCommandExecutor()  # файлов нет

    result = await read_file(fake, "/work/nope.md")

    assert "Ошибка чтения файла /work/nope.md" in result
    assert "No such file" in result


async def test_read_file_infra_error_returned_as_text():
    result = await read_file(BrokenExecutor(), "/work/notes.md")

    assert result.startswith("Не удалось выполнить команду")


async def test_read_file_empty_file_reports_empty():
    fake = notes_executor("")

    result = await read_file(fake, "/work/notes.md")

    assert "пуст" in result


def test_format_page_single_overlong_line_is_hard_cut():
    page = format_page(["y" * 5000], 1, has_more=False)

    assert len(page) <= READ_FILE_OUTPUT_LIMIT
    assert page.startswith("1: ")
    assert continuation_line(page) == 2


def test_read_file_spec_declares_optional_offset_and_limit():
    function = READ_FILE_TOOL_SPEC["function"]
    parameters = function["parameters"]

    assert function["name"] == READ_FILE_TOOL_NAME == "read_file"
    assert parameters["required"] == ["path"]
    assert parameters["properties"]["path"]["type"] == "string"
    assert parameters["properties"]["offset"]["type"] == "integer"
    assert parameters["properties"]["limit"]["type"] == "integer"


def test_exec_description_directs_files_to_read_file():
    description = EXEC_TOOL_SPEC["function"]["description"]

    assert "read_file" in description
    assert "cat" in description
