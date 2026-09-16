"""Контракт JSON-аргументов инструментов: параметризация по evaluation dataset."""
from __future__ import annotations

import pytest

from dev_helper_bot.agent import validate_tool_call
from dev_helper_bot.tools import EXEC_TOOL_SPEC, GET_SKILL_TOOL_SPEC, READ_FILE_TOOL_SPEC
from tests.conftest import agent_eval_cases, tool_call

SPECS = {
    "exec": EXEC_TOOL_SPEC,
    "read_file": READ_FILE_TOOL_SPEC,
    "get_skill": GET_SKILL_TOOL_SPEC,
}

CASES = agent_eval_cases("tool_schema")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_validate_tool_call_matches_dataset(case):
    spec = SPECS[case["tool_name"]]
    error = validate_tool_call(
        tool_call(name=case["tool_name"], arguments=case["arguments"]),
        spec,
    )
    expected = case["expect"]
    if expected["ok"]:
        assert error is None
        return
    assert error is not None
    for fragment in expected["error_contains"]:
        assert fragment in error
