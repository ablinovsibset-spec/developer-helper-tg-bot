"""Разбор JSON судьи: целиком валидный документ, без вырезания подстроки."""
from __future__ import annotations

import pytest

from tests.live.test_llm_judge import parse_judge_scores


def test_broken_judge_json_is_case_failure():
    with pytest.raises(AssertionError, match="битый JSON"):
        parse_judge_scores("это не json {")


def test_judge_json_with_leading_commentary_is_case_failure():
    with pytest.raises(AssertionError, match="битый JSON"):
        parse_judge_scores(
            'commentary {"politeness": 1, "accuracy": 1, "conciseness": 1}'
        )


def test_valid_judge_json_document_parses():
    scores = parse_judge_scores(
        '{"politeness": 0.9, "accuracy": 0.8, "conciseness": 0.7}'
    )
    assert scores == {
        "politeness": 0.9,
        "accuracy": 0.8,
        "conciseness": 0.7,
    }
