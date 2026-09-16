"""LLM-as-a-Judge: субъект luna и судья sol на routerai, не на LLM_BASE_URL бота."""
from __future__ import annotations

import json
import os

import pytest

from dev_helper_bot.agent import run_agent
from tests.conftest import agent_eval_cases
from tests.live.conftest import ROUTERAI_BASE_URL

pytestmark = pytest.mark.live

JUDGE_THRESHOLD = 0.8
JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "politeness": {"type": "number"},
                "accuracy": {"type": "number"},
                "conciseness": {"type": "number"},
            },
            "required": ["politeness", "accuracy", "conciseness"],
            "additionalProperties": False,
        },
    },
}


def parse_judge_scores(content: str | None) -> dict[str, float]:
    if not content or not str(content).strip():
        raise AssertionError("судья вернул пустой ответ")
    text = str(content).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise AssertionError(f"битый JSON ответа судьи: {text!r}")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise AssertionError(f"битый JSON ответа судьи: {text!r}")
    if not isinstance(payload, dict):
        raise AssertionError(f"битый JSON ответа судьи: {text!r}")
    scores: dict[str, float] = {}
    for key in ("politeness", "accuracy", "conciseness"):
        if key not in payload:
            raise AssertionError(f"в ответе судьи нет поля {key}: {payload!r}")
        value = float(payload[key])
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"{key}={value} вне диапазона 0–1")
        scores[key] = value
    return scores


def test_broken_judge_json_is_case_failure():
    with pytest.raises(AssertionError, match="битый JSON"):
        parse_judge_scores("это не json {")


def test_routerai_clients_are_not_bot_llm_base_url(routerai_subject, routerai_judge):
    assert routerai_subject._base_url == ROUTERAI_BASE_URL
    assert routerai_judge._base_url == ROUTERAI_BASE_URL
    bot_base = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
    if bot_base and bot_base != ROUTERAI_BASE_URL:
        assert routerai_subject._base_url != bot_base
        assert routerai_judge._base_url != bot_base


async def test_five_open_answers_average_at_least_threshold(
    routerai_subject, routerai_judge, fake_executor
):
    cases = agent_eval_cases("judge")
    assert len(cases) >= 5
    means: list[float] = []
    for case in cases[:5]:
        question = case["turns"][0]["content"]
        history = [
            {
                "role": "system",
                "content": "Ты помощник. Отвечай кратко, точно и вежливо.",
            },
            {"role": "user", "content": question},
        ]
        reply = await run_agent(
            routerai_subject, history, tools=None, executor=fake_executor
        )
        judge_turn = await routerai_judge.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты судья качества ответа. Верни JSON с полями "
                        "politeness, accuracy, conciseness — числа от 0 до 1."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Вопрос:\n{question}\n\nОтвет:\n{reply}\n\n"
                        "Оцени вежливость, точность и краткость."
                    ),
                },
            ],
            response_format=JUDGE_RESPONSE_FORMAT,
        )
        scores = parse_judge_scores(judge_turn["content"])
        means.append(sum(scores.values()) / len(scores))
    average = sum(means) / len(means)
    assert average >= JUDGE_THRESHOLD, f"среднее судьи {average:.3f} < {JUDGE_THRESHOLD}"
