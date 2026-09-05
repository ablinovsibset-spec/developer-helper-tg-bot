from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dev_helper_bot.llm import AssistantTurn, LLMUnavailable, Message, ResponseFormat
from dev_helper_bot.llm.openai_compat import (
    DEFAULT_TIMEOUT_SECONDS,
    OpenAICompatibleClient,
)
from dev_helper_bot.tools import EXEC_TOOL_SPEC

MODEL = "test-model"
MESSAGES: list[Message] = [{"role": "user", "content": "привет"}]

RESPONSE_FORMAT: ResponseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "reply",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
}


class FakeLLMServer:
    """Управляемый двойник OpenAI-совместимого /v1/chat/completions.

    Поля status/payload/delay меняются из теста между запросами;
    все входящие запросы (заголовки + JSON) записываются в requests.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.payload: dict[str, Any] = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "тест-ответ"},
                    "finish_reason": "stop",
                }
            ]
        }
        self.delay = 0.0
        self.reject_response_format = False

    async def handle(self, request: web.Request) -> web.Response:
        self.requests.append(
            {"headers": dict(request.headers), "json": await request.json()}
        )
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.status >= 400:
            return web.Response(status=self.status, text="server error")
        if self.reject_response_format and "response_format" in self.requests[-1]["json"]:
            return web.Response(status=400, text="response_format is not supported")
        return web.json_response(self.payload)


@pytest.fixture
async def fake_llm_server() -> FakeLLMServer:
    server = FakeLLMServer()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", server.handle)
    test_server = TestServer(app)
    await test_server.start_server()
    server.base_url = f"http://{test_server.host}:{test_server.port}/v1"
    yield server
    await test_server.close()


def make_client(base_url: str, **kwargs: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url=base_url, model=MODEL, **kwargs)


async def test_complete_without_tools_returns_turn_without_calls(fake_llm_server):
    client = make_client(fake_llm_server.base_url)

    turn = await client.complete(MESSAGES)

    assert turn == {
        "content": "тест-ответ",
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": None,
    }
    assert len(fake_llm_server.requests) == 1
    request = fake_llm_server.requests[0]
    assert request["json"] == {"model": MODEL, "messages": MESSAGES}
    assert "tools" not in request["json"]
    assert request["headers"]["Content-Type"] == "application/json"


def usage_payload(
    prompt: int | None = 10,
    completion: int | None = 5,
    cached: int | None = None,
    reasoning: int | None = None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return usage


async def test_complete_with_full_usage_carries_all_fields(fake_llm_server):
    """Сценарий «Ответ содержит usage»: полный usage — все четыре значения."""
    usage = usage_payload(prompt=100, completion=40, cached=60, reasoning=12)
    fake_llm_server.payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }
    client = make_client(fake_llm_server.base_url)

    turn = await client.complete(MESSAGES)

    assert turn["usage"] == {
        "input_tokens": 100,
        "output_tokens": 40,
        "cached_tokens": 60,
        "reasoning_tokens": 12,
        "raw": usage,
    }


async def test_complete_with_partial_usage_keeps_missing_fields_null(
    fake_llm_server,
):
    """Сценарий «Ответ без части полей»: cached/reasoning нет — они null."""
    usage = usage_payload(prompt=7, completion=3, cached=None, reasoning=None)
    fake_llm_server.payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }
    client = make_client(fake_llm_server.base_url)

    turn = await client.complete(MESSAGES)

    assert turn["usage"] is not None
    assert turn["usage"]["input_tokens"] == 7
    assert turn["usage"]["output_tokens"] == 3
    assert turn["usage"]["cached_tokens"] is None
    assert turn["usage"]["reasoning_tokens"] is None
    assert turn["usage"]["raw"] == usage


async def test_complete_without_usage_carries_null_usage(fake_llm_server):
    """Сценарий «Ответ без usage»: ход несёт null-usage и разбирается без ошибки."""
    fake_llm_server.payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ]
    }
    client = make_client(fake_llm_server.base_url)

    turn = await client.complete(MESSAGES)

    assert turn["content"] == "ok"
    assert turn["tool_calls"] == []
    assert turn["usage"] is None


async def test_complete_with_non_integer_usage_fields_keeps_null(fake_llm_server):
    """Мусор в usage не роняет разбор: нечисловые поля остаются null."""
    fake_llm_server.payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": "много", "completion_tokens": None},
    }
    client = make_client(fake_llm_server.base_url)

    turn = await client.complete(MESSAGES)

    assert turn["usage"] is not None
    assert turn["usage"]["input_tokens"] is None
    assert turn["usage"]["output_tokens"] is None


async def test_api_key_adds_bearer_header(fake_llm_server):
    client = make_client(fake_llm_server.base_url, api_key="secret-key")

    await client.complete(MESSAGES)

    headers = fake_llm_server.requests[0]["headers"]
    assert headers.get("Authorization") == "Bearer secret-key"


async def test_no_api_key_means_no_authorization_header(fake_llm_server):
    client = make_client(fake_llm_server.base_url)

    await client.complete(MESSAGES)

    headers = fake_llm_server.requests[0]["headers"]
    assert "Authorization" not in headers


async def test_http_error_raises_llm_unavailable(fake_llm_server):
    fake_llm_server.status = 500
    client = make_client(fake_llm_server.base_url)

    with pytest.raises(LLMUnavailable, match="HTTP 500"):
        await client.complete(MESSAGES)


async def test_malformed_response_raises_llm_unavailable(fake_llm_server):
    fake_llm_server.payload = {"error": "no choices here"}
    client = make_client(fake_llm_server.base_url)

    with pytest.raises(LLMUnavailable, match="Malformed"):
        await client.complete(MESSAGES)


async def test_timeout_raises_llm_unavailable(fake_llm_server):
    fake_llm_server.delay = 0.5
    client = make_client(fake_llm_server.base_url, timeout=0.2)

    with pytest.raises(LLMUnavailable, match="timed out after 0.2"):
        await client.complete(MESSAGES)


def test_default_timeout_is_120_seconds():
    client = OpenAICompatibleClient(base_url="http://localhost:1234/v1", model=MODEL)

    assert client._timeout == DEFAULT_TIMEOUT_SECONDS == 120.0


async def test_tool_calls_round_trip(fake_llm_server):
    """Полный круг: tool_calls из ответа → история с tool-сообщением → финал."""
    fake_llm_server.payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "внутренние рассуждения модели",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "exec",
                                "arguments": '{"command": "echo hi"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    client = make_client(fake_llm_server.base_url)

    turn = await client.complete(MESSAGES, tools=[EXEC_TOOL_SPEC])

    assert turn["content"] is None
    assert turn["finish_reason"] == "tool_calls"
    assert turn["tool_calls"] == [
        {"id": "call_1", "name": "exec", "arguments": '{"command": "echo hi"}'}
    ]
    assert fake_llm_server.requests[0]["json"]["tools"] == [EXEC_TOOL_SPEC]

    history: list[Message] = MESSAGES + [
        {
            "role": "assistant",
            "content": turn["content"],
            "tool_calls": turn["tool_calls"],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "exit_code: 0\nstdout:\nhi",
        },
    ]
    fake_llm_server.payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "финал"},
                "finish_reason": "stop",
            }
        ]
    }

    final_turn: AssistantTurn = await client.complete(history)

    assert final_turn["content"] == "финал"
    assert final_turn["tool_calls"] == []
    sent_messages = fake_llm_server.requests[1]["json"]["messages"]
    assert sent_messages[0] == {"role": "user", "content": "привет"}
    assert sent_messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "exec", "arguments": '{"command": "echo hi"}'},
            }
        ],
    }
    assert sent_messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "exit_code: 0\nstdout:\nhi",
    }


async def test_response_format_is_sent_in_payload(fake_llm_server):
    client = make_client(fake_llm_server.base_url)

    turn = await client.complete(MESSAGES, response_format=RESPONSE_FORMAT)

    assert turn["content"] == "тест-ответ"
    assert len(fake_llm_server.requests) == 1
    assert fake_llm_server.requests[0]["json"]["response_format"] == RESPONSE_FORMAT


async def test_no_response_format_leaves_payload_unchanged(fake_llm_server):
    client = make_client(fake_llm_server.base_url)

    await client.complete(MESSAGES, response_format=None)

    request = fake_llm_server.requests[0]
    assert request["json"] == {"model": MODEL, "messages": MESSAGES}
    assert "response_format" not in request["json"]


async def test_http_400_with_response_format_retries_without_field_and_caches(
    fake_llm_server,
):
    """Деградация: 400 → повтор без поля; кэш живёт до конца процесса клиента."""
    fake_llm_server.reject_response_format = True
    client = make_client(fake_llm_server.base_url)

    turn = await client.complete(MESSAGES, response_format=RESPONSE_FORMAT)

    assert turn["content"] == "тест-ответ"
    assert len(fake_llm_server.requests) == 2
    first, second = (r["json"] for r in fake_llm_server.requests)
    assert first["response_format"] == RESPONSE_FORMAT
    assert "response_format" not in second
    assert second == {"model": MODEL, "messages": MESSAGES}

    # Кэш: повторный вызов с response_format не кладёт поле в payload.
    await client.complete(MESSAGES, response_format=RESPONSE_FORMAT)

    assert len(fake_llm_server.requests) == 3
    assert "response_format" not in fake_llm_server.requests[2]["json"]


async def test_http_400_without_response_format_raises(fake_llm_server):
    """400 без response_format в запросе — настоящая ошибка, без повтора."""
    fake_llm_server.status = 400
    client = make_client(fake_llm_server.base_url)

    with pytest.raises(LLMUnavailable, match="HTTP 400"):
        await client.complete(MESSAGES)

    assert len(fake_llm_server.requests) == 1
