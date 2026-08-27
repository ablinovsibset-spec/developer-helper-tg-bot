from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dev_helper_bot.llm import LLMUnavailable, Message
from dev_helper_bot.llm.openai_compat import OpenAICompatibleClient

MODEL = "test-model"
MESSAGES: list[Message] = [{"role": "user", "content": "привет"}]


class FakeLLMServer:
    """Управляемый двойник OpenAI-совместимого /v1/chat/completions.

    Поля status/payload/delay меняются из теста между запросами;
    все входящие запросы (заголовки + JSON) записываются в requests.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.payload: dict[str, Any] = {
            "choices": [{"message": {"role": "assistant", "content": "тест-ответ"}}]
        }
        self.delay = 0.0

    async def handle(self, request: web.Request) -> web.Response:
        self.requests.append(
            {"headers": dict(request.headers), "json": await request.json()}
        )
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.status >= 400:
            return web.Response(status=self.status, text="server error")
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


async def test_complete_returns_content_and_sends_valid_payload(fake_llm_server):
    client = make_client(fake_llm_server.base_url)

    reply = await client.complete(MESSAGES)

    assert reply == "тест-ответ"
    assert len(fake_llm_server.requests) == 1
    request = fake_llm_server.requests[0]
    assert request["json"] == {"model": MODEL, "messages": MESSAGES}
    assert request["headers"]["Content-Type"] == "application/json"


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
