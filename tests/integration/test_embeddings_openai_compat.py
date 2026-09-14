"""OpenAI-совместимый клиент эмбеддингов против локального двойника сервера.

Тот же приём, что у chat-клиента: aiohttp TestServer на localhost вместо
реального провайдера — без сети за пределы loopback и без ключей.
"""
from __future__ import annotations

from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dev_helper_bot.embeddings import EmbeddingClient, EmbeddingsUnavailable
from dev_helper_bot.embeddings import openai_compat
from dev_helper_bot.embeddings.openai_compat import OpenAICompatibleEmbeddingClient

DIM = 4


def vector(seed: float) -> list[float]:
    return [seed, seed + 1, seed + 2, seed + 3]


class FakeEmbeddingServer:
    """Управляемый двойник `POST /v1/embeddings`."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.payload: dict[str, Any] | None = None
        self.fail_times = 0
        self.fail_status = 503
        self.base_url = ""

    async def handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append({"headers": dict(request.headers), "json": body})
        if self.fail_times > 0:
            self.fail_times -= 1
            return web.Response(status=self.fail_status, text="temporarily down")
        if self.status >= 400:
            return web.Response(status=self.status, text="bad request")
        if self.payload is not None:
            return web.json_response(self.payload)
        return web.json_response(
            {
                "data": [
                    {"index": index, "embedding": vector(float(index))}
                    for index, _ in enumerate(body["input"])
                ],
                "model": body["model"],
            }
        )


@pytest.fixture
async def server() -> FakeEmbeddingServer:
    fake = FakeEmbeddingServer()
    app = web.Application()
    app.router.add_post("/v1/embeddings", fake.handle)
    test_server = TestServer(app)
    await test_server.start_server()
    fake.base_url = f"http://{test_server.host}:{test_server.port}/v1"
    yield fake
    await test_server.close()


def client(server: FakeEmbeddingServer, **kwargs) -> OpenAICompatibleEmbeddingClient:
    options = {
        "base_url": server.base_url,
        "model": "baai/bge-m3",
        "dimension": DIM,
    } | kwargs
    return OpenAICompatibleEmbeddingClient(**options)


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ретраи проверяем без реальных пауз."""
    monkeypatch.setattr(openai_compat, "RETRY_DELAYS_SECONDS", (0.0, 0.0))


async def test_client_satisfies_embedding_protocol(server):
    assert isinstance(client(server), EmbeddingClient)


async def test_single_text_returns_one_vector_of_declared_dimension(server):
    embedder = client(server)

    vectors = await embedder.embed(["политика отпусков"])

    assert vectors == [vector(0.0)]
    assert len(vectors[0]) == embedder.dimension == DIM


async def test_batch_keeps_input_order_even_when_provider_shuffles(server):
    """Порядок задаёт поле index, а не позиция в массиве ответа."""
    server.payload = {
        "data": [
            {"index": 1, "embedding": vector(1.0)},
            {"index": 0, "embedding": vector(0.0)},
        ]
    }

    vectors = await client(server).embed(["первый", "второй"])

    assert vectors == [vector(0.0), vector(1.0)]


async def test_embedding_model_is_sent_and_differs_from_chat_model(server):
    await client(server, model="baai/bge-m3").embed(["текст"])

    assert server.requests[-1]["json"]["model"] == "baai/bge-m3"
    assert server.requests[-1]["json"]["input"] == ["текст"]


async def test_api_key_goes_to_authorization_header(server):
    await client(server, api_key="sk-secret").embed(["текст"])

    assert server.requests[-1]["headers"]["Authorization"] == "Bearer sk-secret"


async def test_no_api_key_sends_no_authorization_header(server):
    await client(server).embed(["текст"])

    assert "Authorization" not in server.requests[-1]["headers"]


async def test_empty_input_short_circuits_without_request(server):
    assert await client(server).embed([]) == []
    assert server.requests == []


async def test_http_error_raises_embeddings_unavailable(server):
    server.status = 401

    with pytest.raises(EmbeddingsUnavailable, match="HTTP 401"):
        await client(server).embed(["текст"])


async def test_malformed_payload_raises_embeddings_unavailable(server):
    server.payload = {"unexpected": "shape"}

    with pytest.raises(EmbeddingsUnavailable, match="Malformed"):
        await client(server).embed(["текст"])


async def test_vector_count_mismatch_raises_embeddings_unavailable(server):
    server.payload = {"data": [{"index": 0, "embedding": vector(0.0)}]}

    with pytest.raises(EmbeddingsUnavailable, match="1 vectors for 2 texts"):
        await client(server).embed(["первый", "второй"])


async def test_dimension_mismatch_is_reported_with_both_numbers(server):
    """Модель отдаёт не ту размерность, под которую создан индекс."""
    server.payload = {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}

    with pytest.raises(EmbeddingsUnavailable, match="dimension 2, configured 4"):
        await client(server).embed(["текст"])


async def test_temporary_failure_is_retried_then_succeeds(server):
    server.fail_times = 2

    vectors = await client(server).embed(["текст"])

    assert vectors == [vector(0.0)]
    assert len(server.requests) == 3


async def test_retries_are_exhausted_into_embeddings_unavailable(server):
    server.fail_times = 5

    with pytest.raises(EmbeddingsUnavailable, match="after 3 attempts"):
        await client(server).embed(["текст"])

    assert len(server.requests) == 3


async def test_unreachable_endpoint_raises_embeddings_unavailable():
    embedder = OpenAICompatibleEmbeddingClient(
        base_url="http://127.0.0.1:1/v1", model="m", dimension=DIM
    )

    with pytest.raises(EmbeddingsUnavailable):
        await embedder.embed(["текст"])
