from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from .base import EmbeddingClient, EmbeddingsUnavailable

DEFAULT_TIMEOUT_SECONDS = 60.0

MAX_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (1.0, 3.0)
"""Паузы между транспортными попытками POST — тот же профиль, что у
chat-клиента (llm.openai_compat): до 3 попыток суммарно."""

RETRYABLE_HTTP_STATUSES = frozenset({404, 500, 502, 503, 504})


class _Retryable(Exception):
    """Временный сбой вызова (обрыв/таймаут/5xx/404) — кандидат на повтор."""


def _parse_vectors(data: dict[str, Any], expected: int) -> list[list[float]]:
    """Достаёт векторы из ответа `/embeddings` в порядке входных текстов.

    Поставщик вправе вернуть элементы не по порядку — порядок задаёт поле
    `index`, поэтому сортируем по нему, а не по позиции в массиве.
    """
    items = data["data"]
    if len(items) != expected:
        raise EmbeddingsUnavailable(
            f"Embeddings response has {len(items)} vectors for {expected} texts"
        )
    ordered = sorted(items, key=lambda item: item.get("index", 0))
    vectors = [[float(value) for value in item["embedding"]] for item in ordered]
    if any(not vector for vector in vectors):
        raise EmbeddingsUnavailable("Embeddings response contains an empty vector")
    return vectors


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """OpenAI-совместимый `POST /embeddings` (design D5).

    Тот же класс endpoint'ов и, как правило, тот же base URL / ключ, что у
    chat-клиента, но своя модель: `EMBEDDING_MODEL` конфигурируется отдельно
    от `LLM_MODEL`. Размерность задаётся конфигом, а не выводится из ответа:
    схема векторной таблицы создаётся до первого вызова провайдера.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._api_key = api_key or None
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self._model, "input": texts}

        data = await self._request_with_retries(url, headers, payload)
        try:
            vectors = _parse_vectors(data, len(texts))
        except EmbeddingsUnavailable:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EmbeddingsUnavailable(
                f"Malformed embeddings response: {data!r}"
            ) from exc
        mismatched = next(
            (len(v) for v in vectors if len(v) != self._dimension), None
        )
        if mismatched is not None:
            raise EmbeddingsUnavailable(
                f"Embedding model {self._model!r} returned dimension "
                f"{mismatched}, configured {self._dimension}"
            )
        return vectors

    async def _request_with_retries(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        last_retryable: _Retryable | None = None
        for attempt in range(MAX_ATTEMPTS):
            if attempt > 0:
                await asyncio.sleep(RETRY_DELAYS_SECONDS[attempt - 1])
            try:
                return await self._request(url, headers, payload)
            except _Retryable as exc:
                last_retryable = exc
        raise EmbeddingsUnavailable(
            f"Embeddings unavailable after {MAX_ATTEMPTS} attempts: {last_retryable}"
        ) from last_retryable

    async def _request(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Один POST к /embeddings; постоянная ошибка — EmbeddingsUnavailable,
        временный сбой — _Retryable."""
        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        if resp.status in RETRYABLE_HTTP_STATUSES:
                            raise _Retryable(f"HTTP {resp.status}: {body}")
                        raise EmbeddingsUnavailable(
                            f"Embeddings returned HTTP {resp.status}: {body[:200]}"
                        )
                    return await resp.json()
        except (_Retryable, EmbeddingsUnavailable):
            raise
        except asyncio.TimeoutError as exc:
            raise _Retryable(
                f"Embeddings request timed out after {self._timeout}s"
            ) from exc
        except aiohttp.ClientError as exc:
            raise _Retryable(f"Embeddings connection error: {exc}") from exc
        except Exception as exc:
            raise EmbeddingsUnavailable(f"Unexpected embeddings error: {exc}") from exc
