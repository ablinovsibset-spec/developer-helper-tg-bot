from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from .base import LLMClient, LLMUnavailable, Message


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or None
        self._timeout = timeout

    async def complete(self, messages: list[Message]) -> str:
        url = f"{self._base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise LLMUnavailable(
                            f"LLM returned HTTP {resp.status}: {body[:200]}"
                        )
                    data = await resp.json()
        except asyncio.TimeoutError as exc:
            raise LLMUnavailable(
                f"LLM request timed out after {self._timeout}s"
            ) from exc
        except aiohttp.ClientError as exc:
            raise LLMUnavailable(f"LLM connection error: {exc}") from exc
        except LLMUnavailable:
            raise
        except Exception as exc:
            raise LLMUnavailable(f"Unexpected LLM error: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"Malformed LLM response: {data!r}") from exc
