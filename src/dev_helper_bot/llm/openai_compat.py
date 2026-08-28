from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from .base import (
    AssistantTurn,
    LLMClient,
    LLMUnavailable,
    Message,
    ToolCall,
    ToolSpec,
)

DEFAULT_TIMEOUT_SECONDS = 120.0


def _serialize_message(message: Message) -> dict[str, Any]:
    """Преобразует внутреннее Message в wire-формат OpenAI-совместимого API."""
    wire: dict[str, Any] = {
        "role": message["role"],
        "content": message["content"],
    }
    tool_calls = message.get("tool_calls")
    if tool_calls:
        wire["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for call in tool_calls
        ]
    tool_call_id = message.get("tool_call_id")
    if tool_call_id is not None:
        wire["tool_call_id"] = tool_call_id
    return wire


def _parse_turn(data: dict[str, Any]) -> AssistantTurn:
    """Извлекает ход ассистента из ответа /chat/completions.

    `reasoning` (если поставщик его прислал) отбрасывается: в контракт
    и в историю он не попадает.
    """
    choice = data["choices"][0]
    message = choice["message"]
    tool_calls: list[ToolCall] = [
        {
            "id": raw["id"],
            "name": raw["function"]["name"],
            "arguments": raw["function"].get("arguments") or "",
        }
        for raw in message.get("tool_calls") or []
    ]
    return {
        "content": message.get("content"),
        "tool_calls": tool_calls,
        "finish_reason": choice.get("finish_reason") or "stop",
    }


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or None
        self._timeout = timeout

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AssistantTurn:
        url = f"{self._base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_serialize_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools

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
            return _parse_turn(data)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"Malformed LLM response: {data!r}") from exc
