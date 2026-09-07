from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from .base import (
    AssistantTurn,
    LLMClient,
    LLMUnavailable,
    Message,
    ResponseFormat,
    ToolCall,
    ToolSpec,
    Usage,
)

DEFAULT_TIMEOUT_SECONDS = 120.0

MAX_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (1.0, 3.0)
"""Паузы между транспортными попытками POST (design D2): до 3 попыток
суммарно — первый повтор через 1с, второй через 3с."""

RETRYABLE_HTTP_STATUSES = frozenset({404, 500, 502, 503, 504})
"""404 при жёстком пине endpoint'а — «тир временно недоступен» (design D1);
остальные 4xx — постоянные ошибки запроса, без повторов."""


class _HTTPError(Exception):
    """HTTP-ответ >= 400 от поставщика; статус и тело для диагностики."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class _Retryable(Exception):
    """Временный сбой вызова (обрыв/таймаут/5xx/404) — кандидат на повтор."""


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


def _int_or_none(value: Any) -> int | None:
    """Число из ответа поставщика; всё остальное — null (absence ≠ 0)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _cost_or_none(value: Any) -> float | None:
    """Фактическая стоимость из usage ответа (design D3): число → значение
    (int/float, включая честный ноль), отсутствие/мусор — null."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_usage(data: dict[str, Any]) -> Usage | None:
    """Извлекает usage из ответа /chat/completions; отсутствие — None.

    Понимает OpenAI-совместимую форму (prompt_tokens/completion_tokens
    + *_tokens_details) и устойчива к неполному ответу: поля, которых
    нет, остаются null. `raw` хранит исходный объект как есть.
    Провайдерская стоимость (`cost`, routerai) проходит опциональным
    `billed_cost`: число → значение, отсутствие → null (absence ≠ 0).
    """
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    billed_cost = _cost_or_none(usage.get("cost"))
    parsed: Usage = {
        "input_tokens": _int_or_none(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        ),
        "output_tokens": _int_or_none(
            usage.get("completion_tokens", usage.get("output_tokens"))
        ),
        "cached_tokens": _int_or_none(
            (prompt_details or {}).get("cached_tokens")
            if isinstance(prompt_details, dict)
            else None
        ),
        "reasoning_tokens": _int_or_none(
            (completion_details or {}).get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        ),
        "raw": usage,
    }
    if "cost" in usage:
        parsed["billed_cost"] = billed_cost
    return parsed


def _parse_turn(data: dict[str, Any]) -> AssistantTurn:
    """Извлекает ход ассистента из ответа /chat/completions.

    `reasoning` (если поставщик его прислал) отбрасывается: в контракт
    и в историю он не попадает. `usage` проходит в ход: отсутствие —
    null, на разбор текста/вызовов не влияет.
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
        "usage": _parse_usage(data),
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
        self._response_format_supported = True

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_format: ResponseFormat | None = None,
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
        if response_format is not None and self._response_format_supported:
            payload["response_format"] = response_format

        try:
            data = await self._request_with_retries(url, headers, payload)
        except _HTTPError as exc:
            # Деградация: 400 на запрос с response_format — поставщик поле
            # не поддерживает. Повторяем без поля, факт кэшируется
            # до конца жизни процесса (design D5).
            if exc.status == 400 and "response_format" in payload:
                self._response_format_supported = False
                del payload["response_format"]
                data = await self._request_with_retries(url, headers, payload)
            else:
                raise LLMUnavailable(
                    f"LLM returned HTTP {exc.status}: {exc.body[:200]}"
                ) from exc

        try:
            return _parse_turn(data)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"Malformed LLM response: {data!r}") from exc

    async def _request_with_retries(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST к /chat/completions с ретраями временных сбоев (design D2).

        До MAX_ATTEMPTS попыток суммарно на обрыв соединения/ответа,
        таймаут и HTTP 5xx/404 (404 при жёстком пине — «endpoint временно
        недоступен»). Постоянные 4xx и исчерпание попыток — исключение
        без повторов. Повтор — новый POST целиком: частично прочитанный
        оборванный ответ не используется.
        """
        last_retryable: _Retryable | None = None
        for attempt in range(MAX_ATTEMPTS):
            if attempt > 0:
                await asyncio.sleep(RETRY_DELAYS_SECONDS[attempt - 1])
            try:
                return await self._request(url, headers, payload)
            except _Retryable as exc:
                last_retryable = exc
        raise LLMUnavailable(
            f"LLM unavailable after {MAX_ATTEMPTS} attempts: {last_retryable}"
        ) from last_retryable

    async def _request(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Один POST к /chat/completions; HTTP >= 400 — _HTTPError,
        временный сбой — _Retryable."""
        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        if resp.status in RETRYABLE_HTTP_STATUSES:
                            raise _Retryable(f"HTTP {resp.status}: {body}")
                        raise _HTTPError(resp.status, body)
                    return await resp.json()
        except (_HTTPError, _Retryable):
            raise
        except asyncio.TimeoutError as exc:
            raise _Retryable(
                f"LLM request timed out after {self._timeout}s"
            ) from exc
        except aiohttp.ClientError as exc:
            raise _Retryable(f"LLM connection error: {exc}") from exc
        except LLMUnavailable:
            raise
        except Exception as exc:
            raise LLMUnavailable(f"Unexpected LLM error: {exc}") from exc
