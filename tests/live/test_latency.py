"""Latency SLA короткого вызова LLM: полный ответ и TTFT (stream-зонд)."""
from __future__ import annotations

import os
import time

import aiohttp
import pytest

pytestmark = pytest.mark.live

DEFAULT_FULL_SECONDS = 4.0
DEFAULT_TTFT_SECONDS = 1.5


async def test_complete_under_full_threshold(live_llm):
    threshold = float(os.getenv("LIVE_FULL_SECONDS", DEFAULT_FULL_SECONDS))
    start = time.perf_counter()
    await live_llm.complete([{"role": "user", "content": "ping"}])
    elapsed = time.perf_counter() - start
    assert elapsed < threshold, f"complete() {elapsed:.3f}s >= {threshold}s"


async def test_ttft_under_threshold_or_skip_without_streaming(live_llm):
    threshold = float(os.getenv("LIVE_TTFT_SECONDS", DEFAULT_TTFT_SECONDS))
    url = f"{live_llm._base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if live_llm._api_key:
        headers["Authorization"] = f"Bearer {live_llm._api_key}"
    payload = {
        "model": live_llm._model,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True,
        "max_tokens": 8,
    }
    timeout = aiohttp.ClientTimeout(total=30)
    start = time.perf_counter()
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    pytest.skip(
                        f"endpoint does not support streaming (HTTP {resp.status})"
                    )
                async for chunk in resp.content.iter_any():
                    if chunk.strip():
                        elapsed = time.perf_counter() - start
                        assert elapsed < threshold, (
                            f"TTFT {elapsed:.3f}s >= {threshold}s"
                        )
                        return
    except aiohttp.ClientError as exc:
        pytest.skip(f"streaming probe failed: {exc}")
    pytest.skip("endpoint did not yield a stream chunk")
