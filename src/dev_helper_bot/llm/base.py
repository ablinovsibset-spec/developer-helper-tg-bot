from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: str
    content: str


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, messages: list[Message]) -> str:
        ...


class LLMUnavailable(Exception):
    pass
