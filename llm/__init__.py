from .base import LLMClient, LLMUnavailable, Message
from .openai_compat import OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "LLMUnavailable",
    "Message",
    "OpenAICompatibleClient",
]


def make_llm() -> LLMClient:
    from config import make_llm as _make_llm

    return _make_llm()
