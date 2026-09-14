from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Шов векторных представлений текста — отдельный от chat-LLM (design D5).

    Chat и embeddings разведены сознательно: у них разные endpoint'ы, разные
    модели и разная цена ошибки (падение embeddings ломает индексацию, но не
    диалог). `dimension` объявляет клиент, потому что схема векторной таблицы
    создаётся под конкретную размерность и не может её угадать.
    """

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingsUnavailable(Exception):
    """Провайдер эмбеддингов недоступен, ответил ошибкой или прислал
    malformed payload. Отдельный тип от LLMUnavailable: обработчик загрузки
    документа отвечает про индексацию, а не про недоступность диалога."""
