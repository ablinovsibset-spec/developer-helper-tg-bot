"""Персистентный индекс документов на SQLite + sqlite-vec
(change add-document-rag, design D1/D2/D10).

Отдельный от переписки файл БД на VM-локальном диске: у индекса другой
lifecycle и своё расширение SQLite. Владелец каждого документа — Telegram
`user_id` (design D3), и он же PARTITION KEY векторной таблицы: KNN физически
не выходит за документы своего пользователя, а не фильтрует чужое после
поиска.

Драйвер соединения — `sqlean.py`, а не stdlib `sqlite3` (design D10): многие
сборки CPython собраны без загрузки расширений, и sqlite-vec на них не
поднять. Асинхронность прежняя: `aiosqlite` получает фабрику соединения.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import sqlean
import sqlite_vec

from dev_helper_bot.embeddings import EmbeddingClient, EmbeddingsUnavailable

DEFAULT_SEARCH_K = 5
"""Top-K по умолчанию (design D7): хватает для attribution и не раздувает
результат инструмента под бюджет компакции агентного цикла."""

SEARCH_SNIPPET_CHARS = 600
"""Потолок одного фрагмента в результате инструмента: Top-K фрагментов
по 800 символов иначе съедают бюджет tool-вывода агентного цикла."""

SEARCH_NOT_FOUND_TEMPLATE = (
    "В загруженных документах нет ничего по запросу {query!r}. "
    "Не отвечай общими знаниями как содержимым документа."
)

SEARCH_EMBEDDINGS_ERROR = (
    "Поиск по документам сейчас недоступен: сервис эмбеддингов не отвечает."
)

SEARCH_NO_DOCUMENTS_MESSAGE = (
    "У пользователя нет загруженных документов — искать негде. "
    "Предложи прислать файл (.txt, .md, .docx, .pdf)."
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, filename)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id, filename);
"""

_VEC_SCHEMA_TEMPLATE = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0("
    "chunk_id INTEGER PRIMARY KEY, "
    "embedding float[{dim}] distance_metric=cosine, "
    "user_id INTEGER PARTITION KEY)"
)

_SEARCH_SQL = (
    "SELECT d.filename, c.chunk_index, c.text, m.distance FROM ("
    " SELECT chunk_id, distance FROM chunk_vectors"
    " WHERE embedding MATCH ? AND user_id = ? AND k = ?"
    ") m "
    "JOIN chunks c ON c.id = m.chunk_id "
    "JOIN documents d ON d.id = c.document_id "
    "WHERE d.user_id = ? "
    "ORDER BY m.distance"
)

_LIST_SQL = (
    "SELECT d.filename, d.created_at, COUNT(c.id) "
    "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
    "WHERE d.user_id = ? "
    "GROUP BY d.id "
    "ORDER BY d.filename"
)


class DocumentStoreUnavailable(RuntimeError):
    """Индекс документов нельзя открыть: нет расширения sqlite-vec или файл
    создан под другую размерность векторов. Fail-fast при старте (design D10):
    RAG — обязательная часть бота, молча работать без него нечестно."""


@dataclass(frozen=True)
class DocumentInfo:
    """Документ пользователя для команды /documents."""

    filename: str
    created_at: str
    chunk_count: int


@dataclass(frozen=True)
class ChunkMatch:
    """Найденный фрагмент: текст плюс источник для attribution."""

    filename: str
    chunk_index: int
    text: str
    distance: float


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentStore:
    """Документы, их chunks и векторы одного файла БД.

    Жизненным циклом владеет main, как памятью и телеметрией: open при
    старте, close в finally.
    """

    def __init__(self, db_path: str | Path, dimension: int) -> None:
        self._path = Path(db_path).expanduser()
        self._dimension = dimension
        self._db: aiosqlite.Connection | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    async def open(self) -> None:
        """Открывает/создаёт файл БД, грузит sqlite-vec, создаёт схему."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = aiosqlite.Connection(
            lambda: sqlean.connect(str(self._path)), iter_chunk_size=64
        )
        await self._db
        try:
            await self._db.enable_load_extension(True)
            await self._db.load_extension(sqlite_vec.loadable_path())
            await self._db.enable_load_extension(False)
        except Exception as exc:
            await self.close()
            raise DocumentStoreUnavailable(
                f"Не удалось загрузить расширение sqlite-vec для {self._path}: "
                f"{exc}. Проверьте, что установлены пакеты sqlite-vec и sqlean.py."
            ) from exc
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA)
        await self._db.execute(_VEC_SCHEMA_TEMPLATE.format(dim=self._dimension))
        await self._db.commit()
        await self._ensure_dimension()

    async def _ensure_dimension(self) -> None:
        """Сверяет размерность существующей векторной таблицы с конфигом.

        `CREATE ... IF NOT EXISTS` оставит таблицу от прошлой модели
        эмбеддингов, и расхождение вылезло бы как невнятная ошибка вставки.
        Лучше сказать прямо: сменили модель — сотрите файл индекса.
        """
        cursor = await self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunk_vectors'"
        )
        row = await cursor.fetchone()
        expected = f"float[{self._dimension}]"
        if row is not None and expected not in (row[0] or ""):
            await self.close()
            raise DocumentStoreUnavailable(
                f"Индекс {self._path} создан под другую размерность векторов, "
                f"ожидается {expected}. Смена EMBEDDING_MODEL/EMBEDDING_DIM "
                "требует удалить файл индекса и загрузить документы заново."
            )

    async def close(self) -> None:
        db, self._db = self._db, None
        if db is not None:
            await db.close()

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("DocumentStore не открыт: сначала вызовите open()")
        return self._db

    async def index_document(
        self,
        user_id: int,
        filename: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> int:
        """Пишет документ с его chunks и векторами; при совпадении имени
        заменяет прошлую версию (design D4).

        Удаление старого и вставка нового — одна транзакция с единственным
        commit: окно, в котором документ отсутствует в индексе, не видно
        другим запросам.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks/vectors mismatch: {len(chunks)} vs {len(vectors)}"
            )
        if any(len(vector) != self._dimension for vector in vectors):
            raise ValueError(
                f"vectors must have dimension {self._dimension}"
            )
        await self._delete_rows(user_id, filename)
        cursor = await self._conn.execute(
            "INSERT INTO documents (user_id, filename, file_type, created_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, filename, Path(filename).suffix.lower(), _utcnow_iso()),
        )
        document_id = cursor.lastrowid
        for index, (text, vector) in enumerate(zip(chunks, vectors)):
            cursor = await self._conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, text) "
                "VALUES (?, ?, ?)",
                (document_id, index, text),
            )
            await self._conn.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding, user_id) "
                "VALUES (?, ?, ?)",
                (
                    cursor.lastrowid,
                    sqlite_vec.serialize_float32(vector),
                    user_id,
                ),
            )
        await self._conn.commit()
        return len(chunks)

    async def _delete_rows(self, user_id: int, filename: str) -> bool:
        """Каскадное удаление векторов → chunks → документа (design D4).

        Каскад руками: vec0-таблица не участвует в foreign keys.
        Без commit — вызывающий решает, чем закрыть транзакцию.
        """
        cursor = await self._conn.execute(
            "SELECT id FROM documents WHERE user_id = ? AND filename = ?",
            (user_id, filename),
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        document_id = row[0]
        await self._conn.execute(
            "DELETE FROM chunk_vectors WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE document_id = ?)",
            (document_id,),
        )
        await self._conn.execute(
            "DELETE FROM chunks WHERE document_id = ?", (document_id,)
        )
        await self._conn.execute(
            "DELETE FROM documents WHERE id = ?", (document_id,)
        )
        return True

    async def delete_by_filename(self, user_id: int, filename: str) -> bool:
        """Удаляет документ пользователя по имени; False — такого нет."""
        deleted = await self._delete_rows(user_id, filename)
        await self._conn.commit()
        return deleted

    async def list_documents(self, user_id: int) -> list[DocumentInfo]:
        """Документы пользователя по алфавиту; чужие не видны."""
        cursor = await self._conn.execute(_LIST_SQL, (user_id,))
        rows = await cursor.fetchall()
        return [
            DocumentInfo(filename=filename, created_at=created_at, chunk_count=count)
            for filename, created_at, count in rows
        ]

    async def search(
        self,
        user_id: int,
        query_vector: list[float],
        k: int = DEFAULT_SEARCH_K,
    ) -> list[ChunkMatch]:
        """Top-K ближайших фрагментов среди документов пользователя.

        Изоляция держится на PARTITION KEY векторной таблицы (design D2):
        KNN выполняется внутри партиции владельца. Условие по
        `documents.user_id` оставлено вторым контуром — утечка чужого
        фрагмента не должна зависеть от одной реализации.
        """
        if len(query_vector) != self._dimension:
            raise ValueError(
                f"query vector must have dimension {self._dimension}"
            )
        cursor = await self._conn.execute(
            _SEARCH_SQL,
            (
                sqlite_vec.serialize_float32(query_vector),
                user_id,
                max(1, k),
                user_id,
            ),
        )
        rows = await cursor.fetchall()
        return [
            ChunkMatch(
                filename=filename,
                chunk_index=chunk_index,
                text=text,
                distance=distance,
            )
            for filename, chunk_index, text, distance in rows
        ]


def _snippet(text: str, chars: int = SEARCH_SNIPPET_CHARS) -> str:
    text = text.strip()
    if len(text) <= chars:
        return text
    return text[:chars] + "…"


def format_matches(matches: list[ChunkMatch], query: str) -> str:
    """Результат поиска для модели: фрагменты с явным источником.

    Имя файла и номер фрагмента стоят перед текстом, чтобы attribution
    («Источник: …») собиралась из того же сообщения, где взят факт.
    Пустой результат — прямая инструкция не выдумывать (design D11).
    """
    if not matches:
        return SEARCH_NOT_FOUND_TEMPLATE.format(query=query)
    lines = [f"Найдено фрагментов: {len(matches)}"]
    for match in matches:
        lines.append(
            f"— [источник: {match.filename}, фрагмент {match.chunk_index + 1}]\n"
            f"{_snippet(match.text)}"
        )
    return "\n".join(lines)


class UserDocumentSearcher:
    """Поиск по документам одного пользователя: эмбеддинг запроса + KNN.

    Реализует шов `tools.DocumentSearcher`, привязывая операцию к владельцу
    документов — `user_id` отправителя сообщения (design D3). Ошибка
    эмбеддингов возвращается текстом: недоступность поиска не должна
    ронять агентный цикл.
    """

    def __init__(
        self,
        store: DocumentStore,
        embeddings: EmbeddingClient,
        user_id: int,
        k: int = DEFAULT_SEARCH_K,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._user_id = user_id
        self._k = k

    async def search(self, query: str) -> str:
        if not await self._store.list_documents(self._user_id):
            return SEARCH_NO_DOCUMENTS_MESSAGE
        try:
            vectors = await self._embeddings.embed([query])
        except EmbeddingsUnavailable:
            return SEARCH_EMBEDDINGS_ERROR
        matches = await self._store.search(self._user_id, vectors[0], self._k)
        return format_matches(matches, query)
