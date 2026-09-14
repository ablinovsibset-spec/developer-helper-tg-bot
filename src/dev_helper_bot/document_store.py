"""Персистентный индекс документов на SQLite + sqlite-vec
(change add-document-rag, design D1/D2/D10; бонусы — add-rag-bonus).

Отдельный от переписки файл БД на VM-локальном диске: у индекса другой
lifecycle и своё расширение SQLite. Владелец каждого документа — Telegram
`user_id` (design D3), и он же PARTITION KEY векторной таблицы: KNN физически
не выходит за документы своего пользователя, а не фильтрует чужое после
поиска. Текстовый канал — FTS5 по chunks с тем же фильтром `user_id`.

Драйвер соединения — `sqlean.py`, а не stdlib `sqlite3` (design D10): многие
сборки CPython собраны без загрузки расширений, и sqlite-vec на них не
поднять. Асинхронность прежняя: `aiosqlite` получает фабрику соединения.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import sqlean
import sqlite_vec

from dev_helper_bot.documents import TextChunk
from dev_helper_bot.embeddings import EmbeddingClient, EmbeddingsUnavailable

log = logging.getLogger(__name__)

DEFAULT_SEARCH_K = 5
"""Top-K по умолчанию (design D7): хватает для attribution и не раздувает
результат инструмента под бюджет компакции агентного цикла."""

CANDIDATE_K = 15
"""Ширина каждого канала до RRF (add-rag-bonus D5): шире финального Top-K."""

RRF_K = 60
"""Константа Reciprocal Rank Fusion: вклад канала = 1/(60 + rank)."""

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

_FTS_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]+", re.UNICODE)

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
    text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id, filename);
"""

_FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5("
    "text, content='chunks', content_rowid='id')"
)

_VEC_SCHEMA_TEMPLATE = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0("
    "chunk_id INTEGER PRIMARY KEY, "
    "embedding float[{dim}] distance_metric=cosine, "
    "user_id INTEGER PARTITION KEY)"
)

_SEARCH_SQL = (
    "SELECT c.id, d.filename, c.chunk_index, c.text, m.distance, "
    "c.page_start, c.page_end FROM ("
    " SELECT chunk_id, distance FROM chunk_vectors"
    " WHERE embedding MATCH ? AND user_id = ? AND k = ?"
    ") m "
    "JOIN chunks c ON c.id = m.chunk_id "
    "JOIN documents d ON d.id = c.document_id "
    "WHERE d.user_id = ? "
    "ORDER BY m.distance"
)

_FTS_SQL = (
    "SELECT c.id, d.filename, c.chunk_index, c.text, "
    "c.page_start, c.page_end "
    "FROM chunk_fts "
    "JOIN chunks c ON c.id = chunk_fts.rowid "
    "JOIN documents d ON d.id = c.document_id "
    "WHERE chunk_fts MATCH ? AND d.user_id = ? "
    "ORDER BY rank "
    "LIMIT ?"
)

_LIST_SQL = (
    "SELECT d.filename, d.created_at, COUNT(c.id) "
    "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
    "WHERE d.user_id = ? "
    "GROUP BY d.id "
    "ORDER BY d.filename"
)


class DocumentStoreUnavailable(RuntimeError):
    """Индекс документов нельзя открыть: нет sqlite-vec / FTS5 или файл
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
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: int = 0


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_fts_query(query: str) -> str:
    """Литералы MATCH: слова в кавычках через OR, без операторов FTS5."""
    tokens = _FTS_TOKEN.findall(query)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens[:32])


async def fts5_is_available(db: aiosqlite.Connection) -> bool:
    """Есть ли FTS5 в этой сборке SQLite (sqlean)."""
    try:
        cursor = await db.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')")
        row = await cursor.fetchone()
    except Exception:
        return False
    return bool(row and row[0])


def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def lexical_rerank_score(query: str, text: str, rrf: float) -> float:
    """RRF + точное вхождение строки запроса + перекрытие токенов (D5)."""
    query_l = query.lower().strip()
    text_l = text.lower()
    exact = 1.0 if query_l and query_l in text_l else 0.0
    q_tokens = set(re.findall(r"\w+", query_l, flags=re.UNICODE))
    t_tokens = set(re.findall(r"\w+", text_l, flags=re.UNICODE))
    overlap = (len(q_tokens & t_tokens) / len(q_tokens)) if q_tokens else 0.0
    return rrf + exact + overlap


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
        """Открывает/создаёт файл БД, грузит sqlite-vec и FTS5, создаёт схему."""
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
        if not await fts5_is_available(self._db):
            await self.close()
            raise DocumentStoreUnavailable(
                f"В SQLite для {self._path} нет FTS5. "
                "Проверьте, что установлен пакет sqlean.py со сборкой SQLite с FTS5."
            )
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA)
        await self._db.execute(_VEC_SCHEMA_TEMPLATE.format(dim=self._dimension))
        await self._migrate_chunk_pages()
        await self._ensure_fts()
        await self._db.commit()
        await self._ensure_dimension()

    async def _migrate_chunk_pages(self) -> None:
        """Добавляет page_start/page_end к уже существующей таблице chunks."""
        cursor = await self._conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "page_start" not in columns:
            await self._conn.execute(
                "ALTER TABLE chunks ADD COLUMN page_start INTEGER"
            )
        if "page_end" not in columns:
            await self._conn.execute(
                "ALTER TABLE chunks ADD COLUMN page_end INTEGER"
            )

    async def _ensure_fts(self) -> None:
        """Создаёт FTS и пересобирает индекс из chunks (миграция D8).

        У content-таблицы SELECT/COUNT читают chunks, а не инвертированный
        индекс: без `rebuild` MATCH на старом rag.db был бы пуст.
        """
        await self._conn.execute(_FTS_SCHEMA)
        await self._conn.execute(
            "INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')"
        )

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
        chunks: list[str] | list[TextChunk],
        vectors: list[list[float]],
    ) -> int:
        """Пишет документ с его chunks и векторами; при совпадении имени
        заменяет прошлую версию (design D4).

        Удаление старого и вставка нового — одна транзакция с единственным
        commit: окно, в котором документ отсутствует в индексе, не видно
        другим запросам. FTS синхронизируется в той же транзакции, что vec.
        """
        normalized = [
            item if isinstance(item, TextChunk) else TextChunk(text=item)
            for item in chunks
        ]
        if len(normalized) != len(vectors):
            raise ValueError(
                f"chunks/vectors mismatch: {len(normalized)} vs {len(vectors)}"
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
        for index, (chunk, vector) in enumerate(zip(normalized, vectors)):
            cursor = await self._conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, text, "
                "page_start, page_end) VALUES (?, ?, ?, ?, ?)",
                (
                    document_id,
                    index,
                    chunk.text,
                    chunk.page_start,
                    chunk.page_end,
                ),
            )
            chunk_id = cursor.lastrowid
            await self._conn.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding, user_id) "
                "VALUES (?, ?, ?)",
                (
                    chunk_id,
                    sqlite_vec.serialize_float32(vector),
                    user_id,
                ),
            )
            await self._conn.execute(
                "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                (chunk_id, chunk.text),
            )
        await self._conn.commit()
        return len(normalized)

    async def _delete_rows(self, user_id: int, filename: str) -> bool:
        """Каскадное удаление FTS → векторов → chunks → документа (design D4).

        Каскад руками: vec0 и FTS не участвуют в foreign keys.
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
            "DELETE FROM chunk_fts WHERE rowid IN "
            "(SELECT id FROM chunks WHERE document_id = ?)",
            (document_id,),
        )
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
        """Top-K ближайших фрагментов среди документов пользователя (vec KNN).

        Изоляция держится на PARTITION KEY векторной таблицы (design D2):
        KNN выполняется внутри партиции владельца. Условие по
        `documents.user_id` оставлено вторым контуром — утечка чужого
        фрагмента не должна зависеть от одной реализации.
        Публичный отбор для инструмента — `retrieve` (гибрид + rerank).
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
        return [_match_from_vec_row(row) for row in rows]

    async def search_text(
        self,
        user_id: int,
        query: str,
        k: int = DEFAULT_SEARCH_K,
    ) -> list[ChunkMatch]:
        """Текстовый канал: FTS5 MATCH по chunks этого пользователя.

        Ошибка MATCH не пробрасывается: канал молча пуст, чтобы гибридный
        поиск не ронял агентный цикл (vec при этом остаётся).
        """
        match_query = sanitize_fts_query(query)
        if not match_query:
            return []
        try:
            cursor = await self._conn.execute(
                _FTS_SQL, (match_query, user_id, max(1, k))
            )
            rows = await cursor.fetchall()
        except Exception:
            log.warning("FTS MATCH failed for query %r", query, exc_info=True)
            return []
        return [_match_from_fts_row(row) for row in rows]

    async def retrieve(
        self,
        user_id: int,
        query: str,
        query_vector: list[float],
        k: int = DEFAULT_SEARCH_K,
    ) -> list[ChunkMatch]:
        """Гибридный отбор: vec + FTS → RRF → лексический rerank → Top-K.

        Кандидатный набор каждого канала шире финального k (CANDIDATE_K).
        Чат-LLM не вызывается.
        """
        top_k = max(1, k)
        candidate_k = max(CANDIDATE_K, top_k)
        vec_matches = await self.search(user_id, query_vector, k=candidate_k)
        fts_matches = await self.search_text(user_id, query, k=candidate_k)
        merged = _rrf_merge(vec_matches, fts_matches)
        ranked = sorted(
            merged,
            key=lambda item: lexical_rerank_score(query, item[0].text, item[1]),
            reverse=True,
        )
        return [match for match, _rrf in ranked[:top_k]]


def _match_from_vec_row(row: tuple) -> ChunkMatch:
    chunk_id, filename, chunk_index, text, distance, page_start, page_end = row
    return ChunkMatch(
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        distance=distance,
        page_start=page_start,
        page_end=page_end,
        chunk_id=chunk_id,
    )


def _match_from_fts_row(row: tuple) -> ChunkMatch:
    chunk_id, filename, chunk_index, text, page_start, page_end = row
    return ChunkMatch(
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        distance=float("inf"),
        page_start=page_start,
        page_end=page_end,
        chunk_id=chunk_id,
    )


def _rrf_merge(
    vec_matches: list[ChunkMatch],
    fts_matches: list[ChunkMatch],
) -> list[tuple[ChunkMatch, float]]:
    """Объединяет id фрагментов двух каналов через Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    by_id: dict[int, ChunkMatch] = {}
    for rank, match in enumerate(vec_matches, start=1):
        by_id[match.chunk_id] = match
        scores[match.chunk_id] = scores.get(match.chunk_id, 0.0) + rrf_score(rank)
    for rank, match in enumerate(fts_matches, start=1):
        by_id.setdefault(match.chunk_id, match)
        scores[match.chunk_id] = scores.get(match.chunk_id, 0.0) + rrf_score(rank)
    return [(by_id[chunk_id], score) for chunk_id, score in scores.items()]


def _snippet(text: str, chars: int = SEARCH_SNIPPET_CHARS) -> str:
    text = text.strip()
    if len(text) <= chars:
        return text
    return text[:chars] + "…"


def page_label(match: ChunkMatch) -> str | None:
    """«стр. N» или «стр. N–M»; None, если страницы нет — не выдумывать."""
    if match.page_start is None:
        return None
    if match.page_end is None or match.page_end == match.page_start:
        return f"стр. {match.page_start}"
    return f"стр. {match.page_start}–{match.page_end}"


def format_matches(matches: list[ChunkMatch], query: str) -> str:
    """Результат поиска для модели: фрагменты с явным источником.

    Имя файла, страница PDF (если известна) и номер фрагмента стоят перед
    текстом, чтобы attribution («Источник: …») собиралась из того же
    сообщения, где взят факт. Пустой результат — прямая инструкция не
    выдумывать (design D11).
    """
    if not matches:
        return SEARCH_NOT_FOUND_TEMPLATE.format(query=query)
    lines = [f"Найдено фрагментов: {len(matches)}"]
    for match in matches:
        source = match.filename
        page = page_label(match)
        if page:
            source = f"{source}, {page}"
        lines.append(
            f"— [источник: {source}, фрагмент {match.chunk_index + 1}]\n"
            f"{_snippet(match.text)}"
        )
    return "\n".join(lines)


class UserDocumentSearcher:
    """Поиск по документам одного пользователя: эмбеддинг запроса + retrieve.

    Реализует шов `tools.DocumentSearcher`, привязывая операцию к владельцу
    документов — `user_id` отправителя сообщения (design D3). Ошибка
    эмбеддингов возвращается текстом: недоступность поиска не должна
    ронять агентный цикл. Предыдущие реплики сессии подмешиваются в
    поисковую строку (add-rag-bonus D6), контракт `search(query)` прежний.
    """

    def __init__(
        self,
        store: DocumentStore,
        embeddings: EmbeddingClient,
        user_id: int,
        k: int = DEFAULT_SEARCH_K,
        conversation_context: str = "",
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._user_id = user_id
        self._k = k
        self._conversation_context = conversation_context.strip()

    def _retrieval_query(self, query: str) -> str:
        if not self._conversation_context:
            return query
        return f"{self._conversation_context}\n{query}"

    async def search(self, query: str) -> str:
        if not await self._store.list_documents(self._user_id):
            return SEARCH_NO_DOCUMENTS_MESSAGE
        retrieval_query = self._retrieval_query(query)
        try:
            vectors = await self._embeddings.embed([retrieval_query])
        except EmbeddingsUnavailable:
            return SEARCH_EMBEDDINGS_ERROR
        matches = await self._store.retrieve(
            self._user_id, retrieval_query, vectors[0], self._k
        )
        return format_matches(matches, query)
