"""Персистентное хранилище переписки на SQLite (change add-conversation-memory).

Сессии чатов и user/финальные assistant-сообщения хранятся в файле БД
на VM-локальном диске сандбокса (design D2) и переживают рестарты бота
и паузы сандбокса. Инструментальный транскрипт в БД не пишется (design D4):
агентный цикл работает с эпемерным списком сообщений, БД — единственный
источник истины контекста. Доступ — aiosqlite, WAL (design D1–D3).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from dev_helper_bot.llm import Message

OPEN_HISTORY_LIMIT = 50
"""Хвост открытой сессии для восстановления контекста (design D8)."""

SEARCH_EXCERPT_LIMIT = 10
SEARCH_EXCERPT_CHARS = 300

SEARCH_NOT_FOUND_TEMPLATE = (
    "По запросу {query!r} ничего не найдено "
    "в завершённых беседах этого чата."
)

LIST_SESSIONS_LIMIT = 10
PREVIEW_CHARS = 120

LIST_SESSIONS_EMPTY_MARKER = "Завершённых бесед в этом чате нет"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_chat ON sessions(chat_id, ended_at);
"""

_SEARCH_SQL = (
    "SELECT m.role, m.content, s.started_at "
    "FROM messages m JOIN sessions s ON m.session_id = s.id "
    "WHERE s.chat_id = ? AND s.ended_at IS NOT NULL "
    "AND m.content LIKE ('%' || ? || '%') ESCAPE '\\' "
    "ORDER BY s.started_at, m.id "
    "LIMIT ?"
)

_LIST_SESSIONS_SQL = (
    "SELECT s.started_at, COUNT(m.id), fm.content "
    "FROM sessions s "
    "LEFT JOIN messages m ON m.session_id = s.id "
    "LEFT JOIN messages fm ON fm.id = ("
    "SELECT MIN(f.id) FROM messages f "
    "WHERE f.session_id = s.id AND f.role = 'user'"
    ") "
    "WHERE s.chat_id = ? AND s.ended_at IS NOT NULL "
    "GROUP BY s.id "
    "ORDER BY s.started_at DESC, s.id DESC "
    "LIMIT ?"
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _escape_like(query: str) -> str:
    """Экранирует спецсимволы LIKE, чтобы запрос искался буквально."""
    return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _excerpt(content: str, query: str, chars: int = SEARCH_EXCERPT_CHARS) -> str:
    """Выдержка ~chars символов вокруг первого вхождения запроса."""
    pos = content.find(query)
    if pos < 0:
        # Совпадение могло быть регистронезависимым (ASCII) — берём начало.
        start = 0
    else:
        start = max(0, pos - (chars - min(len(query), chars)) // 2)
    end = start + chars
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{content[start:end]}{suffix}"


def _preview(content: str, chars: int = PREVIEW_CHARS) -> str:
    """Превью начала сообщения: до chars символов, длиннее — с маркером."""
    if len(content) <= chars:
        return content
    return content[:chars] + "…"


class MemoryStore:
    """Хранилище переписки: сессии + user/assistant-сообщения чатов.

    Жизненным циклом владеет main (паттерн песочницы, design D7): open при
    старте, close в finally. Отсутствующий файл создаётся при open.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        """Открывает/создаёт файл БД, включает WAL, создаёт схему."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        db, self._db = self._db, None
        if db is not None:
            await db.close()

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("MemoryStore не открыт: сначала вызовите open()")
        return self._db

    async def _open_session_id(self, chat_id: int) -> int:
        """id открытой сессии чата; отсутствие сессии — новая (design D5)."""
        cursor = await self._conn.execute(
            "SELECT id FROM sessions WHERE chat_id = ? AND ended_at IS NULL",
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row is not None:
            return row[0]
        cursor = await self._conn.execute(
            "INSERT INTO sessions (chat_id, started_at) VALUES (?, ?)",
            (chat_id, _utcnow_iso()),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def _append(self, chat_id: int, role: str, content: str) -> None:
        session_id = await self._open_session_id(chat_id)
        await self._conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, _utcnow_iso()),
        )
        await self._conn.commit()

    async def append_user(self, chat_id: int, content: str) -> None:
        """Пишет сообщение пользователя в открытую сессию чата (design D4)."""
        await self._append(chat_id, "user", content)

    async def append_assistant(self, chat_id: int, content: str) -> None:
        """Пишет финальный ответ ассистента в открытую сессию чата."""
        await self._append(chat_id, "assistant", content)

    async def close_session(self, chat_id: int) -> None:
        """Закрывает открытую сессию чата — команда /new (design D5)."""
        await self._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE chat_id = ? AND ended_at IS NULL",
            (_utcnow_iso(), chat_id),
        )
        await self._conn.commit()

    async def load_open_history(self, chat_id: int) -> list[Message]:
        """Хвост открытой сессии чата: до OPEN_HISTORY_LIMIT сообщений (design D8)."""
        cursor = await self._conn.execute(
            "SELECT m.role, m.content FROM messages m "
            "JOIN sessions s ON m.session_id = s.id "
            "WHERE s.chat_id = ? AND s.ended_at IS NULL "
            "ORDER BY m.id DESC LIMIT ?",
            (chat_id, OPEN_HISTORY_LIMIT),
        )
        rows = await cursor.fetchall()
        return [
            {"role": role, "content": content}
            for role, content in reversed(rows)
        ]

    async def search_completed(self, chat_id: int, query: str) -> str:
        """LIKE-поиск по завершённым сессиям чата (design D6).

        До SEARCH_EXCERPT_LIMIT выдержек по ~SEARCH_EXCERPT_CHARS символов,
        каждая с датой сессии; пустой результат — явный маркер. Строго читающая
        операция: история и состояние сессий не меняются.
        """
        cursor = await self._conn.execute(
            _SEARCH_SQL,
            (chat_id, _escape_like(query), SEARCH_EXCERPT_LIMIT),
        )
        rows = await cursor.fetchall()
        if not rows:
            return SEARCH_NOT_FOUND_TEMPLATE.format(query=query)
        lines = [f"Найдено выдержек в завершённых беседах: {len(rows)}"]
        for role, content, started_at in rows:
            lines.append(f"— [{started_at[:10]}] {role}: {_excerpt(content, query)}")
        return "\n".join(lines)

    async def list_completed_sessions(self, chat_id: int) -> str:
        """Обзор завершённых сессий чата (design D1/D5).

        Последние LIST_SESSIONS_LIMIT сессий по времени начала, одним
        SQL-запросом: дата начала, число сообщений и превью первого
        user-сообщения. Открытая сессия не попадает; пустой результат —
        явный маркер. Строго читающая операция.
        """
        cursor = await self._conn.execute(
            _LIST_SESSIONS_SQL, (chat_id, LIST_SESSIONS_LIMIT)
        )
        rows = await cursor.fetchall()
        if not rows:
            return LIST_SESSIONS_EMPTY_MARKER
        lines = [f"Завершённые беседы этого чата: {len(rows)}"]
        for started_at, message_count, first_user in rows:
            preview = (
                _preview(first_user) if first_user else "(нет сообщений пользователя)"
            )
            lines.append(
                f"— [{started_at[:10]}] сообщений: {message_count}; начало: {preview}"
            )
        return "\n".join(lines)


class ChatHistorySearcher:
    """Инструменты доступа к прошлым беседам конкретного чата:
    обёртка MemoryStore для search_history и list_sessions.

    Реализует шов HistorySearcher (tools.py), привязывая операции
    к chat_id обрабатываемого сообщения.
    """

    def __init__(self, store: MemoryStore, chat_id: int) -> None:
        self._store = store
        self._chat_id = chat_id

    async def search(self, query: str) -> str:
        return await self._store.search_completed(self._chat_id, query)

    async def list_sessions(self) -> str:
        return await self._store.list_completed_sessions(self._chat_id)
