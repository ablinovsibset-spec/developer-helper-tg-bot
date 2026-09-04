from __future__ import annotations

import re

import pytest

from dev_helper_bot.memory import (
    LIST_SESSIONS_EMPTY_MARKER,
    LIST_SESSIONS_LIMIT,
    OPEN_HISTORY_LIMIT,
    PREVIEW_CHARS,
    SEARCH_EXCERPT_LIMIT,
    SEARCH_NOT_FOUND_TEMPLATE,
    MemoryStore,
)

CHAT_ID = 42
OTHER_CHAT_ID = 4242

DATE_IN_BRACKETS = re.compile(r"\[\d{4}-\d{2}-\d{2}\]")


@pytest.fixture
async def store(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.open()
    yield store
    await store.close()


async def test_open_creates_db_file(tmp_path):
    db_path = tmp_path / "nested" / "memory.db"

    fresh = MemoryStore(db_path)
    assert not db_path.exists()
    await fresh.open()
    try:
        assert db_path.exists()
        # Схема применена: запись и чтение работают сразу
        await fresh.append_user(CHAT_ID, "привет")
        assert await fresh.load_open_history(CHAT_ID) == [
            {"role": "user", "content": "привет"}
        ]
    finally:
        await fresh.close()


async def test_messages_accumulate_in_single_open_session(store):
    await store.append_user(CHAT_ID, "первое")
    await store.append_assistant(CHAT_ID, "ответ на первое")
    await store.append_user(CHAT_ID, "второе")
    await store.append_assistant(CHAT_ID, "ответ на второе")

    history = await store.load_open_history(CHAT_ID)

    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "первое"),
        ("assistant", "ответ на первое"),
        ("user", "второе"),
        ("assistant", "ответ на второе"),
    ]


async def test_new_command_closes_session_and_next_opens_new_one(store):
    await store.append_user(CHAT_ID, "старая тема")
    await store.append_assistant(CHAT_ID, "старый ответ")

    await store.close_session(CHAT_ID)

    await store.append_user(CHAT_ID, "новая тема")
    history = await store.load_open_history(CHAT_ID)

    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "новая тема"),
    ]
    # Прежняя сессия осталась в хранилище завершённой и доступна поиску
    assert "старая тема" in await store.search_completed(CHAT_ID, "старая тема")


async def test_close_session_without_open_session_is_noop(store):
    await store.close_session(CHAT_ID)

    await store.append_user(CHAT_ID, "после пустого закрытия")
    assert await store.load_open_history(CHAT_ID) == [
        {"role": "user", "content": "после пустого закрытия"}
    ]


async def test_history_survives_close_and_reopen(tmp_path):
    """Имитация рестарта: новый store на том же файле видит открытую сессию."""
    db_path = tmp_path / "memory.db"
    first = MemoryStore(db_path)
    await first.open()
    await first.append_user(CHAT_ID, "вопрос через рестарт")
    await first.append_assistant(CHAT_ID, "ответ через рестарт")
    await first.close()

    second = MemoryStore(db_path)
    await second.open()
    try:
        assert await second.load_open_history(CHAT_ID) == [
            {"role": "user", "content": "вопрос через рестарт"},
            {"role": "assistant", "content": "ответ через рестарт"},
        ]
    finally:
        await second.close()


async def test_open_history_returns_tail_up_to_limit(store):
    for i in range(OPEN_HISTORY_LIMIT + 10):
        await store.append_user(CHAT_ID, f"сообщение {i:03d}")

    history = await store.load_open_history(CHAT_ID)

    assert len(history) == OPEN_HISTORY_LIMIT
    assert history[0]["content"] == f"сообщение {10:03d}"
    assert history[-1]["content"] == f"сообщение {OPEN_HISTORY_LIMIT + 10 - 1:03d}"


async def test_chats_do_not_see_each_others_history(store):
    await store.append_user(CHAT_ID, "секрет первого чата")
    await store.append_user(OTHER_CHAT_ID, "секрет второго чата")

    assert await store.load_open_history(CHAT_ID) == [
        {"role": "user", "content": "секрет первого чата"}
    ]
    assert await store.load_open_history(OTHER_CHAT_ID) == [
        {"role": "user", "content": "секрет второго чата"}
    ]


async def test_search_completed_finds_completed_session_with_date(store):
    await store.append_user(CHAT_ID, "обсудили деплой базы")
    await store.append_assistant(CHAT_ID, "итог: используем sqlite")
    await store.close_session(CHAT_ID)

    result = await store.search_completed(CHAT_ID, "деплой")

    assert "Найдено выдержек" in result
    assert "обсудили деплой базы" in result
    assert "user" in result
    assert DATE_IN_BRACKETS.search(result)


async def test_search_completed_not_found_returns_explicit_marker(store):
    await store.append_user(CHAT_ID, "разговор о погоде")
    await store.close_session(CHAT_ID)

    result = await store.search_completed(CHAT_ID, "квантовая физика")

    assert result == SEARCH_NOT_FOUND_TEMPLATE.format(query="квантовая физика")


async def test_search_completed_skips_open_session(store):
    await store.append_user(CHAT_ID, "открытая сессия про деплой")
    await store.append_assistant(CHAT_ID, "итог открытой сессии")

    result = await store.search_completed(CHAT_ID, "деплой")

    assert result == SEARCH_NOT_FOUND_TEMPLATE.format(query="деплой")


async def test_search_completed_is_limited_to_ten_excerpts(store):
    await store.append_user(CHAT_ID, "начало")
    for i in range(SEARCH_EXCERPT_LIMIT + 2):
        await store.append_user(CHAT_ID, f"запись маркер {i:02d}")
    await store.close_session(CHAT_ID)

    result = await store.search_completed(CHAT_ID, "маркер")

    assert result.count("маркер") == SEARCH_EXCERPT_LIMIT


async def test_search_completed_isolates_chats(store):
    await store.append_user(OTHER_CHAT_ID, "чужой маркер деплоя")
    await store.close_session(OTHER_CHAT_ID)

    result = await store.search_completed(CHAT_ID, "маркер")

    assert result == SEARCH_NOT_FOUND_TEMPLATE.format(query="маркер")


async def test_search_completed_escapes_like_wildcards(store):
    await store.append_user(CHAT_ID, "сто процентов_точно")
    await store.close_session(CHAT_ID)

    wildcard_miss = await store.search_completed(CHAT_ID, "%")
    assert wildcard_miss == SEARCH_NOT_FOUND_TEMPLATE.format(query="%")

    literal_hit = await store.search_completed(CHAT_ID, "процентов_точно")
    assert "сто процентов_точно" in literal_hit


async def test_search_completed_excerpt_is_windowed(store):
    filler = "я" * 1000
    content = f"{filler}цитируемый фрагмент{filler}"
    await store.append_user(CHAT_ID, content)
    await store.close_session(CHAT_ID)

    result = await store.search_completed(CHAT_ID, "цитируемый фрагмент")

    assert "…" in result
    excerpt = result.split(": ", 1)[1]
    assert len(excerpt) < len(content)
    assert "цитируемый фрагмент" in excerpt


async def test_list_completed_sessions_shows_date_count_and_preview(store):
    await store.append_user(CHAT_ID, "беседа про деплой базы")
    await store.append_assistant(CHAT_ID, "итог: используем sqlite")
    await store.close_session(CHAT_ID)

    result = await store.list_completed_sessions(CHAT_ID)

    assert "Завершённые беседы" in result
    assert DATE_IN_BRACKETS.search(result)
    assert "сообщений: 2" in result
    assert "начало: беседа про деплой базы" in result
    # Строго читающая операция: открытая сессия не создалась
    assert await store.load_open_history(CHAT_ID) == []


async def test_list_completed_sessions_skips_open_session(store):
    await store.append_user(CHAT_ID, "завершённая тема")
    await store.close_session(CHAT_ID)
    await store.append_user(CHAT_ID, "тема открытой сессии")

    result = await store.list_completed_sessions(CHAT_ID)

    assert "завершённая тема" in result
    assert "тема открытой сессии" not in result


async def test_list_completed_sessions_limited_to_last_ten(store):
    for i in range(LIST_SESSIONS_LIMIT + 5):
        await store.append_user(CHAT_ID, f"тема номер {i:02d}")
        await store.close_session(CHAT_ID)

    result = await store.list_completed_sessions(CHAT_ID)

    assert result.count("тема номер") == LIST_SESSIONS_LIMIT
    assert "тема номер 04" not in result  # старые выпали из обзора
    assert "тема номер 05" in result  # последние 10 остаются


async def test_list_completed_sessions_preview_is_truncated(store):
    long_text = "а" * (PREVIEW_CHARS + 50)
    await store.append_user(CHAT_ID, long_text)
    await store.close_session(CHAT_ID)

    result = await store.list_completed_sessions(CHAT_ID)

    assert long_text not in result
    assert "а" * PREVIEW_CHARS in result
    assert result.rstrip().endswith("…")


async def test_list_completed_sessions_empty_returns_explicit_marker(store):
    assert await store.list_completed_sessions(CHAT_ID) == LIST_SESSIONS_EMPTY_MARKER


async def test_list_completed_sessions_isolates_chats(store):
    await store.append_user(OTHER_CHAT_ID, "чужая завершённая тема")
    await store.close_session(OTHER_CHAT_ID)

    assert await store.list_completed_sessions(CHAT_ID) == LIST_SESSIONS_EMPTY_MARKER
