from __future__ import annotations

import pytest

from dev_helper_bot.document_store import (
    SEARCH_EMBEDDINGS_ERROR,
    SEARCH_NO_DOCUMENTS_MESSAGE,
    ChunkMatch,
    DocumentStore,
    DocumentStoreUnavailable,
    UserDocumentSearcher,
    format_matches,
    page_label,
    sanitize_fts_query,
)
from dev_helper_bot.documents import chunk_text
from tests.conftest import (
    FAKE_EMBEDDING_DIM,
    BrokenEmbeddingClient,
    FakeEmbeddingClient,
)

ALICE = 101
BOB = 202

VACATION_POLICY = (
    "Политика отпусков. Ежегодный оплачиваемый отпуск составляет "
    "28 календарных дней. Заявление на отпуск подаётся за две недели."
)
EXPENSE_POLICY = (
    "Политика возмещения расходов. Чеки на командировочные траты "
    "загружаются в систему в течение пяти рабочих дней после поездки."
)
ONBOARDING = (
    "Онбординг разработчика. Доступ к репозиториям выдаёт тимлид "
    "в первый рабочий день, ноутбук — служба поддержки."
)


@pytest.fixture
def embeddings() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
async def store(tmp_path, embeddings):
    store = DocumentStore(tmp_path / "rag.db", embeddings.dimension)
    await store.open()
    yield store
    await store.close()


async def index(store, embeddings, user_id: int, filename: str, text: str) -> int:
    """Индексация текста как документа — путь handler'а без Telegram."""
    chunks = chunk_text(text)
    vectors = await embeddings.embed(chunks)
    return await store.index_document(user_id, filename, chunks, vectors)


async def search_filenames(store, embeddings, user_id: int, query: str) -> list[str]:
    vectors = await embeddings.embed([query])
    matches = await store.search(user_id, vectors[0], k=5)
    return [match.filename for match in matches]


# --- Схема и открытие (задача 3.3) ---


async def test_open_creates_db_file_with_vec_table(tmp_path, embeddings):
    path = tmp_path / "nested" / "rag.db"
    store = DocumentStore(path, embeddings.dimension)

    await store.open()
    try:
        assert path.exists()  # каталог создан вместе с файлом
        assert await store.list_documents(ALICE) == []
    finally:
        await store.close()


async def test_open_rejects_index_built_for_other_dimension(tmp_path):
    """Смена модели эмбеддингов ловится на старте понятной ошибкой, а не
    невнятным сбоем вставки (design: смена модели = wipe rag.db)."""
    path = tmp_path / "rag.db"
    first = DocumentStore(path, 64)
    await first.open()
    await first.close()

    second = DocumentStore(path, 128)

    with pytest.raises(DocumentStoreUnavailable, match="размерность"):
        await second.open()


async def test_index_document_rejects_wrong_vector_dimension(store):
    with pytest.raises(ValueError, match="dimension"):
        await store.index_document(ALICE, "p.txt", ["текст"], [[0.1, 0.2]])


async def test_search_rejects_wrong_query_dimension(store):
    with pytest.raises(ValueError, match="dimension"):
        await store.search(ALICE, [0.1, 0.2])


# --- Retrieval (задача 3.4) ---


async def test_relevant_query_finds_expected_document(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)
    await index(store, embeddings, ALICE, "onboarding.md", ONBOARDING)

    found = await search_filenames(
        store, embeddings, ALICE, "сколько календарных дней ежегодный отпуск"
    )

    assert found[0] == "vacation_policy.txt"


async def test_search_returns_chunk_text_and_source(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)

    vectors = await embeddings.embed(["заявление на отпуск"])
    matches = await store.search(ALICE, vectors[0], k=5)

    assert matches
    assert matches[0].filename == "vacation_policy.txt"
    assert matches[0].chunk_index == 0
    assert "28 календарных дней" in matches[0].text
    assert matches[0].distance >= 0.0


async def test_search_top_k_limits_result_count(store, embeddings):
    long_text = " ".join(
        f"Пункт {i}: правила компании о работе и отпуске сотрудника." for i in range(40)
    )
    await index(store, embeddings, ALICE, "handbook.md", long_text)
    assert len(chunk_text(long_text)) > 3

    vectors = await embeddings.embed(["правила работы"])

    assert len(await store.search(ALICE, vectors[0], k=3)) == 3


async def test_search_without_documents_returns_nothing(store, embeddings):
    vectors = await embeddings.embed(["любой вопрос"])

    assert await store.search(ALICE, vectors[0], k=5) == []


# --- Изоляция пользователей (задача 3.4, spec: изоляция по владельцу) ---


async def test_other_users_document_is_never_returned(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)

    bob_results = await search_filenames(
        store, embeddings, BOB, "сколько календарных дней ежегодный отпуск"
    )

    assert bob_results == []
    assert await store.list_documents(BOB) == []


async def test_each_user_sees_only_own_documents(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)
    await index(store, embeddings, BOB, "expenses.md", EXPENSE_POLICY)

    alice_hits = await search_filenames(
        store, embeddings, ALICE, "чеки командировочные расходы"
    )
    bob_hits = await search_filenames(
        store, embeddings, BOB, "чеки командировочные расходы"
    )

    # Запрос про расходы релевантен документу Боба, но Алисе он не виден.
    assert alice_hits == ["vacation_policy.txt"]
    assert bob_hits == ["expenses.md"]
    assert [info.filename for info in await store.list_documents(ALICE)] == [
        "vacation_policy.txt"
    ]


async def test_same_filename_of_two_users_are_separate_documents(store, embeddings):
    await index(store, embeddings, ALICE, "policy.txt", VACATION_POLICY)
    await index(store, embeddings, BOB, "policy.txt", EXPENSE_POLICY)

    vectors = await embeddings.embed(["отпуск календарных дней"])
    alice_match = (await store.search(ALICE, vectors[0], k=1))[0]
    bob_match = (await store.search(BOB, vectors[0], k=1))[0]

    assert "28 календарных дней" in alice_match.text
    assert "Чеки" in bob_match.text


# --- Несколько документов, replace, удаление (задача 3.4) ---


async def test_multiple_documents_are_both_searchable(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)
    await index(store, embeddings, ALICE, "expenses.md", EXPENSE_POLICY)

    assert [info.filename for info in await store.list_documents(ALICE)] == [
        "expenses.md",
        "vacation_policy.txt",
    ]
    assert await search_filenames(
        store, embeddings, ALICE, "ежегодный отпуск дней"
    ) != []
    assert (
        await search_filenames(store, embeddings, ALICE, "чеки после поездки")
    )[0] == "expenses.md"


async def test_reupload_same_filename_replaces_previous_version(store, embeddings):
    await index(store, embeddings, ALICE, "policy.txt", VACATION_POLICY)
    await index(store, embeddings, ALICE, "policy.txt", EXPENSE_POLICY)

    infos = await store.list_documents(ALICE)
    vectors = await embeddings.embed(["отпуск календарных дней"])
    matches = await store.search(ALICE, vectors[0], k=10)

    # Имя осталось одно, а прошлый текст и его векторы исчезли из индекса.
    assert [info.filename for info in infos] == ["policy.txt"]
    assert matches
    assert all("28 календарных дней" not in match.text for match in matches)
    assert all("Чеки" in match.text for match in matches)


async def test_delete_by_filename_removes_document_from_retrieval(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)
    await index(store, embeddings, ALICE, "expenses.md", EXPENSE_POLICY)

    assert await store.delete_by_filename(ALICE, "vacation_policy.txt") is True

    assert [info.filename for info in await store.list_documents(ALICE)] == [
        "expenses.md"
    ]
    assert await search_filenames(
        store, embeddings, ALICE, "ежегодный отпуск календарных дней"
    ) == ["expenses.md"]


async def test_delete_unknown_filename_reports_miss_and_keeps_index(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)

    assert await store.delete_by_filename(ALICE, "нет-такого.pdf") is False

    assert len(await store.list_documents(ALICE)) == 1


async def test_delete_does_not_touch_other_users_document(store, embeddings):
    await index(store, embeddings, ALICE, "policy.txt", VACATION_POLICY)
    await index(store, embeddings, BOB, "policy.txt", EXPENSE_POLICY)

    assert await store.delete_by_filename(ALICE, "policy.txt") is True

    assert await store.list_documents(ALICE) == []
    assert len(await store.list_documents(BOB)) == 1


async def test_list_documents_reports_chunk_count(store, embeddings):
    count = await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)

    infos = await store.list_documents(ALICE)

    assert infos[0].chunk_count == count
    assert infos[0].created_at


# --- Персистентность (spec: индекс переживает рестарт) ---


async def test_index_survives_store_reopen(tmp_path, embeddings):
    """«Рестарт» бота — новый store на том же файле (как у memory)."""
    path = tmp_path / "rag.db"
    first = DocumentStore(path, embeddings.dimension)
    await first.open()
    await index(first, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)
    await first.close()

    second = DocumentStore(path, embeddings.dimension)
    await second.open()
    try:
        assert [info.filename for info in await second.list_documents(ALICE)] == [
            "vacation_policy.txt"
        ]
        assert await search_filenames(
            second, embeddings, ALICE, "ежегодный отпуск календарных дней"
        ) == ["vacation_policy.txt"]
    finally:
        await second.close()


# --- Результат инструмента: attribution и отказ (задача 4.3, design D11) ---


async def test_searcher_result_carries_filename_for_attribution(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.pdf", VACATION_POLICY)
    searcher = UserDocumentSearcher(store, embeddings, ALICE)

    result = await searcher.search("сколько дней отпуска")

    assert "vacation_policy.pdf" in result
    assert "28 календарных дней" in result
    assert "Найдено фрагментов:" in result


async def test_searcher_without_documents_tells_there_is_nothing(store, embeddings):
    searcher = UserDocumentSearcher(store, embeddings, ALICE)

    assert await searcher.search("что угодно") == SEARCH_NO_DOCUMENTS_MESSAGE


async def test_searcher_empty_result_forbids_inventing(store, embeddings):
    """Пустой Top-K возвращает прямой запрет выдумывать (design D11)."""
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)
    searcher = UserDocumentSearcher(store, embeddings, ALICE, k=0)

    # k=0 нормализуется в 1, поэтому пустоту проверяем на самом форматере
    assert "Не отвечай общими знаниями" in format_matches([], "тариф на Марсе")
    assert (await searcher.search("отпуск")).startswith("Найдено фрагментов:")


async def test_searcher_reports_embeddings_outage_as_text(store):
    await index(store, FakeEmbeddingClient(), ALICE, "p.txt", VACATION_POLICY)
    searcher = UserDocumentSearcher(store, BrokenEmbeddingClient(), ALICE)

    # Недоступность эмбеддингов не должна ронять агентный цикл
    assert await searcher.search("отпуск") == SEARCH_EMBEDDINGS_ERROR


async def test_searcher_isolation_holds_through_tool_seam(store, embeddings):
    await index(store, embeddings, ALICE, "vacation_policy.txt", VACATION_POLICY)
    bob_searcher = UserDocumentSearcher(store, embeddings, BOB)

    result = await bob_searcher.search("сколько календарных дней отпуска")

    assert "vacation_policy.txt" not in result
    assert "28 календарных дней" not in result


def test_fake_embeddings_are_deterministic_and_fixed_width():
    client = FakeEmbeddingClient()

    assert client.dimension == FAKE_EMBEDDING_DIM
    first = client._vector("одинаковый текст")
    second = client._vector("одинаковый текст")
    assert first == second
    assert len(first) == FAKE_EMBEDDING_DIM


async def test_open_rejects_missing_fts5(tmp_path, embeddings, monkeypatch):
    import dev_helper_bot.document_store as store_module

    async def no_fts(_db):
        return False

    monkeypatch.setattr(store_module, "fts5_is_available", no_fts)
    store = DocumentStore(tmp_path / "rag.db", embeddings.dimension)

    with pytest.raises(DocumentStoreUnavailable, match="FTS5"):
        await store.open()


async def test_open_migrates_legacy_chunks_and_fills_fts(tmp_path, embeddings):
    """Старый rag.db без page_* и FTS открывается, FTS заполняется из текста."""
    import sqlean
    import sqlite_vec

    path = tmp_path / "rag.db"
    conn = sqlean.connect(str(path))
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, filename)
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id),
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE chunk_vectors USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding float[{embeddings.dimension}] "
        "distance_metric=cosine, user_id INTEGER PARTITION KEY)"
    )
    conn.execute(
        "INSERT INTO documents (user_id, filename, file_type, created_at) "
        "VALUES (?, ?, ?, ?)",
        (ALICE, "legacy.txt", ".txt", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO chunks (document_id, chunk_index, text) VALUES (1, 0, ?)",
        ("уникальныйТокенQX9917 в старом индексе",),
    )
    conn.commit()
    conn.close()

    store = DocumentStore(path, embeddings.dimension)
    await store.open()
    try:
        cursor = await store._conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "page_start" in columns and "page_end" in columns
        hits = await store.search_text(ALICE, "уникальныйТокенQX9917", k=5)
        assert hits
        assert hits[0].filename == "legacy.txt"
    finally:
        await store.close()


async def test_fts_is_isolated_by_owner(store, embeddings):
    await index(
        store,
        embeddings,
        ALICE,
        "secret.txt",
        "уникальныйТокенQX9917 только в документе Алисы",
    )
    await index(store, embeddings, BOB, "other.txt", "обычный текст без того токена")

    bob_fts = await store.search_text(BOB, "уникальныйТокенQX9917", k=5)
    assert bob_fts == []
    query_vec = (await embeddings.embed(["уникальныйТокенQX9917"]))[0]
    bob_hybrid = await store.retrieve(BOB, "уникальныйТокенQX9917", query_vec, k=5)
    assert all(match.filename != "secret.txt" for match in bob_hybrid)
    alice_fts = await store.search_text(ALICE, "уникальныйТокенQX9917", k=5)
    assert alice_fts[0].filename == "secret.txt"


async def test_fts_special_characters_do_not_break_search(store, embeddings):
    await index(store, embeddings, ALICE, "policy.txt", VACATION_POLICY)

    assert sanitize_fts_query("*** (((") == ""
    assert await store.search_text(ALICE, 'foo " OR bar', k=5) is not None
    assert await store.search_text(ALICE, "***", k=5) == []


async def test_retrieve_reranks_candidates_down_to_top_k(store, embeddings):
    long_text = " ".join(
        f"Пункт {i}: правила компании о работе и отпуске сотрудника." for i in range(40)
    )
    await index(store, embeddings, ALICE, "handbook.md", long_text)
    assert len(chunk_text(long_text)) > 3
    query_vec = (await embeddings.embed(["правила работы"]))[0]

    matches = await store.retrieve(ALICE, "правила работы", query_vec, k=3)

    assert 1 <= len(matches) <= 3


async def test_retrieve_not_vec_only_when_fts_hits(store, embeddings, monkeypatch):
    """Точный редкий токен добирается текстовым каналом, даже если vec пуст."""
    await index(
        store,
        embeddings,
        ALICE,
        "codes.md",
        "Код ошибки QX-9917 означает отказ в доступе к репозиторию.",
    )

    async def empty_vec(user_id, query_vector, k=5):
        return []

    monkeypatch.setattr(store, "search", empty_vec)
    query_vec = (await embeddings.embed(["QX-9917"]))[0]
    matches = await store.retrieve(ALICE, "QX-9917", query_vec, k=5)

    assert matches
    assert "QX-9917" in matches[0].text


async def test_searcher_uses_conversation_context_for_follow_up(store, embeddings):
    vacation = (
        "Ежегодный отпуск 28 календарных дней. Неиспользованные дни отпуска "
        "переносятся на следующий календарный год, но не более 10 дней."
    )
    hardware = (
        "Администратор может перенести виртуальный сервер в другой датацентр "
        "по заявке на обслуживание инфраструктуры."
    )
    await index(store, embeddings, ALICE, "vacation_policy.md", vacation)
    await index(store, embeddings, ALICE, "hardware.md", hardware)
    short = "а можно перенести их?"
    with_history = UserDocumentSearcher(
        store,
        embeddings,
        ALICE,
        conversation_context="Сколько дней ежегодного отпуска положено сотруднику?",
    )

    result = await with_history.search(short)

    assert "vacation_policy.md" in result
    assert "переносятся" in result


def test_format_matches_includes_pdf_page_or_range_but_not_for_md():
    one_page = ChunkMatch(
        filename="company_policy.pdf",
        chunk_index=11,
        text="правило",
        distance=0.1,
        page_start=17,
        page_end=17,
    )
    spanning = ChunkMatch(
        filename="company_policy.pdf",
        chunk_index=12,
        text="стык",
        distance=0.2,
        page_start=16,
        page_end=17,
    )
    markdown = ChunkMatch(
        filename="notes.md",
        chunk_index=0,
        text="заметка",
        distance=0.3,
    )

    one = format_matches([one_page], "q")
    span = format_matches([spanning], "q")
    md = format_matches([markdown], "q")

    assert "стр. 17" in one
    assert "фрагмент 12" in one
    assert "стр. 16–17" in span
    assert "notes.md" in md
    assert "стр." not in md
    assert page_label(markdown) is None
