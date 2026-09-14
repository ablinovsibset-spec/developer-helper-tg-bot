"""Telegram-уровень RAG: загрузка документа, команды и e2e-ответ агента.

Всё герметично: Telegram — двойник бота, LLM и эмбеддинги — двойники,
индекс — файл в tmp_path. Ни сети, ни секретов, ни Docker.
"""
from __future__ import annotations

import pytest

from dev_helper_bot import main as main_module
from dev_helper_bot.document_store import DocumentStore, DocumentStoreUnavailable
from dev_helper_bot.main import (
    AGENT_TOOLS,
    DELETE_DONE_TEMPLATE,
    DELETE_NOT_FOUND_TEMPLATE,
    DELETE_USAGE_MESSAGE,
    DOCUMENTS_DISABLED_MESSAGE,
    DOCUMENTS_EMPTY_MESSAGE,
    DOCUMENTS_LIST_HEADER,
    EMBEDDINGS_ERROR_MESSAGE,
    INDEXED_MESSAGE,
    INDEXING_CHUNKS_MESSAGE,
    INDEXING_EMBED_MESSAGE,
    INDEXING_EXTRACT_MESSAGE,
    INDEXING_MESSAGE,
    UNSUPPORTED_DOCUMENT_MESSAGE,
    handle_delete,
    handle_document,
    handle_documents,
    handle_text,
)
from dev_helper_bot.memory import MemoryStore
from dev_helper_bot.skills import Skill
from tests.conftest import (
    BrokenEmbeddingClient,
    FakeCommandExecutor,
    FakeDocument,
    FakeEmbeddingClient,
    FakeMessage,
    assistant_turn,
    make_scripted_llm,
    tool_call,
)
from tests.unit.test_documents import docx_bytes

CHAT_ID = 42
ALICE = 101
BOB = 202
SKILLS = {
    "document-qa": Skill(
        name="document-qa",
        description="Вопросы по загруженным документам",
        body="Указывай источник.",
    )
}

VACATION_POLICY = (
    "Политика отпусков компании. Ежегодный оплачиваемый отпуск составляет "
    "28 календарных дней. Заявление подаётся за две недели до начала."
)
EXPENSE_POLICY = (
    "Политика расходов. Чеки на командировочные траты загружаются "
    "в течение пяти рабочих дней после поездки."
)


@pytest.fixture
def embeddings() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
async def documents(tmp_path, embeddings):
    store = DocumentStore(tmp_path / "rag.db", embeddings.dimension)
    await store.open()
    yield store
    await store.close()


@pytest.fixture
async def memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.open()
    yield store
    await store.close()


def upload(
    filename: str,
    text: str = VACATION_POLICY,
    *,
    user_id: int = ALICE,
    file_size: int | None = None,
) -> FakeMessage:
    return FakeMessage(
        chat_id=CHAT_ID,
        user_id=user_id,
        document=FakeDocument(filename, text.encode("utf-8"), file_size),
    )


async def index_via_handler(
    fake_bot, documents, embeddings, message: FakeMessage
) -> None:
    await handle_document(message, fake_bot, documents, embeddings)
    fake_bot.sent.clear()
    fake_bot.edits.clear()


def last_chat_text(bot) -> str:
    """Последний текст статуса: edit, если был, иначе исходная отправка."""
    if bot.edits:
        return bot.edits[-1]["text"]
    return bot.sent[-1]["text"]


# --- Загрузка документа (задача 4.1) ---


async def test_upload_notifies_start_then_ready(fake_bot, documents, embeddings):
    await handle_document(
        upload("vacation_policy.txt"), fake_bot, documents, embeddings
    )

    assert fake_bot.sent[0] == {
        "chat_id": CHAT_ID,
        "text": INDEXING_MESSAGE.format(filename="vacation_policy.txt"),
    }
    assert len(fake_bot.sent) == 1  # шаги — edit одного сообщения, не пачка
    texts = [edit["text"] for edit in fake_bot.edits]
    assert INDEXING_EXTRACT_MESSAGE.format(filename="vacation_policy.txt") in texts
    assert (
        INDEXING_CHUNKS_MESSAGE.format(filename="vacation_policy.txt", count=1)
        in texts
    )
    assert (
        INDEXING_EMBED_MESSAGE.format(
            filename="vacation_policy.txt", done=1, total=1
        )
        in texts
    )
    assert texts[-1] == INDEXED_MESSAGE.format(
        filename="vacation_policy.txt", count=1
    )
    assert [info.filename for info in await documents.list_documents(ALICE)] == [
        "vacation_policy.txt"
    ]


async def test_upload_indexes_docx(fake_bot, documents, embeddings):
    message = FakeMessage(
        chat_id=CHAT_ID,
        user_id=ALICE,
        document=FakeDocument("handbook.docx", docx_bytes(["Отпуск — 28 дней."])),
    )

    await handle_document(message, fake_bot, documents, embeddings)

    assert "проиндексирован" in last_chat_text(fake_bot)
    assert len(await documents.list_documents(ALICE)) == 1


async def test_upload_unsupported_format_is_rejected_without_indexing(
    fake_bot, documents, embeddings
):
    await handle_document(upload("archive.zip"), fake_bot, documents, embeddings)

    assert fake_bot.sent == [
        {
            "chat_id": CHAT_ID,
            "text": UNSUPPORTED_DOCUMENT_MESSAGE.format(filename="archive.zip"),
        }
    ]
    assert await documents.list_documents(ALICE) == []


async def test_upload_empty_document_reports_error_and_leaves_no_index(
    fake_bot, documents, embeddings
):
    await handle_document(upload("empty.txt", "   \n  "), fake_bot, documents, embeddings)

    assert "Не удалось проиндексировать" in last_chat_text(fake_bot)
    assert await documents.list_documents(ALICE) == []


async def test_upload_corrupted_docx_reports_error(fake_bot, documents, embeddings):
    message = FakeMessage(
        chat_id=CHAT_ID,
        user_id=ALICE,
        document=FakeDocument("broken.docx", b"not a zip"),
    )

    await handle_document(message, fake_bot, documents, embeddings)

    assert "Не удалось проиндексировать" in last_chat_text(fake_bot)
    assert await documents.list_documents(ALICE) == []


async def test_upload_over_size_limit_is_refused_before_download(
    fake_bot, documents, embeddings, monkeypatch
):
    """Лимит сырых байт проверяется до скачивания (design D8): тяжёлой
    работы не происходит вовсе."""
    monkeypatch.setattr(main_module, "rag_max_upload_bytes", lambda: 10)

    await handle_document(
        upload("huge.txt", file_size=5_000_000), fake_bot, documents, embeddings
    )

    assert len(fake_bot.sent) == 1  # уведомления о начале обработки не было
    assert "больше лимита" in fake_bot.sent[0]["text"]
    assert await documents.list_documents(ALICE) == []
    assert embeddings.calls == []


async def test_upload_over_chunk_limit_is_not_indexed(
    fake_bot, documents, embeddings, monkeypatch
):
    monkeypatch.setattr(main_module, "rag_max_chunks_per_doc", lambda: 1)
    text = " ".join(f"пункт{i} политики компании" for i in range(300))

    await handle_document(upload("handbook.md", text), fake_bot, documents, embeddings)

    assert "фрагментов при допустимых" in last_chat_text(fake_bot)
    assert await documents.list_documents(ALICE) == []


async def test_upload_embeddings_outage_reports_and_leaves_no_index(
    fake_bot, documents
):
    await handle_document(
        upload("vacation_policy.txt"), fake_bot, documents, BrokenEmbeddingClient()
    )

    assert last_chat_text(fake_bot) == EMBEDDINGS_ERROR_MESSAGE.format(
        filename="vacation_policy.txt"
    )
    assert await documents.list_documents(ALICE) == []


async def test_upload_download_failure_reports_error(documents, embeddings):
    from tests.conftest import FakeBot

    bot = FakeBot(download_error=RuntimeError("telegram is down"))

    await handle_document(upload("vacation_policy.txt"), bot, documents, embeddings)

    assert "Не удалось скачать" in last_chat_text(bot)
    assert await documents.list_documents(ALICE) == []


async def test_upload_without_store_says_documents_disabled(fake_bot):
    await handle_document(upload("vacation_policy.txt"), fake_bot, None, None)

    assert fake_bot.sent == [
        {"chat_id": CHAT_ID, "text": DOCUMENTS_DISABLED_MESSAGE}
    ]


async def test_reupload_same_name_replaces_index_via_handler(
    fake_bot, documents, embeddings
):
    await index_via_handler(
        fake_bot, documents, embeddings, upload("policy.txt", VACATION_POLICY)
    )

    await handle_document(
        upload("policy.txt", EXPENSE_POLICY), fake_bot, documents, embeddings
    )

    infos = await documents.list_documents(ALICE)
    vectors = await embeddings.embed(["отпуск календарных дней"])
    matches = await documents.search(ALICE, vectors[0], k=10)
    assert [info.filename for info in infos] == ["policy.txt"]
    assert all("28 календарных дней" not in match.text for match in matches)


async def test_upload_does_not_call_llm_or_touch_memory(
    fake_bot, documents, embeddings, memory
):
    """Индексация вне агентного цикла (design D6): переписка не меняется."""
    await handle_document(
        upload("vacation_policy.txt"), fake_bot, documents, embeddings
    )

    assert await memory.load_open_history(CHAT_ID) == []


# --- Команда /documents (задача 4.2) ---


async def test_documents_command_lists_own_files_without_llm(
    fake_bot, documents, embeddings
):
    await index_via_handler(
        fake_bot, documents, embeddings, upload("vacation_policy.txt")
    )
    await index_via_handler(
        fake_bot, documents, embeddings, upload("expenses.md", EXPENSE_POLICY)
    )

    await handle_documents(
        FakeMessage("/documents", chat_id=CHAT_ID, user_id=ALICE), fake_bot, documents
    )

    text = fake_bot.sent[-1]["text"]
    assert text.startswith(DOCUMENTS_LIST_HEADER)
    assert "vacation_policy.txt" in text
    assert "expenses.md" in text


async def test_documents_command_empty_list_message(fake_bot, documents):
    await handle_documents(
        FakeMessage("/documents", chat_id=CHAT_ID, user_id=ALICE), fake_bot, documents
    )

    assert fake_bot.sent == [{"chat_id": CHAT_ID, "text": DOCUMENTS_EMPTY_MESSAGE}]


async def test_documents_command_does_not_show_other_users_files(
    fake_bot, documents, embeddings
):
    await index_via_handler(
        fake_bot, documents, embeddings, upload("vacation_policy.txt", user_id=ALICE)
    )

    await handle_documents(
        FakeMessage("/documents", chat_id=CHAT_ID, user_id=BOB), fake_bot, documents
    )

    assert fake_bot.sent[-1]["text"] == DOCUMENTS_EMPTY_MESSAGE


# --- Команда /delete (задача 4.2) ---


async def test_delete_command_removes_document(fake_bot, documents, embeddings):
    await index_via_handler(
        fake_bot, documents, embeddings, upload("vacation_policy.txt")
    )

    await handle_delete(
        FakeMessage("/delete vacation_policy.txt", chat_id=CHAT_ID, user_id=ALICE),
        fake_bot,
        documents,
    )

    assert fake_bot.sent[-1]["text"] == DELETE_DONE_TEMPLATE.format(
        filename="vacation_policy.txt"
    )
    assert await documents.list_documents(ALICE) == []


async def test_delete_command_unknown_file_reports_miss(fake_bot, documents):
    await handle_delete(
        FakeMessage("/delete нет-такого.pdf", chat_id=CHAT_ID, user_id=ALICE),
        fake_bot,
        documents,
    )

    assert fake_bot.sent[-1]["text"] == DELETE_NOT_FOUND_TEMPLATE.format(
        filename="нет-такого.pdf"
    )


async def test_delete_command_without_argument_shows_usage(fake_bot, documents):
    await handle_delete(
        FakeMessage("/delete", chat_id=CHAT_ID, user_id=ALICE), fake_bot, documents
    )

    assert fake_bot.sent == [{"chat_id": CHAT_ID, "text": DELETE_USAGE_MESSAGE}]


async def test_delete_command_cannot_remove_other_users_document(
    fake_bot, documents, embeddings
):
    await index_via_handler(
        fake_bot, documents, embeddings, upload("policy.txt", user_id=ALICE)
    )

    await handle_delete(
        FakeMessage("/delete policy.txt", chat_id=CHAT_ID, user_id=BOB),
        fake_bot,
        documents,
    )

    assert "не найден" in fake_bot.sent[-1]["text"]
    assert len(await documents.list_documents(ALICE)) == 1


# --- E2E: индекс → вопрос → retrieval → ответ (задача 5.1) ---


def search_documents_call(query: str):
    return assistant_turn(
        content=None,
        tool_calls=[
            tool_call(
                name="search_documents",
                arguments=f'{{"query": "{query}"}}',
            )
        ],
        finish_reason="tool_calls",
    )


async def test_search_documents_is_offered_to_the_model():
    assert any(
        (spec.get("function") or {}).get("name") == "search_documents"
        for spec in AGENT_TOOLS
    )


async def test_e2e_question_retrieves_chunk_and_answer_cites_source(
    fake_bot, documents, embeddings, memory
):
    await index_via_handler(
        fake_bot, documents, embeddings, upload("vacation_policy.txt")
    )
    llm = make_scripted_llm(
        [
            search_documents_call("сколько дней отпуска"),
            assistant_turn(
                content="Отпуск — 28 календарных дней.\nИсточник: vacation_policy.txt"
            ),
        ]
    )

    await handle_text(
        FakeMessage("Сколько дней отпуска?", chat_id=CHAT_ID, user_id=ALICE),
        fake_bot,
        llm,
        memory,
        SKILLS,
        FakeCommandExecutor(),
        documents=documents,
        embeddings=embeddings,
    )

    tool_messages = [m for m in llm.requests[1] if m["role"] == "tool"]
    assert "vacation_policy.txt" in tool_messages[-1]["content"]
    assert "28 календарных дней" in tool_messages[-1]["content"]
    assert "Источник: vacation_policy.txt" in fake_bot.sent[-1]["text"]
    assert await memory.load_open_history(CHAT_ID) == [
        {"role": "user", "content": "Сколько дней отпуска?"},
        {
            "role": "assistant",
            "content": "Отпуск — 28 календарных дней.\nИсточник: vacation_policy.txt",
        },
    ]


async def test_e2e_empty_retrieval_gives_refusal_material_not_facts(
    fake_bot, documents, embeddings, memory
):
    """Нет документов — инструмент прямо говорит, что искать негде,
    и запрещает подменять это общими знаниями (design D11)."""
    llm = make_scripted_llm(
        [
            search_documents_call("ставка налога на Марсе"),
            assistant_turn(content="В загруженных документах этого нет."),
        ]
    )

    await handle_text(
        FakeMessage("Что в документе про Марс?", chat_id=CHAT_ID, user_id=ALICE),
        fake_bot,
        llm,
        memory,
        SKILLS,
        FakeCommandExecutor(),
        documents=documents,
        embeddings=embeddings,
    )

    tool_messages = [m for m in llm.requests[1] if m["role"] == "tool"]
    assert "нет загруженных документов" in tool_messages[-1]["content"]
    assert fake_bot.sent[-1]["text"] == "В загруженных документах этого нет."


async def test_e2e_other_users_document_never_reaches_the_model(
    fake_bot, documents, embeddings, memory
):
    await index_via_handler(
        fake_bot, documents, embeddings, upload("vacation_policy.txt", user_id=ALICE)
    )
    llm = make_scripted_llm(
        [
            search_documents_call("сколько дней отпуска"),
            assistant_turn(content="Не нашёл в ваших документах."),
        ]
    )

    await handle_text(
        FakeMessage("Сколько дней отпуска?", chat_id=CHAT_ID, user_id=BOB),
        fake_bot,
        llm,
        memory,
        SKILLS,
        FakeCommandExecutor(),
        documents=documents,
        embeddings=embeddings,
    )

    transcript = str(llm.requests)
    assert "28 календарных дней" not in transcript
    assert "vacation_policy.txt" not in transcript


async def test_e2e_without_document_store_tool_is_unavailable_not_crashing(
    fake_bot, memory
):
    llm = make_scripted_llm(
        [
            search_documents_call("отпуск"),
            assistant_turn(content="Поиск по документам сейчас не работает."),
        ]
    )

    await handle_text(
        FakeMessage("Что в документе?", chat_id=CHAT_ID, user_id=ALICE),
        fake_bot,
        llm,
        memory,
        SKILLS,
        FakeCommandExecutor(),
    )

    tool_messages = [m for m in llm.requests[1] if m["role"] == "tool"]
    assert "недоступен" in tool_messages[-1]["content"]
    assert fake_bot.sent[-1]["text"] == "Поиск по документам сейчас не работает."


async def test_missing_sqlite_vec_extension_fails_fast_on_open(tmp_path, monkeypatch):
    """RAG обязателен (design D10): не загрузилось расширение — понятная
    ошибка старта, а не бот без поиска по документам."""
    import dev_helper_bot.document_store as store_module

    monkeypatch.setattr(
        store_module.sqlite_vec,
        "loadable_path",
        lambda: str(tmp_path / "no-such-extension"),
    )
    store = DocumentStore(tmp_path / "rag.db", 64)

    with pytest.raises(DocumentStoreUnavailable, match="sqlite-vec"):
        await store.open()


async def test_missing_fts5_fails_fast_on_open(tmp_path, embeddings, monkeypatch):
    import dev_helper_bot.document_store as store_module

    async def no_fts(_db):
        return False

    monkeypatch.setattr(store_module, "fts5_is_available", no_fts)
    store = DocumentStore(tmp_path / "rag.db", embeddings.dimension)

    with pytest.raises(DocumentStoreUnavailable, match="FTS5"):
        await store.open()


async def test_upload_embedding_progress_shows_k_of_n(
    fake_bot, documents, embeddings, monkeypatch
):
    monkeypatch.setattr(main_module, "EMBEDDING_BATCH_SIZE", 2)
    text = " ".join(
        f"пункт{i:03d} политики компании про отпуск и работу сотрудника"
        for i in range(80)
    )

    await handle_document(upload("handbook.md", text), fake_bot, documents, embeddings)

    embed_steps = [
        edit["text"] for edit in fake_bot.edits if "эмбеддинги" in edit["text"]
    ]
    assert embed_steps
    assert any("/" in step for step in embed_steps)
    assert "проиндексирован" in last_chat_text(fake_bot)


async def test_upload_continues_when_status_edit_fails(
    documents, embeddings
):
    from tests.conftest import FakeBot

    bot = FakeBot(edit_error=RuntimeError("telegram flood"))
    await handle_document(upload("vacation_policy.txt"), bot, documents, embeddings)

    assert [info.filename for info in await documents.list_documents(ALICE)] == [
        "vacation_policy.txt"
    ]
    assert any("проиндексирован" in item["text"] for item in bot.sent)


async def test_follow_up_search_uses_session_history(
    fake_bot, documents, embeddings, memory
):
    """Короткий анафорический query с историей находит документ про отпуск."""
    vacation = (
        "Ежегодный отпуск составляет 28 календарных дней. "
        "Неиспользованные дни отпуска переносятся на следующий календарный год."
    )
    hardware = (
        "Администратор может перенести виртуальный сервер в другой датацентр "
        "по заявке на обслуживание инфраструктуры."
    )
    await index_via_handler(
        fake_bot, documents, embeddings, upload("vacation_policy.md", vacation)
    )
    await index_via_handler(
        fake_bot, documents, embeddings, upload("hardware.md", hardware)
    )
    await memory.append_user(CHAT_ID, "Сколько дней ежегодного отпуска положено?")
    await memory.append_assistant(
        CHAT_ID, "28 календарных дней.\nИсточник: vacation_policy.md"
    )
    llm = make_scripted_llm(
        [
            search_documents_call("а можно перенести их?"),
            assistant_turn(
                content="Да, дни отпуска переносятся.\nИсточник: vacation_policy.md"
            ),
        ]
    )

    await handle_text(
        FakeMessage("а можно перенести их?", chat_id=CHAT_ID, user_id=ALICE),
        fake_bot,
        llm,
        memory,
        SKILLS,
        FakeCommandExecutor(),
        documents=documents,
        embeddings=embeddings,
    )

    tool_messages = [m for m in llm.requests[1] if m["role"] == "tool"]
    assert "vacation_policy.md" in tool_messages[-1]["content"]
    assert "переносятся" in tool_messages[-1]["content"]
