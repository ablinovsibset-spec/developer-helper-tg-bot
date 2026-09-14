"""Автоматическая проверка retrieval на evaluation dataset (задача 5.2).

Корпус и вопросы лежат в `eval/` и переживают этот тест: те же файлы можно
прогнать через реальные эмбеддинги. Здесь прогон герметичный — на
детерминированном двойнике эмбеддингов, без сети и ключей.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dev_helper_bot.document_store import DocumentStore
from dev_helper_bot.documents import chunk_text
from tests.conftest import FakeEmbeddingClient

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
DATASET_PATH = EVAL_DIR / "rag_eval_dataset.json"

MIN_QUESTIONS = 5
USER_ID = 4242


def load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


DATASET = load_dataset()
QUESTIONS = DATASET["questions"]
DOCUMENTS_DIR = EVAL_DIR / DATASET["documents_dir"]
TOP_K = DATASET["top_k"]


@pytest.fixture
def embeddings() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
async def indexed_store(tmp_path, embeddings):
    """Весь корпус eval/documents проиндексирован для одного пользователя."""
    store = DocumentStore(tmp_path / "rag.db", embeddings.dimension)
    await store.open()
    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        chunks = chunk_text(path.read_text(encoding="utf-8"))
        vectors = await embeddings.embed(chunks)
        await store.index_document(USER_ID, path.name, chunks, vectors)
    yield store
    await store.close()


def test_dataset_has_at_least_five_questions_with_expected_source():
    assert len(QUESTIONS) >= MIN_QUESTIONS
    for question in QUESTIONS:
        assert question["question"].strip()
        assert question["expected_source"].strip()
        assert (DOCUMENTS_DIR / question["expected_source"]).is_file()


def test_dataset_ids_are_unique():
    ids = [question["id"] for question in QUESTIONS]

    assert len(ids) == len(set(ids))


def test_eval_corpus_is_present_and_indexable():
    documents = sorted(path.name for path in DOCUMENTS_DIR.glob("*.md"))

    assert len(documents) >= 5
    assert all((DOCUMENTS_DIR / name).stat().st_size > 0 for name in documents)


@pytest.mark.parametrize(
    "question", QUESTIONS, ids=[question["id"] for question in QUESTIONS]
)
async def test_expected_source_is_in_top_k(question, indexed_store, embeddings):
    """Для каждого вопроса ожидаемый источник обязан попасть в Top-K."""
    vectors = await embeddings.embed([question["question"]])
    matches = await indexed_store.retrieve(
        USER_ID, question["question"], vectors[0], k=TOP_K
    )

    sources = [match.filename for match in matches]
    assert question["expected_source"] in sources, (
        f"{question['id']}: ожидался {question['expected_source']}, "
        f"Top-{TOP_K} вернул {sources}"
    )


@pytest.mark.parametrize(
    "question",
    [q for q in QUESTIONS if q.get("expected_fragment")],
    ids=[q["id"] for q in QUESTIONS if q.get("expected_fragment")],
)
async def test_expected_fragment_is_retrieved(question, indexed_store, embeddings):
    """Мало вернуть нужный файл — в Top-K должен попасть сам факт."""
    vectors = await embeddings.embed([question["question"]])
    matches = await indexed_store.retrieve(
        USER_ID, question["question"], vectors[0], k=TOP_K
    )

    texts = [
        match.text
        for match in matches
        if match.filename == question["expected_source"]
    ]
    assert any(question["expected_fragment"] in text for text in texts), (
        f"{question['id']}: фрагмент {question['expected_fragment']!r} "
        f"не найден в Top-{TOP_K}"
    )


async def test_eval_retrieval_respects_user_isolation(indexed_store, embeddings):
    """Тот же корпус невидим другому пользователю — изоляция держится
    и на полном наборе документов."""
    vectors = await embeddings.embed([QUESTIONS[0]["question"]])

    assert await indexed_store.retrieve(
        USER_ID + 1, QUESTIONS[0]["question"], vectors[0], k=TOP_K
    ) == []
