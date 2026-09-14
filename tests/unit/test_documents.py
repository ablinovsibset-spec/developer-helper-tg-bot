from __future__ import annotations

import io
import zipfile

import pytest

from dev_helper_bot.documents import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
    DocumentTooLarge,
    EmptyDocumentError,
    UnsupportedDocumentType,
    chunk_text,
    document_extension,
    extract_text,
    is_supported_document,
    split_document,
)

POLICY_TEXT = (
    "Отпуск сотрудника — 28 календарных дней в год.\n"
    "Заявление подаётся за две недели до начала отпуска."
)


def docx_bytes(paragraphs: list[str]) -> bytes:
    """Минимальный валидный .docx: OOXML-архив с одним document.xml.

    Собирается вручную, чтобы фикстура не зависела от записи файлов
    сторонней библиотекой — читается она настоящим python-docx.
    """
    body = "".join(
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


# --- Формат и извлечение текста (задача 3.1) ---


def test_supported_extensions_are_the_documented_four():
    assert SUPPORTED_EXTENSIONS == (".txt", ".md", ".docx", ".pdf")


@pytest.mark.parametrize(
    "filename, supported",
    [
        ("policy.txt", True),
        ("notes.md", True),
        ("handbook.DOCX", True),  # регистр расширения не важен
        ("report.pdf", True),
        ("archive.zip", False),
        ("photo.jpeg", False),
        ("Makefile", False),
        (None, False),
    ],
)
def test_is_supported_document_filters_by_extension(filename, supported):
    assert is_supported_document(filename) is supported


def test_extract_text_reads_txt_and_md_as_utf8():
    assert extract_text("policy.txt", POLICY_TEXT.encode("utf-8")) == POLICY_TEXT
    assert extract_text("notes.md", b"# ## \xd0\xa2\xd0\xb5\xd0\xba\xd1\x81\xd1\x82") == (
        "# ## Текст"
    )


def test_extract_text_broken_encoding_is_replaced_not_rejected():
    """«Почти UTF-8» файл индексируется: битые байты заменяются."""
    text = extract_text("policy.txt", "Отпуск".encode("utf-8") + b"\xff\xfe")

    assert text.startswith("Отпуск")


def test_extract_text_docx_paragraphs():
    data = docx_bytes(["Отпуск — 28 дней.", "", "Заявление за две недели."])

    text = extract_text("handbook.docx", data)

    assert "Отпуск — 28 дней." in text
    assert "Заявление за две недели." in text


def test_extract_text_unsupported_extension_names_alternatives():
    with pytest.raises(UnsupportedDocumentType, match=r"\.txt, \.md, \.docx, \.pdf"):
        extract_text("archive.zip", b"PK\x03\x04")


def test_extract_text_empty_file_is_empty_document_error():
    with pytest.raises(EmptyDocumentError):
        extract_text("policy.txt", b"   \n\t  ")


def test_extract_text_corrupted_docx_reports_format_error():
    """Повреждённый архив — понятная ошибка формата, а не трассировка парсера."""
    with pytest.raises(Exception) as exc_info:
        extract_text("handbook.docx", b"not a zip at all")

    assert "docx" in str(exc_info.value)


def test_extract_text_corrupted_pdf_reports_format_error():
    with pytest.raises(Exception) as exc_info:
        extract_text("report.pdf", b"%PDF-1.7 broken tail")

    assert "pdf" in str(exc_info.value)


def test_document_extension_lowercases_and_keeps_dot():
    assert document_extension("Policy.TXT") == ".txt"
    assert document_extension("Makefile") == ""


# --- Chunking (задача 3.2) ---


def test_chunk_text_short_text_is_single_chunk():
    assert chunk_text(POLICY_TEXT) == [POLICY_TEXT]


def test_chunk_text_empty_text_gives_no_chunks():
    assert chunk_text("   \n  ") == []


def test_chunk_text_respects_size_and_overlaps_neighbours():
    """Каждый chunk в пределах size, соседние перекрываются (design D7)."""
    text = " ".join(f"слово{i}" for i in range(600))

    chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    assert len(chunks) > 1
    assert all(len(chunk) <= CHUNK_SIZE for chunk in chunks)
    for current, following in zip(chunks, chunks[1:]):
        tail = current[-CHUNK_OVERLAP:]
        # Хвост предыдущего окна встречается в начале следующего: факт на
        # стыке не теряется.
        assert any(
            tail[i:] and following.startswith(tail[i:])
            for i in range(len(tail))
        )


def test_chunk_text_does_not_split_words():
    words = [f"слово{i:03d}" for i in range(400)]

    chunks = chunk_text(" ".join(words), size=200, overlap=40)

    known = set(words)
    for chunk in chunks:
        # Первое и последнее слово каждого окна — целые слова из исходника.
        tokens = chunk.split()
        assert tokens[0] in known
        assert tokens[-1] in known


def test_chunk_text_covers_whole_text():
    text = " ".join(f"факт{i}" for i in range(500))

    chunks = chunk_text(text, size=300, overlap=60)

    for index in range(500):
        assert any(f"факт{index}" in chunk for chunk in chunks)


def test_chunk_text_overlap_not_smaller_than_size_terminates():
    """Перекрытие ≥ size обрезается до size-1: разбиение всё равно конечно."""
    chunks = chunk_text("а" * 50, size=10, overlap=99)

    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks)


# --- Лимиты (задача 3.2, design D8) ---


def test_split_document_end_to_end_returns_chunks():
    chunks = split_document(
        "policy.txt",
        POLICY_TEXT.encode("utf-8"),
        max_chars=10_000,
        max_chunks=100,
    )

    assert chunks == [POLICY_TEXT]


def test_split_document_rejects_text_over_char_limit():
    data = ("текст " * 500).encode("utf-8")

    with pytest.raises(DocumentTooLarge, match="символов"):
        split_document("policy.txt", data, max_chars=100, max_chunks=1000)


def test_split_document_rejects_too_many_chunks():
    data = (" ".join(f"слово{i}" for i in range(2000))).encode("utf-8")

    with pytest.raises(DocumentTooLarge, match="фрагментов"):
        split_document("policy.txt", data, max_chars=1_000_000, max_chunks=2)
