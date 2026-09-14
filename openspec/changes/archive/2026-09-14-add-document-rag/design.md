## Context

См. proposal.md — Why. Сейчас: text-only Telegram → agent loop → tools (`exec`, `read_file`, `get_skill`, `search_history`, `list_sessions`); chat через `LLMClient` / RouterAI; две SQLite БД (memory, observability) на VM-локальном диске. Документы игнорируются; эмбеддингов нет. Домашка требует собственный RAG (не LangChain-style black box) и tool-based retrieval; бонус §19 вне скоупа.

## Goals / Non-Goals

**Goals:**
- Pipeline index + retrieve на SQLite + sqlite-vec с изоляцией `user_id`.
- Шов `EmbeddingClient` рядом с `LLMClient`, hermetic Fake.
- Telegram: document handler, `/documents`, `/delete <filename>`, replace по имени.
- Tool `search_documents` в agent loop; attribution + отказ без фантазий.
- Лимиты размера; ≥5 тестов + eval dataset; README-секции из задания.

**Non-Goals:**
- Progress UX, page numbers, hybrid search, reranking, conversation-aware RAG (§19).
- Хранение сырых файлов на диске после индексации.
- Готовые RAG-фреймворки, скрывающие pipeline.
- Смена conversation-memory / sandbox контрактов.

## Decisions

### D1: Отдельная БД `rag.db` (не memory)
Как memory/obs: `RAG_DB_PATH` default `~/.local/share/dev-helper-bot/rag.db`. Не смешивать с перепиской — другой lifecycle и sqlite-vec extension.

**Alternatives:** одна БД на всё — проще backup, выше риск поломать memory при экспериментах с vec.

### D2: Схема A — documents / chunks / chunk_vectors
```
documents(id, user_id, filename, file_type, created_at)
  UNIQUE(user_id, filename)
chunks(id, document_id, chunk_index, text)
chunk_vectors USING vec0(
  chunk_id INTEGER PRIMARY KEY,  -- = chunks.id
  embedding float[1024],
  user_id INTEGER PARTITION KEY
)
```
Поиск: KNN с `user_id` partition + JOIN chunks/documents → text + filename + chunk_index.

**Alternatives:** всё в vec0 auxiliary (B) — дальше от формулировок задания; post-filter без partition (C) — риск утечки Top-K.

### D3: Владелец = Telegram `from_user.id`
Документы и tool scoped по `user_id`, не по `chat_id` (в группах они расходятся). История чата по-прежнему `chat_id`.

### D4: `/delete` и replace по basename
Имя = `document.file_name` от Telegram. `delete_by_filename(user_id, filename)` каскадно чистит vectors → chunks → document. Upload с тем же именем: delete затем insert в одной логической операции store.

### D5: EmbeddingClient отдельно от LLMClient
Protocol `embed(texts) -> list[list[float]]`; реализация OpenAI-compatible `/embeddings` на том же `LLM_BASE_URL`/`LLM_API_KEY`; `EMBEDDING_MODEL=baai/bge-m3` (dim **1024**, подтверждено probe RouterAI). Fake: детерминированный hash→vector фиксированной dim для pytest.

**Alternatives:** local sentence-transformers — тяжелее sbx; `text-embedding-3-small` — ok, но слабее multilingual для RU-сценария домашки.

### D6: Индексация в handler, поиск только tool
Download/extract/chunk/embed/store — в `handle_document`. Вопросы — обычный text → agent → опционально `search_documents`. Не индексировать через tool.

### D7: Chunking / K (стартовые значения для README)
- Chunk size **800** символов, overlap **120** (~15%) — баланс контекста абзаца и числа embed-вызовов для политик/гайдов.
- Top-K **5** — достаточно для attribution без раздувания tool result под лимит compaction.
Значения — константы/config; объяснение trade-off в README (слишком мелкие chunks теряют контекст; слишком крупные — шум и дорогой embed).

### D8: Лимиты
Defaults: `RAG_MAX_UPLOAD_BYTES=5_000_000`, `RAG_MAX_EXTRACT_CHARS=300_000`, `RAG_MAX_CHUNKS_PER_DOC=800`. Проверка bytes до тяжёлой работы; chars/chunks после extract/split.

### D9: Парсеры
- txt/md: decode UTF-8 (fallback replace).
- docx: `python-docx`.
- pdf: `pypdf` (текст; сканы без OCR — ограничение README).
Не писать собственные PDF/DOCX parsers.

### D10: sqlite-vec load через драйвер `sqlean.py`
При `DocumentStore.open`: load extension (`sqlite_vec.loadable_path()`). Нет расширения → fail-fast при старте с понятной ошибкой (RAG обязателен для change).

Драйвер SQLite для `rag.db` — **`sqlean.py`**, а не stdlib `sqlite3`: сборки CPython с python.org (и uv/python-build-standalone частично) компилируются с `SQLITE_OMIT_LOAD_EXTENSION`, поэтому `sqlite3.Connection.enable_load_extension` там отсутствует и sqlite-vec не загрузить (проверено локально на CPython 3.14 с python.org: атрибута нет). `sqlean.py` несёт свою сборку SQLite с включённой загрузкой расширений, работает на любом интерпретаторе и снимает зависимость от того, как собран хостовый Python. Асинхронность сохраняется: `aiosqlite.Connection` принимает фабрику соединения (`connector`), в которую передаётся `sqlean.connect`.

Шов остаётся один: драйвер выбирается только внутри `DocumentStore`; `memory.py` и `telemetry.py` продолжают работать на stdlib `sqlite3` — им расширения не нужны.

**Alternatives:** stdlib `sqlite3` + требование пересобрать dev-venv на интерпретаторе с поддержкой расширений (проверено: uv CPython 3.12 умеет) — без новой зависимости, но `pytest` перестаёт быть переносимым между машинами; `pysqlite3-binary` — то же самое, но без wheels под часть платформ.

### D11: «Не выдумывай»
Skill или блок system prompt: при вопросах о документах вызывать `search_documents`; пустой результат → сообщить, что в загруженных документах нет информации; не выдавать общие знания как содержимое документа. Tool result явно маркирует filename для attribution.

### D12: Зависимости и sbx
Добавить в `pyproject.toml` + `requirements.txt`: sqlite-vec, sqlean.py (драйвер из D10), pypdf, python-docx. После merge напомнить `scripts/sbx-setup.sh`.

## Risks / Trade-offs

- [sqlite-vec на платформе sbx/mac] → проверено на старте apply: stdlib `sqlite3` локальных сборок CPython не умеет загружать расширения; закрыто драйвером `sqlean.py` (D10), путь загрузки и причина зафиксированы в README.
- [Смена embedding model ломает индекс] → dim/model в config; смена = wipe rag.db / re-upload; задокументировать.
- [Replace: краткое окно без документа] → delete-then-insert приемлемо без §19; транзакция store минимизирует окно.
- [Стоимость/latency embed больших PDF] → лимиты D8 + batch embed где API позволяет.
- [OCR отсутствует] → честно в Limitations README.
- [Групповые чаты: user_id vs chat_id] → документы личные; вопросы в группе ищут docs отправителя — задокументировать.

## Migration Plan

1. Установить deps, прогнать `sbx-setup` при работе в sandbox.
2. Запуск создаёт пустой `rag.db` + vec table.
3. Откат: убрать handlers/tool, удалить `rag.db`; memory/obs не затрагиваются.
4. Существующие пользователи без документов — поведение text-бота без изменений.

## Open Questions

Нет блокирующих: стартовые chunk/K/limits зафиксированы в D7–D8 и могут подкручиваться без смены спеков, если поведение (лимиты есть, Top-K используется) сохраняется.
