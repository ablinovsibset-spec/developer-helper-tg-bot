## Why

Домашнее задание требует научить агента работать с документами пользователя (txt/md/docx/pdf) через собственный RAG на SQLite + sqlite-vec. Сейчас бот принимает только текст: загрузка файлов игнорируется, эмбеддингов и векторного поиска нет, а `search_history` ищет лишь прошлые чаты по LIKE — этого недостаточно для Q&A по загруженным политикам/гайдам с attribution и изоляцией пользователей.

## What Changes

- Приём документов в Telegram (`.txt`, `.md`, `.docx`, `.pdf`): скачивание → извлечение текста → chunking → embeddings → запись в отдельную SQLite БД с **sqlite-vec**.
- UX индексации: сообщение о начале обработки и подтверждение готовности (без пошагового progress из бонуса §19).
- Новый инструмент агента `search_documents(query)`: retrieval Top-K по документам **текущего** пользователя; RAG не заливается целиком в system prompt.
- Несколько документов на пользователя; повторная загрузка того же имени — **replace** (каскадное удаление старого индекса + новая индексация).
- Изоляция по Telegram `user_id`: чужие документы не участвуют в поиске.
- Команды `/documents` и `/delete <filename>` вне agent loop (как `/new`).
- Source attribution в ответах (`Источник: filename`); при отсутствии релевантных chunks — отказ без выдумывания «из документа».
- Лимиты размера: сырой файл, объём извлечённого текста, число chunks; понятные ошибки вместо падения.
- Шов `EmbeddingClient` (RouterAI `/embeddings`, модель `baai/bge-m3`, dim 1024) + Fake для hermetic-тестов.
- ≥5 автоматических тестов (parse/chunk/retrieval/isolation/e2e) и evaluation dataset ≥5 вопросов; README с architecture/chunking/embeddings/retrieval/storage/security/limitations.
- **Вне скоупа**: бонус §19 (progress UX, page numbers, hybrid search, reranking, conversation-aware RAG).

## Capabilities

### New Capabilities

- `document-rag`: загрузка и индексация пользовательских документов, хранение в SQLite + sqlite-vec, retrieval через tool `search_documents`, команды `/documents` и `/delete`, изоляция по `user_id`, replace по имени, attribution и отказ при отсутствии фактов, лимиты и ошибки индексации.
- `embedding-provider`: контракт эмбеддингов (отдельный от `LLMClient`), OpenAI-compatible `/embeddings` реализация, конфигурация модели/размера вектора, тестовый двойник.

### Modified Capabilities

- `telegram-bot`: бот SHALL принимать документы поддерживаемых форматов (сейчас нетекстовые сообщения игнорируются); добавляются обработчики `/documents` и `/delete`; document handler не гоняет агентный цикл.
- `test-suite`: минимальный набор тестов расширяется покрытием RAG-pipeline и evaluation dataset для retrieval.

## Impact

- **Код**: `main.py` (document + команды), новый store/pipeline модуль(и), `tools.py` (+ `search_documents`), skill/system guidance для RAG, `config.py` (`RAG_DB_PATH`, embedding env, лимиты), возможно тонкая проводка `user_id` в tool closure рядом с `chat_id`.
- **Зависимости**: sqlite-vec; `sqlean.py` как драйвер SQLite для `rag.db` (stdlib `sqlite3` локальных сборок CPython не умеет загружать расширения, design D10); парсеры PDF/DOCX (например pypdf / python-docx); sync `requirements.txt` ↔ `pyproject.toml`; после смены deps — напомнить `scripts/sbx-setup.sh`.
- **Инфраструктура**: файл `rag.db` на VM-локальном диске (тот же gotcha, что memory/obs); загрузка расширения sqlite-vec в процессе бота.
- **Провайдер**: тот же RouterAI `LLM_BASE_URL` / `LLM_API_KEY`, отдельная `EMBEDDING_MODEL=baai/bge-m3`.
- **Существующее**: conversation memory и sandbox tools без изменения контракта; chat и embeddings — разные швы.
