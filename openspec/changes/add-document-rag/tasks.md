## 1. Dependencies and config

- [x] 1.1 Добавить зависимости `sqlite-vec`, `sqlean.py`, `pypdf`, `python-docx` в `pyproject.toml` и синхронизировать `requirements.txt`
- [x] 1.2 Расширить `config.py` / `.env.example`: `RAG_DB_PATH`, `EMBEDDING_MODEL`, лимиты `RAG_MAX_UPLOAD_BYTES` / `RAG_MAX_EXTRACT_CHARS` / `RAG_MAX_CHUNKS_PER_DOC`, фабрика embedding-клиента

## 2. Embedding provider

- [x] 2.1 Ввести Protocol `EmbeddingClient` + исключение недоступности (отдельно от `LLMClient`)
- [x] 2.2 Реализовать OpenAI-compatible клиент `POST /embeddings` (base URL, API key, отдельная модель, dim 1024 для bge-m3)
- [x] 2.3 Добавить `FakeEmbeddingClient` (детерминированные векторы) в тестовые фикстуры

## 3. Document store and pipeline

- [x] 3.1 Реализовать extractors: txt/md, docx, pdf + ошибки формата/повреждения/пустого текста
- [x] 3.2 Реализовать chunking (size 800, overlap 120) и проверку лимитов chars/chunks
- [x] 3.3 Реализовать `DocumentStore` на aiosqlite + sqlite-vec: schema documents/chunks/chunk_vectors, open с load extension, WAL, путь VM-local
- [x] 3.4 Реализовать `index_document` / `replace`, `delete_by_filename`, `list_documents`, `search(user_id, query_vec, k)` с partition по `user_id`

## 4. Telegram and agent wiring

- [x] 4.1 Handler загрузки документа: validate → notify start → download → extract → chunk → embed → store → notify ready; лимиты и понятные ошибки
- [x] 4.2 Команды `/documents` и `/delete <filename>` вне agent loop, без записи в memory
- [x] 4.3 Tool `search_documents` + ToolSpec; wiring `user_id` в closure; добавить в `AGENT_TOOLS`
- [x] 4.4 Обновить system prompt / skill: когда звать поиск, attribution `Источник: …`, отказ без выдумывания
- [x] 4.5 Скорректировать `telegram-bot` поведение: поддерживаемые документы не игнорируются как «медиа без ответа»

## 5. Tests, eval, docs

- [x] 5.1 ≥5 hermetic тестов: parse, chunking, retrieval, user isolation, e2e (Fake LLM + Fake embeddings; Telegram замокан)
- [x] 5.2 Evaluation dataset ≥5 вопросов с `expected_source` + автоматическая проверка, что Top-K содержит ожидаемый источник
- [x] 5.3 README: architecture, chunking, embeddings, retrieval (metric/K), storage, security (isolation), limitations; упомянуть `sbx-setup` после deps
- [x] 5.4 Прогнать `pytest` (дефолтный hermetic набор)
