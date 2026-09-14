## Context

См. proposal.md — Why. Живой RAG: индексация в `handle_document` (два сообщения: старт / готово), `documents.extract_text` склеивает PDF-страницы, `chunks` без страниц, `DocumentStore.search` — только sqlite-vec KNN с `user_id` PARTITION KEY, `UserDocumentSearcher` эмбеддит голый `query`. История сессии уже есть в `handle_text`, в поиск не передаётся. Capability `document-rag` живёт в ещё не заархивированном `add-document-rag`; этот change его дополняет, не заменяет.

## Goals / Non-Goals

**Goals:**
- Пять бонусов §19 на текущем pipeline без нового провайдерского шва и без новых пакетов.
- Сохранить chunk size/overlap, Top-K=5, изоляцию partition key и hermetic eval.
- Миграция существующего `rag.db` без обязательного wipe.

**Non-Goals:**
- LLM-rewrite запроса, LLM-rerank, cross-encoder, `/rerank` API.
- FTS для `search_history`.
- Смена сигнатуры `search_documents`.
- OCR и хранение сырых файлов.

## Decisions

### D1: Одно статус-сообщение, edit in place
Первое сообщение «документ получен» редактируется (`edit_message_text`) по мере шагов. Не спамит чат; на защите шаги всё равно видны. Если edit не удался — не валить индексацию, дописать новым сообщением.

**Alternatives:** пачка сообщений как в примере домашки — ближе к цитате, хуже UX.

### D2: Батч эмбеддингов с N/M
`EmbeddingClient.embed` уже принимает список: режем chunks на батчи и обновляем статус `k/N`. Размер батча — константа (порядка 32), не env: не наблюдаемое поведение.

**Alternatives:** один вызов на документ — прогресс эмбеддингов бинарный, на большом PDF бесполезен.

### D3: Глобальный chunking + карта страниц
PDF извлекается по страницам; текст склеивается как сейчас; границы чанков мапятся на 1-based номера страниц → `page_start` / `page_end`. Overlap и eval не ломаются. Не-PDF: оба поля NULL.

**Alternatives:** резать по страницам — точный номер, но другой размер окна и дыры на границах страниц.

### D4: Схема chunks + FTS рядом с vec0
```
chunks(..., page_start INTEGER, page_end INTEGER)  -- NULL = нет страницы
chunk_fts USING fts5(text, content='chunks', content_rowid='id')
```
Синхронизация FTS руками в той же транзакции, что vec (как каскад delete): content-таблица без триггеров проще отлаживать. Поиск FTS: `MATCH` + `JOIN documents WHERE user_id = ?`. Нет FTS5 в sqlean → fail-fast при `open()`, тот же тон, что у sqlite-vec.

**Alternatives:** LIKE — не бонусный text search; sparse-векторы bge-m3 — смена embedding-контракта.

### D5: Hybrid = два канала + RRF, затем rerank
Каждый канал: k_cand=15. RRF `1/(60+rank)` объединяет id фрагментов. Rerank на объединённом наборе: RRF + буст точного вхождения строки запроса + перекрытие токенов; наружу DEFAULT_SEARCH_K=5. Без чат-LLM: детерминированно для pytest и без лишней латентности в агентном цикле.

**Alternatives:** взвешенная сумма cosine/BM25 — разные шкалы; LLM-rerank — +1 complete на каждый tool call.

### D6: Conversation context в замыкании searcher
В `handle_text` уже есть `session_history`. В `UserDocumentSearcher` передаём последние реплики до текущего сообщения (последний user и, если есть, последний assistant). Поисковая строка: `previous + "\n" + query`, если previous не пуст. Протокол `search(query)` не меняется. Skill/`DOCUMENTS_ENV_LINE`: follow-up формулировать самодостаточным query — страховка, не замена D6.

**Alternatives:** LLM-rewrite — дороже и дублирует историю, которая уже в чат-контексте; только skill — не видно в retrieval-коде на защите.

### D7: Attribution
Инструмент: `[источник: file.pdf, стр. 17, фрагмент 12]` или диапазон `стр. 16–17`. Русский «стр.», потому что бот русскоязычный. Skill: если страница есть в результате — копировать в ответ.

### D8: Миграция при open()
`ALTER TABLE chunks ADD COLUMN page_start/page_end`; создать FTS и заполнить из существующих `chunks.text`. Старые PDF без страниц до перезагрузки файла. Wipe не требуется (в отличие от смены `EMBEDDING_DIM`).

## Risks / Trade-offs

- [FTS5 нет в sqlean] → проверка на старте; если нет — не стартовать, как с sqlite-vec.
- [Русская морфология FTS] → ожидаемо слабо; vec закрывает перефраз, FTS — точные номера и редкие токены.
- [Спецсимволы в MATCH] → санитайз запроса FTS; ошибка MATCH не роняет агентный цикл, канал vec остаётся.
- [Eval на FakeEmbeddingClient лексический] → vec и FTS часто согласны; отдельный тест на расхождение (редкий токен vs перефраз) плюс eval через публичный retrieve.
- [Edit status vs flood] → D1; короткий файл промелькнёт шагами — для демо нужен PDF покрупнее.
- [Архив add-document-rag] → main `openspec/specs/document-rag` появится только после его archive; этот change пишет дельту к той capability.

## Migration Plan

1. Заархивировать `add-document-rag` до archive этого change (иначе дельта document-rag не к чему мержиться).
2. Выкатить код: при старте `DocumentStore.open` мигрирует схему.
3. Существующие документы ищутся гибридом сразу; страницы PDF — после повторной загрузки того же имени.
4. Откат: вернуть поиск к одному KNN, колонки/FTS можно оставить (не мешают).
5. После deps не требуется `sbx-setup` (пакетов нет); живой бот лучше перезапустить, чтобы подхватить миграцию.

## Open Questions

Нет: размер батча эмбеддингов и k_cand можно подкрутить без смены спеков, пока финальный Top-K и наличие двух каналов + rerank сохраняются.
