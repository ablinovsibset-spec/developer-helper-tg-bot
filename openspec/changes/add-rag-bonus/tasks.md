## 1. Pages and chunk spans

- [x] 1.1 Извлекать PDF постранично и строить карту смещений; `split_document` возвращает фрагменты с `page_start`/`page_end` (NULL для не-PDF)
- [x] 1.2 Тесты: страница у PDF-фрагмента, диапазон на стыке страниц, отсутствие страницы у txt/md/docx

## 2. Store schema, FTS, migration

- [x] 2.1 Проверить FTS5 на sqlean при `DocumentStore.open`; нет — fail-fast как у sqlite-vec
- [x] 2.2 Добавить колонки `page_start`/`page_end`, виртуальную FTS-таблицу; синхронизировать FTS в той же транзакции, что vec (index/replace/delete)
- [x] 2.3 Миграция при `open()`: ALTER существующих `rag.db`, заполнить FTS из текущего текста chunks

## 3. Hybrid search and rerank

- [x] 3.1 Текстовый поиск FTS с фильтром `user_id`; санитайз MATCH, ошибка канала не роняет цикл
- [x] 3.2 Объединить vec (k_cand) и FTS (k_cand) через RRF, затем лексический rerank до Top-K без чат-LLM
- [x] 3.3 `format_matches`: страница в источнике (`стр. N` / диапазон), без выдуманной страницы для не-PDF
- [x] 3.4 Публичный retrieve (гибрид + rerank) — единственный путь `UserDocumentSearcher` и eval

## 4. Indexing progress

- [x] 4.1 Одно статус-сообщение с edit по шагам extract → chunks → embeddings; сбой edit не валит индексацию
- [x] 4.2 Эмбеддинги батчами с прогрессом `k/N`; ошибка после прогресса — понятное сообщение, без частичного индекса

## 5. Conversation-aware retrieval

- [x] 5.1 Передать в `UserDocumentSearcher` последние реплики открытой сессии (user + assistant до текущего сообщения) и склеить с `query`
- [x] 5.2 Обновить `DOCUMENTS_ENV_LINE` и `skills/document-qa.md`: страница в «Источник:», самодостаточный follow-up query

## 6. Tests, eval, docs

- [x] 6.1 Hermetic тесты бонусов: progress (fake Telegram), страницы PDF, изоляция FTS, rerank до Top-K без LLM, follow-up с историей
- [x] 6.2 Eval dataset гонять через гибридный retrieve, не через голый vec KNN; поправить кейсы, если Top-K съехал
- [x] 6.3 README: progress, страницы, hybrid/RRF, rerank, conversation-aware; убрать эти пункты из limitations
- [x] 6.4 Прогнать `pytest` (дефолтный hermetic набор)
