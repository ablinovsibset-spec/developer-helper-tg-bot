## Why

Базовый RAG из `add-document-rag` закрывает обязательную часть домашки, но бонус §19 сознательно остался снаружи: индексация без пошагового прогресса, attribution без страниц PDF, чистый векторный поиск без FTS/rerank, а follow-up вроде «а можно перенести их?» зависит от того, напишет ли модель самодостаточный `query`. На защите это минус до шести баллов и дыры, которые уже честно перечислены в README.

## What Changes

Один change закрывает все пять пунктов §19; резать на отдельные PR не будем.

- **Progress**: во время индексации пользователь видит шаги extract → chunk → embed → indexed (одно редактируемое статус-сообщение; эмбеддинги батчами с `N/M` на длинных документах).
- **Page numbers**: для PDF chunk хранит span страниц; attribution в результате `search_documents` и в ответе агента — `Источник: file.pdf, стр. 17` (диапазон, если фрагмент на стыке).
- **Hybrid search**: рядом с sqlite-vec — FTS5 по тексту chunks; результаты сливаются Reciprocal Rank Fusion; оба канала фильтруют по `user_id`.
- **Reranking**: отдельный этап после объединения кандидатов (шире, чем финальный Top-K) — скоринг без нового LLM/cross-encoder.
- **Conversation-aware RAG**: retrieval подмешивает предыдущие реплики открытой сессии в поисковый запрос (конкатенация, не отдельный LLM-rewrite), чтобы анафорический follow-up находил те же документы, что и полный вопрос.
- Тесты и eval: progress, страницы PDF, изоляция FTS, follow-up retrieval; README перестаёт помечать эти пункты как limitations.
- **Вне скоупа**: OCR; LLM-rerank и отдельный rerank-API; FTS для `search_history`; смена chunk size/overlap; смена контракта `search_documents(query)`.

## Capabilities

### New Capabilities

- (нет)

### Modified Capabilities

- `document-rag`: пошаговый progress индексации; страницы PDF в chunks и attribution; hybrid retrieval (vec + FTS5 + RRF); rerank кандидатов; conversation-aware query; изоляция `user_id` на всех каналах поиска.
- `test-suite`: дефолтный pytest покрывает progress UX (через fake Telegram), page span PDF, гибридный поиск с изоляцией FTS, rerank как отдельный этап, retrieval follow-up с историей; eval остаётся герметичным и идёт через тот же retrieval path, что и инструмент.

## Impact

- **Код**: `documents.py` (extract по страницам, chunk со span), `document_store.py` (колонки страниц, FTS5, hybrid+rerank), `main.py` (edit статус-сообщения, батч embed, прокидка истории сессии в `UserDocumentSearcher`), skill/`DOCUMENTS_ENV_LINE` (attribution со страницей, самодостаточный follow-up query).
- **Схема `rag.db`**: новые колонки у chunks и виртуальная FTS-таблица; миграция при `open()` либо documented wipe старого индекса.
- **Зависимости**: новых пакетов не ожидается (FTS5 в SQLite/sqlean). Если FTS5 в sqlean отсутствует — fail-fast при старте, как у sqlite-vec.
- **Существующее**: `search_documents` остаётся одним аргументом `query`; `EmbeddingClient` / `LLMClient` / conversation-memory контракты не меняются; агентный цикл не показывает промежуточные шаги поиска.
