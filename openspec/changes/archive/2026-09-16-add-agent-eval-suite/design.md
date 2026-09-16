## Context

См. proposal.md — Why. Сейчас: герметичный pytest на двойниках (`FakeLLM`, `FakeBot`, `FakeCommandExecutor`); JSON-аргументы уже валидирует `validate_tool_call` в `agent.py`; история и `/new` покрыты в `tests/unit/test_main.py`; чанкинг исходящих 4096 — `send_chunked`. Бот шлёт с `ParseMode.HTML`, не MarkdownV2. Стриминга в `OpenAICompatibleClient` нет. Дефолтный chat-endpoint проекта — `https://routerai.ru/api/v1`, модель `openai/gpt-5.6-luna`. Маркер `docker` уже исключён из `addopts`.

## Goals / Non-Goals

**Goals:**
- Один датасет как источник правды для герметичной политики и live-проверок.
- Дефолтный `pytest` остаётся без сети, без живого LLM и без секретов.
- Live — отдельный маркер; судья жёстко на routerai (luna / sol).

**Non-Goals:**
- Менять продуктовую логику (санитайзер входа, output-filter jailbreak, streaming в клиенте, смена parse mode).
- Переносить существующие тесты из `test_agent.py` / `test_main.py`.
- E2E с Telegram API.
- Пинить `@provider=openai/flex` у судьи.

## Decisions

### D1: Гибрид герметичность / live
Дефолт — двойники и политика (system на месте, секреты не в запросе, история, `/new`). Поведение модели — только `@pytest.mark.live`.

**Alternatives:** всё в дефолтном pytest на живом LLM — ломает спеку изоляции; всё на FakeLLM, включая «jailbreak отклонён» — проверяет скрипт двойника, не агента.

### D2: Датасет `eval/agent_eval_dataset.json`
Рядом с `eval/rag_eval_dataset.json`. Поля: `id`, `kind` (`jailbreak` | `refusal` | `memory` | `reset` | `tool_schema` | `judge`), `turns`, ожидаемые проверки (запрещённые подстроки / сущности / схема JSON). ≥10 кейсов L2 + schema + 5 judge-вопросов.

**Alternatives:** YAML — лишний формат; класть в `tests/fixtures/` — датасет не переиспользуется вне pytest.

### D3: Санитизация фиксирует текущее поведение
Пустой `message.text` → `""`, цикл не падает (LLM всё равно вызывается). 4096 символов входа уходят целиком. Спецсимволы во входе не фильтруются. HTML-escape — только шаблоны бота (`escape_html` / `format_html`), не сырой ответ модели (скилл habr-feed отдаёт HTML).

**Alternatives:** отклонять пустой ввод / экранировать ответы модели — продуктовое изменение, вне скоупа.

### D4: JSON-контракт параметризует существующий валидатор
Новый узкий тест читает `tool_schema` из датасета и зовёт `validate_tool_call`. Существующие сценарии retry / `finish_reason=length` в `test_agent.py` не дублируем переносом.

**Alternatives:** вынести валидатор в отдельный модуль — не нужно для домашки.

### D5: Маркер `live` как `docker`
`addopts = "-m 'not docker and not live'"`. Live-фикстуры: `live_llm = make_llm()` для red team и latency; при `LLMUnavailable` — `pytest.skip`. Docker-песочницу live не поднимает: `FakeCommandExecutor`.

**Alternatives:** отдельный tox env — тяжелее для домашки.

### D6: Судья всегда routerai, две модели
Не `make_llm()` / не `LLM_BASE_URL` бота. Два `OpenAICompatibleClient` на `https://routerai.ru/api/v1`, ключ `LLM_API_KEY`. Субъект: `openai/gpt-5.6-luna` (`JUDGE_SUBJECT_MODEL`). Судья: `openai/gpt-5.6-sol` (`JUDGE_MODEL`). Рубрика JSON: вежливость / точность / краткость 0–1; `response_format` json_schema, если агрегатор принимает, иначе разбор текста тем же `json.loads`. Битый JSON судьи — fail, не 0. Порог: среднее ≥ 0.8. `run_agent` субъекта без tools (или с фейковым executor).

**Alternatives:** одна модель на оба шага — слабее домашки «более сильная модель»; судья = env LLM — локальный Ollama подмешается в сценарий, который должен идти через routerai.

### D7: Latency — env LLM, пороги из env
Короткий пинг без tools. Полный `complete()` < `LIVE_FULL_SECONDS` (дефолт 4). TTFT: отдельный aiohttp POST `stream=true` к тому же endpoint, время до первого SSE-чанка < `LIVE_TTFT_SECONDS` (дефолт 1.5). Нет streaming — skip только TTFT. В боте ответа стрима нет; зонд — для домашки, не контракт продукта.

**Alternatives:** добавить streaming в клиент — продуктовый change.

### D8: Секреты
Live/judge читают ключ только в заголовке клиента. Датасет и ассерты не содержат ключей. Monkeypatch в герметичном jailbreak задаёт фейковые значения env и проверяет их отсутствие в `llm.requests`.

## Risks / Trade-offs

- [Live red team / судья флакуют] → узкие запрещённые подстроки; skip при недоступности; порог 0.8 может потребовать `JUDGE_*` override.
- [TTFT 1.5 с / full 4 с не держатся на холодном или облачном endpoint] → env-пороги; skip TTFT без stream.
- [Судья дороже luna] → только `pytest -m live`, не дефолт; sol без flex-пина ради стабильного JSON.
- [Пустой ввод всё равно зовёт LLM] → честно фиксируем текущее поведение; не маскируем тестом «агент отказал».

## Migration Plan

1. Датасет + unit + маркер `live` — дефолтный `pytest` зелёный как раньше плюс новые unit.
2. `pytest -m live` вручную при ключе routerai / доступном LLM.
3. Откат: удалить датасет, `tests/live/`, новые unit и вернуть `addopts` к `not docker`.
