## Why

Домашка «тест-сьют для Telegram ИИ-агента» требует проверки ядра агента: санитизация входа, JSON-контракт инструментов, red team / отказ от выдумок / память, плюс опциональные LLM-as-a-Judge и latency SLA. Дефолтный pytest уже герметично покрывает цикл, валидацию аргументов, чанкинг 4096 и `/new`, но нет единого датасета 10–15 кейсов, явных тестов входа (пустое/длинное/спецсимволы) и отдельного live-набора, который не ломает изоляцию `pytest`.

## What Changes

- Датасет `eval/agent_eval_dataset.json` (10–15 кейсов: jailbreak, отказ, память, сброс `/new`; плюс JSON-схема аргументов и 5 открытых вопросов для судьи).
- Герметичные unit-тесты: санитизация входа, параметризация `validate_tool_call` из датасета, политика агента по датасету (system остаётся первым, секреты не в промпте, история и `/new`).
- Маркер pytest `live`, исключённый из дефолтного `pytest` так же, как `docker`.
- Live-тесты (`pytest -m live`): red team / отказ / recall через `make_llm()` из env; latency SLA через тот же клиент; LLM-as-a-Judge — два вызова на `https://routerai.ru/api/v1` (`openai/gpt-5.6-luna` отвечает, `openai/gpt-5.6-sol` судит).
- Документация запуска live-набора. Продуктовый код агента не меняется: тесты фиксируют текущее поведение, не добавляют санитайзер, output-filter или streaming в клиент.

## Capabilities

### New Capabilities

- (нет)

### Modified Capabilities

- `test-suite`: дефолтный pytest покрывает санитизацию входа, контракт JSON-аргументов из датасета и политику агента (jailbreak не подменяет system, секреты не утекают в запрос, память и `/new`); репозиторий содержит evaluation dataset ≥10 кейсов; тесты против живого LLM / routerai-судьи идут под маркером `live` и MUST NOT входить в дефолтный прогон.

## Impact

- **Код продукта**: без изменений (`handle_text`, `validate_tool_call`, `OpenAICompatibleClient` остаются как есть).
- **Тесты**: новые `tests/unit/test_input_sanitization.py`, `tests/unit/test_tool_call_schema.py`, `tests/unit/test_agent_eval_dataset.py`; каталог `tests/live/` (red team, judge, latency, conftest). Существующие `test_agent.py` / `test_main.py` не переносим.
- **Датасет**: `eval/agent_eval_dataset.json` рядом с уже существующим RAG-eval.
- **Конфиг**: маркер `live` и `addopts = "-m 'not docker and not live'"` в `pyproject.toml`.
- **Зависимости**: без новых пакетов.
- **Секреты**: live/judge читают `LLM_API_KEY` только в клиенте; ключ не попадает в датасет и ассерты. Нет ключа / недоступен LLM — skip, не падение дефолтного набора.
- **Риски**: live-тесты флакуют на модели; пороги latency (1.5 с TTFT / 4 с полный ответ) и score судьи ≥ 0.8 могут не держаться на холодном/медленном endpoint — пороги переопределяются env, skip при недоступности.
