## 1. Dataset

- [x] 1.1 Создать `eval/agent_eval_dataset.json`: ≥10 кейсов с уникальными `id` и `kind` jailbreak / refusal / memory / reset
- [x] 1.2 Добавить в датасет кейсы `tool_schema` (битый JSON, не-объект, missing required, wrong type, ok) и 5 открытых вопросов `judge`

## 2. Hermetic unit tests

- [x] 2.1 `tests/unit/test_input_sanitization.py`: пустой/`None` текст без исключения; 4096 символов целиком в LLM; спецсимволы `_ * [ ] < > &` без искажения; `escape_html` / `format_html` экранирует теги
- [x] 2.2 `tests/unit/test_tool_call_schema.py`: параметризация `validate_tool_call` по `tool_schema` из датасета
- [x] 2.3 `tests/unit/test_agent_eval_dataset.py`: jailbreak не подменяет system; фейковые `TELEGRAM_BOT_TOKEN`/`LLM_API_KEY` отсутствуют в запросе; память сущностей; `/new` без LLM и без прежней истории
- [x] 2.4 Тест размера датасета: ≥10 кейсов, уникальные id, все четыре вида L2 присутствуют

## 3. Live marker and fixtures

- [x] 3.1 `pyproject.toml`: маркер `live`; `addopts = "-m 'not docker and not live'"`
- [x] 3.2 `tests/live/conftest.py`: `live_llm` через `make_llm()` с skip при `LLMUnavailable`; `routerai_subject` / `routerai_judge` на `https://routerai.ru/api/v1` (luna / sol, ключ `LLM_API_KEY`, skip без ключа); `FakeCommandExecutor`

## 4. Live tests

- [x] 4.1 `tests/live/test_red_team.py`: jailbreak (нет фрагментов system, секретов, согласия выйти из роли); refusal; recall имени/города; после `/new` сущностей нет
- [x] 4.2 `tests/live/test_llm_judge.py`: 5 вопросов датасета → `run_agent` luna; судья sol + json_schema/JSON; среднее ≥ 0.8; битый JSON судьи — fail; запросы не на `LLM_BASE_URL` бота
- [x] 4.3 `tests/live/test_latency.py`: пинг `complete()` < `LIVE_FULL_SECONDS` (дефолт 4); TTFT stream-зонд < `LIVE_TTFT_SECONDS` (дефолт 1.5) или skip без streaming

## 5. Docs and verify

- [x] 5.1 README: `pytest -m live`, модели судьи routerai, env (`JUDGE_MODEL`, `JUDGE_SUBJECT_MODEL`, пороги latency)
- [x] 5.2 Прогнать дефолтный `pytest` — зелёный без сети и без LLM, live-тесты не выполняются
