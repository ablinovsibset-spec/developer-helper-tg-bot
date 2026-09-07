# Tasks: add-agent-observability

## 1. Контракт LLM: usage

- [x] 1.1 Расширить `AssistantTurn` опциональным полем `usage` (nullable:
  input/output/cached/reasoning + raw JSON), извлечь usage в `_parse_turn`
  (отсутствие полей → null, не ошибка)
- [x] 1.2 Тесты: ответ с полным/неполным usage и без него (llm-provider delta,
  сценарии «Ответ содержит usage» / «Ответ без usage»)

## 2. Хранилище телеметрии

- [x] 2.1 Модуль `telemetry.py`: `TelemetryStore` (aiosqlite, WAL, схема
  runs/llm_calls/tool_calls, FK на run), open/close по паттерну MemoryStore
- [x] 2.2 Best-effort запись: каждая вставка в try/except с log.warning;
  путь — `OBS_DB_PATH`, дефолт на VM-локальном диске (design D4)
- [x] 2.3 Тесты: создание схемы, вставки, толерантность к null-usage,
  сбой записи не возбуждает исключение наружу

## 3. Recorder и обёртка LLM

- [x] 3.1 `RunRecorder`: создаёт run (chat_id, label), счётчик turn_number по
  вызовам `complete()`, финализация прогона со статусом и агрегатами
- [x] 3.2 `ObservingClient` (декоратор над `LLMClient`): latency вокруг
  `complete()`, usage/messages_count/prompt_chars/model, запись в store;
  неуспешный вызов — запись с исходом «ошибка» (design D2)
- [x] 3.3 Тесты: turn_number по возрастанию, повторная ошибка LLM фиксируется,
  поведение обёртки идентично голому клиенту (прозрачность)

## 4. Инструментация инструментов и проводка

- [x] 4.1 Точка записи в цикле `run_agent`: имя, размер аргументов/результата,
  токены-оценка (chars/4, design D6), duration, исход (вкл. отказ валидации)
- [x] 4.2 Проводка в `main.py`: открыть/закрыть TelemetryStore, создать
  recorder на сообщение, обернуть LLM, финализировать прогоны (успех/лимит/
  ошибка LLM); прайс из окружения (design D5), дефолт по прайсу gpt-oss-20b
  (open question из design — уточнить значения)
- [x] 4.3 Тесты: статус прогона для терминальных возвратов run_agent и ветки
  LLMUnavailable; телеметрия не меняет ответы агента (герметичный прогон
  с фейками)

## 5. Dashboard

- [x] 5.1 `scripts/obs-dashboard.py`: режим агрегатов (токены по типам,
  стоимость, средние на прогон, топ инструментов, cache-hit при наличии
  данных) и режим `--run <id>` (timeline ходов)
- [x] 5.2 Обработка пустой базы и отсутствия run_id — понятные сообщения,
  не трейсбек; N/A вместо нуля для недоступных данных
- [x] 5.3 Тесты dashboard на засеянной БД (агрегаты + timeline) и пустой

## 6. Документация и приёмка

- [x] 6.1 README: переменные окружения (OBS_DB_PATH, OBS_*_PRICE_*), запуск
  dashboard, замечание о ротации БД; .env.example дополнить
- [x] 6.2 Прогон живого бота: несколько сообщений с exec-вызовами, проверить
  dashboard-агрегаты и timeline против журнала; `openspec validate
  add-agent-observability` без ошибок
