# Tasks: add-obs-web-dashboard

## 1. Read-API телеметрии

- [x] 1.1 Добавить в `TelemetryStore` async read-методы: агрегаты (`label`/`since`), список прогонов, run + timeline (llm_calls/tool_calls)
- [x] 1.2 Добавить read-метод(ы) аудита (топ tools, дорогой turn, prompt_roles/context types, repeats, raw/effective cost) с той же семантикой, что `obs-audit.py`
- [x] 1.3 Unit-тесты чтения: фильтры, пустая БД, неизвестный run_id, null ≠ 0 для cached/reasoning; сверка ключевых цифр аудита с фикстурой

## 2. HTTP-дашборд

- [x] 2.1 Модуль `obs_web`: aiohttp app, bind `127.0.0.1`, HTML+CSS страницы `/`, `/runs`, `/runs/{id}`, `/audit` и JSON `/api/*` с query-фильтрами
- [x] 2.2 Клиентский poll ~1.5 с; выбранный `run_id` в URL не сбрасывается при обновлении списка
- [x] 2.3 Режим без store: все страницы отдают «телеметрия недоступна»
- [x] 2.4 Unit/integration-тесты хендлеров (aiohttp test utilities): сводка, timeline, audit, stub без store

## 3. Lifecycle в main

- [x] 3.1 `config`: дефолт порта 8765, переопределение `OBS_WEB_PORT`
- [x] 3.2 В `main`: старт AppRunner до polling; stop в `finally`; занятый порт → процесс не стартует polling, понятное сообщение
- [x] 3.3 Документация: README + `.env.example` (URL дашборда, порт, что CLI остаётся)

## 4. Проверка

- [x] 4.1 Ручная проверка: бот + чат → на `http://127.0.0.1:8765` видны список/timeline/агрегаты/аудит с poll; фильтры label/since
- [x] 4.2 Регрессия: `obs-dashboard.py` и `obs-audit.py` без изменений поведения; unit-тесты телеметрии/агента зелёные
