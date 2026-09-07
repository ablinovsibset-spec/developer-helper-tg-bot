# Design: add-obs-web-dashboard

## Context

Телеметрия пишется в `TelemetryStore` (aiosqlite, WAL); CLI читает файл
отдельным `sqlite3`. Бот — long-polling в `main.py`; `aiohttp` уже в
зависимостях. См. proposal.md — Why. Решения UX/эксплуатации зафиксированы
в разведке: in-process, poll 1–2 с, `127.0.0.1:8765`, fail-fast на порте,
заглушка без store, ручной выбор run, фильтры на сводке/аудите, HTML+CSS,
read через store, CLI без изменений контракта.

## Goals / Non-Goals

**Goals:**

- HTTP-UI в процессе бота с тремя зонами данных + аудит, обновляемый poll'ом.
- Единый read-слой в `TelemetryStore`, чтобы веб не дублировал SQL CLI в обход store.
- Предсказуемый lifecycle: порт обязателен; телеметрия по-прежнему best-effort.

**Non-Goals:**

- SSE/WebSocket, auth, bind не-loopback, React/SPA.
- Рефакторинг CLI на общий Python-пакет с вебом (можно позже; SQL в store —
  источник истины для веба, CLI может остаться самодостаточным).
- Проброс порта наружу из sbx / reverse-proxy.

## Decisions

### D1. aiohttp Application рядом с polling
Веб — `aiohttp.web.AppRunner`/`TCPSite` на `127.0.0.1`, стартует в `main`
до `start_polling`, останавливается в `finally`. Альтернатива — отдельный
процесс — отвергнута (требование in-process). Альтернатива — поток с
stdlib `http.server` — отвергнута: хуже стык с asyncio и store.

### D2. Fail-fast bind, soft-fail telemetry
Сбой `TCPSite.start()` / занятый порт → исключение до polling, сообщение
с адресом/портом/errno. Сбой `TelemetryStore.open()` → `telemetry=None`,
бот и веб живут, хендлеры отдают stub-страницу. Альтернатива «веб не
поднимать без store» — отвергнута явным UX-решением.

### D3. Read-методы на TelemetryStore
Новые async-методы (агрегаты, list runs, get run + events, audit snapshot)
с параметрами `label`/`since`. Возвращают dataclass/словари, null-tolerant.
Запись остаётся best-effort; чтение при ошибке SQL логирует и отдаёт
пустой результат или пробрасывает только если store закрыт — веб
трактует как «недоступно». Альтернатива — второе sqlite3-соединение в
веб-модуле — отвергнута решением разведки.

### D4. HTML-страницы + JSON API для poll
Сервер отдаёт HTML-оболочки (`/`, `/runs`, `/runs/{id}`, `/audit`) и
JSON endpoints (`/api/summary`, `/api/runs`, `/api/runs/{id}`, `/api/audit`)
с query `label`/`since`. Клиентский JS (~десяток строк) делает
`fetch` каждые ~1500 мс и перерисовывает блоки. Выбранный `run_id`
хранится в URL/query, чтобы poll не сбрасывал выбор. Альтернатива —
полный SSR на каждый poll — отвергнута (мигание, лишний HTML).

### D5. Аудит в store, семантика как у obs-audit
Логику Q1–Q5 (в т.ч. session history vs user на turn 1, repeats,
effective cost) перенести в read-методы store (или чистые функции,
вызываемые store), чтобы веб и будущие читатели не расходились с CLI.
CLI в этом change не обязан перейти на store (остаётся файл-ориентированным);
семантика веб-аудита сверяется тестами с эталоном CLI/спеки token-audit.

### D6. Конфиг порта
Дефолт `8765`; переопределение через env (например `OBS_WEB_PORT`) для
занятого порта без правки кода. Bind всегда `127.0.0.1` (не конфигурируется
в v1 — меньше шансов случайно открыть LAN).

### D7. Модуль `obs_web` в пакете
`src/dev_helper_bot/obs_web.py` (или подпакет): сборка app, маршруты,
шаблоны как строки/файлы рядом. Не класть в `scripts/` — это runtime бота,
не утилита.

## Risks / Trade-offs

- [Один процесс: баг в веб-хендлере теоретически влияет на loop бота] →
  тонкие хендлеры, только чтение store; тяжёлую логику аудита покрыть
  unit-тестами вне HTTP.
- [WAL: читатель в том же соединении, что писатель] → aiosqlite serializes
  на одном connection; при долгом audit-запросе возможна задержка записи.
  Митигация: простые SQL как в CLI; при проблеме позже — отдельное
  read-connection (вне v1).
- [Fail-fast порта vs «наблюдение не роняет агента»] → осознанный trade-off:
  запись телеметрии soft, HTTP-конфиг hard; документировать в README.
- [Дублирование семантики аудита CLI vs store] → тесты на одинаковые
  фикстуры БД; в follow-up можно перевести CLI на store.

## Migration Plan

Новых миграций схемы БД нет. Откат: убрать старт веб-app из `main`;
read-методы store безопасно оставить. CLI не затрагивается.

## Open Questions

Нет — решения разведки закрывают порт, bind, poll, UX выбора run и fail-режимы.
