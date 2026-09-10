## Why

При `scripts/sbx-start.sh` адрес LLM всегда молча подменяется на LM Studio (`host.docker.internal:1234`), а модель и ключ берутся из workspace-`.env` без вопросов. Если в `.env` уже лежит облачный endpoint (routerai и т.п.), получается рассинхрон: локальный URL + чужие model/key. Нужен явный интерактивный выбор LLM на каждый запуск, с безопасным дефолтом на LM Studio.

## What Changes

- `scripts/sbx-start.sh` перед запуском бота интерактивно запрашивает `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`.
- Дефолты в промптах: LM Studio внутри VM (`http://host.docker.internal:1234/v1`), модель `openai/gpt-oss-20b`, пустой API key; Enter принимает дефолт.
- Выбранные значения передаются в процесс бота только через `-e` на этот запуск; workspace-`.env` скрипт **не** переписывает.
- README: ежедневный запуск описывает интерактивный confirm и примеры смены на облачный endpoint с клавиатуры.
- Тихий режим (`-y` / non-TTY) **не** вводится — скрипт рассчитан на запуск из терминала человеком.

## Capabilities

### New Capabilities

- (нет)

### Modified Capabilities

- `sbx-hosting`: скрипт запуска SHALL запрашивать URL/модель/ключ LLM интерактивно с дефолтами LM Studio; override на процесс — через env запуска, без записи в `.env`; сценарий «ежедневный запуск» уточняется под интерактивный confirm.

## Impact

- **Скрипты**: `scripts/sbx-start.sh` (промпты + проброс трёх `LLM_*` через `sbx exec -e`).
- **Документация**: README (раздел ежедневного запуска / override LLM).
- **Не меняется**: `config.py` / `make_llm()` (по-прежнему читают env; dotenv не перекрывает уже заданные `-e`), граница доверия `.env`, `sbx-setup.sh`, Python-код бота.
- **Спека**: уточнение требования runbook / связи с LLM в `sbx-hosting`; сценарий «одна команда» → команда + Enter×3 для дефолтов.
- **Тесты**: hermetic pytest не затрагивается (скрипт вне unit-набора); ручная проверка runbook.
