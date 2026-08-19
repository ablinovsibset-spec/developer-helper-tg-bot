## Why

Курсовое задание требует простого Telegram-бота, который пересылает текст пользователя в локальную LLM и возвращает ответ. Горизонт обучения — 3 месяца, значит бот будет жить и обрастать (память, команды, переключение моделей), поэтому фундамент должен быть расширяемым с первого дня, а не переписываться позже. Дополнительно задание явно поощряет понимание устройства Bot API и изоляцию доступа к LLM, чтобы поставщика можно было сменить без правок бота.

## What Changes

- Новый бот на Python + aiogram 3.x (async), long-polling через `Dispatcher.start_polling`.
- Приём текстового сообщения пользователя → запрос к LLM → возврат ответа в чат.
- Одноразовый контекст: каждое сообщение обрабатывается как независимый запрос, история не сохраняется.
- Доступ к LLM вынесен в отдельный модуль за швом `LLMClient` (`complete(messages) -> str`); `bot.py` не знает про конкретного поставщика.
- Реализация `OpenAICompatibleClient` покрывает LM Studio, Ollama (`/v1`), vLLM, OpenAI и др. — смена поставщика в 90% случаев = правка `.env`.
- Конфигурация и секреты (`TELEGRAM_BOT_TOKEN`, `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`) только через `.env`, никогда в коде и в git.
- Чанкинг ответа по 4096 символов (лимит Telegram).
- Graceful fallback: если LLM-сервер недоступен или модель не загружена, бот отвечает человекочитаемым сообщением, не роняя поллинг.

## Capabilities

### New Capabilities
- `telegram-bot`: приём текстовых сообщений в Telegram, one-shot пересылка в LLM, возврат ответа с чанкингом по 4096 символов.
- `llm-provider`: шов `LLMClient` с контрактом `complete(messages) -> str`, фабрика по конфигу, swappable поставщики (OpenAI-compatible), graceful fallback при недоступности.

### Modified Capabilities
<!-- Нет существующих спек — проект новый. -->

## Impact

- Новый код: `bot.py`, `llm/` (`base.py`, `openai_compat.py`, `__init__.py`), `config.py`, `.env.example`, `requirements.txt`.
- Зависимости: `aiogram` (3.x), `python-dotenv`; `aiohttp` подтягивается транзитивно через aiogram и используется для вызовов LLM.
- Внешние сервисы: Telegram Bot API (long-polling), локальный LM Studio сервер (`http://localhost:1234/v1`, модель `Llama-3.2-3B-Instruct-4bit`) — должен быть запущен вручную до старта бота.
- Секреты: `TELEGRAM_BOT_TOKEN` и конфиг LLM в `.env` (вне git, уже покрыто `.gitignore`).
- Не затрагивает: ничего за пределами нового проекта (репо пустой, кроме `explore/homework.md`).
