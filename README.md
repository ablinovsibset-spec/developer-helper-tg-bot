# developer-helper-tg-bot

Telegram-бот, который пересылает сообщения пользователя локальной LLM и возвращает ответ модели.

## Установка

```bash
# Вариант 1: editable-установка (рекомендуется для разработки)
pip install -e .

# Вариант 2: только зависимости
pip install -r requirements.txt
```

## Настройка

1. Скопируйте `.env.example` в `.env`.
2. Заполните `TELEGRAM_BOT_TOKEN` (получите у [@BotFather](https://t.me/BotFather)).
3. При необходимости скорректируйте параметры LLM (`LLM_BASE_URL`, `LLM_MODEL`).

## Запуск

```bash
# Через модуль
python -m dev_helper_bot.main

# Через console entry point (после pip install -e .)
dev-helper-bot
```

## Структура проекта

```
src/dev_helper_bot/   исходный код пакета
  main.py             точка входа/фасад бота
  config.py           настройки и фабрика LLM
  llm/                подпакет доступа к LLM
tests/                каркас тестов (unit/, integration/)
scripts/              служебные скрипты
pyproject.toml        главный файл конфигурации проекта
```
