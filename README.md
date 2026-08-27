# developer-helper-tg-bot

Telegram-бот, который пересылает сообщения пользователя локальной LLM и возвращает ответ модели.

## Требования

- Python 3.11+
- Запущенный LLM-сервер с OpenAI-совместимым API:
  - [LM Studio](https://lmstudio.ai/) (по умолчанию `http://localhost:1234/v1`), или
  - [Ollama](https://ollama.com/) (`http://localhost:11434/v1`, включите переменную окружения `OLLAMA_HOST` при необходимости), или
  - любой облачный провайдер (OpenAI, vLLM и т.п.)

## Установка

```bash
git clone <url-репозитория>
cd developer-helper-tg-bot

python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e .               # вариант 1: editable-установка (рекомендуется)
# или
pip install -r requirements.txt  # вариант 2: только зависимости
```

## Настройка

1. Скопируйте `.env.example` в `.env`:

   ```bash
   cp .env.example .env
   ```

2. Заполните `TELEGRAM_BOT_TOKEN` — токен из [@BotFather](https://t.me/BotFather).
3. При необходимости скорректируйте параметры LLM:

   | Переменная | По умолчанию | Описание |
   |---|---|---|
   | `LLM_PROVIDER` | `openai_compatible` | Провайдер (LM Studio, Ollama /v1, vLLM, OpenAI) |
   | `LLM_BASE_URL` | `http://localhost:1234/v1` | Адрес OpenAI-совместимого endpoint'а |
   | `LLM_MODEL` | `Llama-3.2-3B-Instruct-4bit` | Имя модели, обслуживаемой backend'ом |
   | `LLM_API_KEY` | — | Для облачных провайдеров; для локальных LM Studio / Ollama оставьте пустым |

## Запуск LLM-backend'а

Перед запуском бота поднимите локальный LLM-сервер:

- **LM Studio**: загрузите модель во вкладке *Developer* и нажмите *Start Server* (порт `1234`).
- **Ollama**:

  ```bash
  brew install ollama        # macOS; или скачайте с ollama.com
  ollama pull llama3.2
  ollama serve
  ```

  Тогда в `.env` укажите `LLM_BASE_URL=http://localhost:11434/v1` и `LLM_MODEL=llama3.2`.

## Запуск бота

```bash
# Через модуль
python -m dev_helper_bot.main

# Через console entry point (после pip install -e .)
dev-helper-bot
```

Бот работает через long-polling — публичный адрес и SSL не нужны. Найдите своего бота в Telegram и отправьте ему сообщение.

## Тесты

```bash
pip install pytest pytest-asyncio
# или, с pip >= 25.1: pip install -e . --group dev
pytest
```

## Структура проекта

```
src/dev_helper_bot/   исходный код пакета
  main.py             точка входа/фасад бота
  config.py           настройки и фабрика LLM
  llm/                подпакет доступа к LLM
tests/                тесты (unit/, integration/)
scripts/              служебные скрипты
pyproject.toml        главный файл конфигурации проекта
```
