## Why

Проект сейчас разложен плоско: `bot.py` и `config.py` лежат в корне репозитория, а пакет `llm/` — рядом с ними. Нет `pyproject.toml`, нет тестов, нет служебных скриптов. Такая структура мешает установке пакета (`pip install -e .`), импорту из тестов без хаков с `sys.path` и масштабированию в рамках трёхмесячного горизонта обучения. Задание `task/01-project-structure.md` требует привести структуру к стандартному Python-проекту с `src/`-layout, зеркальными тестами и `pyproject.toml`.

## What Changes

- **BREAKING**: Перенести исходники из корня в `src/dev_helper_bot/` (`src/`-layout): `bot.py` → `dev_helper_bot/main.py`, `config.py` → `dev_helper_bot/config.py`, пакет `llm/` → `dev_helper_bot/llm/`.
- Добавить `src/dev_helper_bot/__init__.py` и `src/dev_helper_bot/llm/__init__.py` как полноценные пакеты.
- Добавить пустые модули-заглушки `dev_helper_bot/models.py`, `services.py`, `utils.py` под будущую бизнес-логику (содержимое не переносится — его пока нет).
- Ввести `pyproject.toml` как главный файл конфигурации проекта (метаданные, зависимость от `requirements.txt` через `dependencies` или `pip install -e .`), сохранив `requirements.txt` для удобной установки зависимостей.
- Добавить каталог `tests/` с `__init__.py`, `conftest.py` и подкаталогами `unit/`, `integration/`, зеркальными структуре `src/`.
- Добавить каталог `scripts/` для служебных скриптов (миграции, заполнение данными) — пока пустой с `.gitkeep`.
- Обновить `.gitignore` (при необходимости) и `README.md` с актуальными инструкциями по установке и запуску из новой структуры.
- Обновить точку входа: запуск бота через `python -m dev_helper_bot.main` или консольный entry point из `pyproject.toml`.

## Capabilities

### New Capabilities
- `project-structure`: описывает организацию репозитория по `src/`-layout: пакет `dev_helper_bot` под `src/`, зеркальные тесты в `tests/`, служебные скрипты в `scripts/`, главный файл конфигурации `pyproject.toml` и точка входа для запуска бота.

### Modified Capabilities
<!-- Существующие capabilities (llm-provider, telegram-bot) описывают поведение бота и LLM. Их требования не меняются — перенос файлов не затрагивает наблюдаемое поведение. -->

## Impact

- **Код**: `bot.py`, `config.py`, `llm/` перемещаются; обновляются внутренние импорты (с `from config import …` / `from llm import …` на `from dev_helper_bot.config import …` / `from dev_helper_bot.llm import …`).
- **Зависимости**: появляется `pyproject.toml`; `requirements.txt` сохраняется. Установка проекта переключается на `pip install -e .`.
- **Запуск**: команда запуска бота меняется с `python bot.py` на `python -m dev_helper_bot.main` (или entry point).
- **Тесты**: появляется каркас `tests/` с `conftest.py`; до реализации тестов он остаётся пустым, но готовым к наполнению.
- **Спецификации `llm-provider` и `telegram-bot`**: поведение не меняется, дельта-спеки не требуются.
