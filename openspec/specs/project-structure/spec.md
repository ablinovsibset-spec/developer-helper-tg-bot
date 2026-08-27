# project-structure Specification

## Purpose

Описывает каноническую организацию репозитория проекта: исходники в `src/`-layout, пакет `dev_helper_bot`, зеркальные тесты в `tests/`, служебные скрипты в `scripts/` и главный файл конфигурации `pyproject.toml`. Гарантирует, что проект устанавливается как пакет, запускается по единой команде и готов к добавлению тестов без правок `sys.path`.

## Requirements

### Requirement: Исходный код в src-layout
Весь исходный код проекта SHALL находиться под каталогом `src/` в виде пакета `dev_helper_bot` (`src/dev_helper_bot/__init__.py`). Корень репозитория MUST NOT содержать модулей приложения верхнего уровня (`bot.py`, `config.py`), кроме точки входа, делегирующей в пакет.

#### Scenario: Импорт пакета после установки
- **WHEN** проект установлен через `pip install -e .`
- **THEN** пакет `dev_helper_bot` импортируется из любого окружения без манипуляций с `sys.path`

#### Scenario: Отсутствие модулей приложения в корне
- **WHEN** выполнен перенос исходников в `src/dev_helper_bot/`
- **THEN** в корне репозитория не остаётся `bot.py` и `config.py` как модулей приложения

### Requirement: Точка входа для запуска бота
Бот SHALL запускаться одной командой без указания пути к файлу: через `python -m dev_helper_bot.main` либо консольный entry point, объявленный в `pyproject.toml`. Прямой запуск `python bot.py` из корня больше не поддерживается.

#### Scenario: Запуск через модуль
- **WHEN** выполняется `python -m dev_helper_bot.main` в окружении с установленным пакетом
- **THEN** бот стартует и начинает long-polling Telegram

#### Scenario: Запуск через entry point
- **WHEN** в `pyproject.toml` объявлен console script (например, `dev-helper-bot`)
- **THEN** команда `dev-helper-bot` запускает бота без явного указания модуля

### Requirement: Поддерживающие модули пакета
Пакет `dev_helper_bot` SHALL содержать модули `main.py` (точка входа/фасад), `config.py` (настройки), `models.py` (модели данных), `services.py` (бизнес-логика) и `utils.py` (вспомогательные функции). Модули без текущей реализации SHALL существовать как пустые файлы-заглушки, готовые к наполнению.

#### Scenario: Наличие модулей-заглушек
- **WHEN** перенос завершён
- **THEN** в `src/dev_helper_bot/` присутствуют `models.py`, `services.py`, `utils.py` (возможно, пустые) наряду с перенесёнными `main.py` и `config.py`

### Requirement: LLM как подпакет
Доступ к LLM SHALL оставаться изолированным в подпакете `dev_helper_bot.llm` (`src/dev_helper_bot/llm/__init__.py`, `base.py`, `openai_compat.py`). Контракт `LLMClient` и поведение при недоступности провайдера не меняются — меняется только путь импорта.

#### Scenario: Импорт контракта LLM из нового расположения
- **WHEN** вызывающий код импортирует `from dev_helper_bot.llm import LLMClient, Message, LLMUnavailable`
- **THEN** импорт завершается успешно и предоставляет те же объекты, что и до переноса

#### Scenario: Изоляция поставщика сохранена
- **WHEN** меняется поставщик LLM через `.env`
- **THEN** код вне `dev_helper_bot.llm` не изменяется

### Requirement: Зеркальные тесты
Каталог `tests/` SHALL существовать и содержать `__init__.py`, `conftest.py` и подкаталоги `unit/` и `integration/`, зеркалящие структуру `src/dev_helper_bot/`. До появления тестов файлы-заглушки (`__init__.py`, `conftest.py`) SHALL существовать, чтобы каркас был готов к наполнению.

#### Scenario: Каркас тестов после переноса
- **WHEN** перенос завершён
- **THEN** существуют `tests/__init__.py`, `tests/conftest.py`, `tests/unit/`, `tests/integration/` (с `__init__.py` внутри каждого подкаталога)

#### Scenario: Запуск pytest из корня
- **WHEN** в корне выполняется `pytest` после `pip install -e .`
- **THEN** pytest обнаруживает пакет `dev_helper_bot` без правок `sys.path` и завершается без ошибок импорта (коллекция может быть пустой)

### Requirement: Каталог служебных скриптов
Каталог `scripts/` SHALL существовать на верхнем уровне репозитория для служебных скриптов (миграции БД, заполнение данными). До появления скриптов каталог SHALL сохраняться в git через `.gitkeep`.

#### Scenario: Наличие каталога scripts
- **WHEN** перенос завершён
- **THEN** в корне репозитория существует каталог `scripts/`, отслеживаемый git (через `scripts/.gitkeep`)

### Requirement: Главный файл конфигурации pyproject.toml
Проект SHALL содержать `pyproject.toml` в корне как главный файл конфигурации: метаданные пакета, точка входа, настройки сборки и инструменты. `requirements.txt` MAY сохраняться как удобный список зависимостей для установки, но источник истины о пакете — `pyproject.toml`.

#### Scenario: Установка в editable-режиме
- **WHEN** выполняется `pip install -e .` в корне репозитория
- **THEN** пакет `dev_helper_bot` устанавливается и доступен для импорта

#### Scenario: Зависимости согласованы
- **WHEN** `requirements.txt` сохранён
- **THEN** набор зависимостей в `requirements.txt` согласован с зависимостями, объявленными в `pyproject.toml`

### Requirement: Секреты и .env вне git
`.env` SHALL оставаться игнорируемым git; `.env.example` SHALL отслеживаться как шаблон. Перенос структуры MUST NOT приводить к попаданию `.env` в индекс.

#### Scenario: .env не в индексе после переноса
- **WHEN** выполнен перенос и `git status` проверен
- **THEN** `.env` не появляется как отслеживаемый файл

### Requirement: Документация запуска обновлена
`README.md` SHALL содержать актуальные инструкции по установке (`pip install -e .` или `pip install -r requirements.txt`) и запуску бота (`python -m dev_helper_bot.main` или entry point), отражающие новую структуру.

#### Scenario: README отражает новую структуру
- **WHEN** пользователь следует инструкциям в `README.md`
- **THEN** установка и запуск бота завершаются успешно из новой структуры
