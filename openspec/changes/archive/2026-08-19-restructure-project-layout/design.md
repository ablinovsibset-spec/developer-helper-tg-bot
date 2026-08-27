## Context

Текущая структура (см. proposal.md — Why): плоско лежащие `bot.py`, `config.py` и пакет `llm/` в корне репозитория; нет `pyproject.toml`, нет тестов. Импорты внутри кода используют корневые имена: `from config import make_llm, telegram_token`, `from llm import LLMClient, Message, LLMUnavailable`, `from llm.openai_compat import OpenAICompatibleClient`. Точка входа — `python bot.py`. Секреты в `.env`, `.gitignore` уже игнорирует `.env` и `__pycache__/`.

## Goals / Non-Goals

**Goals:**
- Перенести код в `src/dev_helper_bot/` с сохранением наблюдаемого поведения бота и контракта `LLMClient`.
- Сделать пакет устанавливаемым через `pip install -e .` и запускаемым через `python -m dev_helper_bot.main`.
- Подготовить каркас `tests/` (с `conftest.py`, `unit/`, `integration/`) и `scripts/` для будущего наполнения.
- Ввести `pyproject.toml` как источник истины о пакете, сохранив `requirements.txt` для удобной установки.

**Non-Goals:**
- Не реализуем сами тесты — только каркас (заглушки `__init__.py`, `conftest.py`).
- Не реализуем служебные скрипты — только каталог с `.gitkeep`.
- Не наполняем `models.py`, `services.py`, `utils.py` бизнес-логикой — оставляем пустыми/заглушками.
- Не вводим автоматизацию (Makefile / tasks.py) — отложено по решению пользователя.
- Не меняем поведение бота и не трогаем спеки `llm-provider` и `telegram-bot`.

## Decisions

### Решение 1: src-layout с пакетом `dev_helper_bot`
Используем `src/`-layout (`src/dev_helper_bot/`) вместо flat-layout. Это стандарт для защищённого импорта: тесты и сторонние скрипты получают пакет только после `pip install -e .`, что исключает случайный импорт из корня.

**Альтернативы:** flat-layout (пакет прямо в корне). Отвергнута: src-layout лучше выявляет «протечки» импортов и соответствует заданию.

### Решение 2: Имя пакета — `dev_helper_bot`
Имя пакета и директории под `src/` — `dev_helper_bot` (PEP 8: нижний регистр, подчёркивания). Имя дистрибутива в `pyproject.toml` — `dev-helper-bot` (дефисы для PyPI-стиля), console script — `dev-helper-bot`.

### Решение 3: Перенос модулей
- `bot.py` → `src/dev_helper_bot/main.py` (точка входа/фасад).
- `config.py` → `src/dev_helper_bot/config.py`.
- `llm/` → `src/dev_helper_bot/llm/` (подпакет, файлы переносятся как есть: `__init__.py`, `base.py`, `openai_compat.py`).
- Новые пустые заглушки: `src/dev_helper_bot/models.py`, `services.py`, `utils.py`.

**Альтернатива:** слить `llm/` в `services.py`/`utils.py`. Отвергнута по решению пользователя — сохраняем изоляцию LLM как подпакет.

### Решение 4: Обновление импортов
Внутрипакетные импорты переключаются на абсолютные с именем пакета:
- `from config import …` → `from dev_helper_bot.config import …`
- `from llm import …` → `from dev_helper_bot.llm import …`
- `from llm.openai_compat import …` → `from dev_helper_bot.llm.openai_compat import …`
- В `llm/__init__.py` убрать обратный импорт `from config import make_llm` (он создаёт цикл через корень) — `make_llm` живёт в `dev_helper_bot.config`; `llm` больше не реэкспортирует фабрику.

### Решение 5: Точка входа
- `main.py` остаётся исполняемым через `if __name__ == "__main__": asyncio.run(main())` — поддерживает `python -m dev_helper_bot.main`.
- В `pyproject.toml` объявляем `[project.scripts]` `dev-helper-bot = "dev_helper_bot.main:main"`. Так как `main` — корутина, оборачиваем в синхронную обёртку `def cli(): asyncio.run(main())` и ссылаемся на `cli` в entry point.

### Решение 6: pyproject.toml + requirements.txt
`pyproject.toml` описывает метаданные, `[build-system]` (setuptools), `[project]` с `dependencies` и `optional-dependencies`, и `[tool.setuptools.packages.find]` с `where = ["src"]`. `requirements.txt` оставляем как плоский список для `pip install -r requirements.txt`; дублирует `dependencies` из `pyproject.toml`. Это компромисс между удобством и единственным источником истины — выбираем удобство (задание явно допускает оба файла).

### Решение 7: Каркас тестов
- `tests/__init__.py`, `tests/conftest.py` (пустой, с заголовком-комментарием о назначении).
- `tests/unit/__init__.py`, `tests/integration/__init__.py`.
- pytest-настройки в `pyproject.toml` (`[tool.pytest.ini_options]` с `pythonpath` не нужен при editable-установке; `testpaths = ["tests"]`).

### Решение 8: scripts/ через .gitkeep
`scripts/.gitkeep` — единственный файл в каталоге, чтобы git отслеживал пустую директорию.

## Risks / Trade-offs

- **[Дублирование зависимостей]** между `pyproject.toml` и `requirements.txt` может разойтись. → Mitigation: в `requirements.txt` комментарием указать, что источник истины — `pyproject.toml`; при расхождении править оба.
- **[Циклический импорт]** если оставить в `llm/__init__.py` реэкспорт `make_llm` через `from config import …`, после переноса возникнет цикл. → Mitigation: убрать реэкспорт из `llm/__init__.py`; `make_llm` импортировать напрямую из `dev_helper_bot.config`.
- **[Сломанный запуск у пользователя]** после переноса старая команда `python bot.py` перестанет работать. → Mitigation: обновить `README.md` с новой командой; в `tasks.md` явно отметить шаг проверки запуска.
- **[__pycache__ в корне]** старые кэши могут остаться. → Mitigation: удалить `__pycache__/` в корне и в `llm/` в рамках переноса.

## Migration Plan

1. Создать `src/dev_helper_bot/` и `src/dev_helper_bot/llm/`, перенести файлы (git mv для сохранения истории).
2. Добавить `__init__.py`, заглушки `models.py`/`services.py`/`utils.py`.
3. Обновить импорты в перенесённых файлах; убрать циклический реэкспорт в `llm/__init__.py`.
4. Создать `pyproject.toml`, `tests/`-каркас, `scripts/.gitkeep`.
5. Обновить `README.md` и при необходимости `.gitignore`.
6. `pip install -e .` и `pytest` (проверка коллекции), `python -m dev_helper_bot.main` (проверка запуска — с заглушкой токена).
7. Коммит.

**Откат:** удалить новую структуру и вернуть исходное состояние из git (перенос делается одним коммитом, откат — `git revert`/`git reset`).

## Open Questions
<!-- Отложенных неизвестных, влияющих на спеки или разбиение задач, нет. -->
