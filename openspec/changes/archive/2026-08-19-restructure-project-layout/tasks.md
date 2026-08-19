## 1. Подготовка и перенос исходников

- [x] 1.1 Создать каталог `src/dev_helper_bot/` и `src/dev_helper_bot/llm/`
- [x] 1.2 Перенести `bot.py` → `src/dev_helper_bot/main.py` (через `git mv` для сохранения истории)
- [x] 1.3 Перенести `config.py` → `src/dev_helper_bot/config.py`
- [x] 1.4 Перенести `llm/__init__.py`, `llm/base.py`, `llm/openai_compat.py` → `src/dev_helper_bot/llm/`
- [x] 1.5 Добавить `src/dev_helper_bot/__init__.py` (пустой или с кратким docstring)
- [x] 1.6 Добавить пустые заглушки `src/dev_helper_bot/models.py`, `services.py`, `utils.py`
- [x] 1.7 Удалить старые `__pycache__/` в корне и в `llm/`

## 2. Обновление импортов и точки входа

- [x] 2.1 В `main.py` заменить `from config import …` → `from dev_helper_bot.config import …` и `from llm import …` → `from dev_helper_bot.llm import …`
- [x] 2.2 В `config.py` заменить `from llm import …` / `from llm.openai_compat import …` → `from dev_helper_bot.llm import …` / `from dev_helper_bot.llm.openai_compat import …`
- [x] 2.3 В `llm/__init__.py` убрать реэкспорт `make_llm` и импорт `from config import make_llm` (устраняет цикл); оставить экспорт `LLMClient`, `LLMUnavailable`, `Message`, `OpenAICompatibleClient`
- [x] 2.4 В `main.py` оставить `if __name__ == "__main__": asyncio.run(main())` для `python -m dev_helper_bot.main`
- [x] 2.5 Добавить синхронную обёртку `cli()` в `main.py` (`def cli(): asyncio.run(main())`) для console entry point

## 3. pyproject.toml и зависимости

- [x] 3.1 Создать `pyproject.toml` с `[build-system]` (setuptools), `[project]` (имя `dev-helper-bot`, версия, requires-python `>=3.11`, dependencies из текущего `requirements.txt`)
- [x] 3.2 Добавить `[project.scripts]` `dev-helper-bot = "dev_helper_bot.main:cli"`
- [x] 3.3 Добавить `[tool.setuptools.packages.find]` с `where = ["src"]`
- [x] 3.4 Добавить `[tool.pytest.ini_options]` с `testpaths = ["tests"]`
- [x] 3.5 Сверить `requirements.txt` с `dependencies` в `pyproject.toml` (оставить согласованное содержимое; добавить комментарий, что источник истины — `pyproject.toml`)

## 4. Каркас тестов и scripts

- [x] 4.1 Создать `tests/__init__.py`
- [x] 4.2 Создать `tests/conftest.py` (пустой с комментарием о назначении фикстур)
- [x] 4.3 Создать `tests/unit/__init__.py` и `tests/integration/__init__.py`
- [x] 4.4 Создать `scripts/.gitkeep`

## 5. Документация и .gitignore

- [x] 5.1 Обновить `README.md`: инструкции по установке (`pip install -e .` или `pip install -r requirements.txt`) и запуску (`python -m dev_helper_bot.main` или `dev-helper-bot`)
- [x] 5.2 Проверить `.gitignore`: убедиться, что `.env`, `__pycache__/`, `dist/`, `build/`, `*.egg-info/` игнорируются (большая часть уже есть из шаблона)

## 6. Проверка и коммит

- [x] 6.1 Выполнить `pip install -e .` и убедиться, что пакет `dev_helper_bot` импортируется
- [x] 6.2 Выполнить `pytest` — проверить, что коллекция запускается без ошибок импорта (тестов может не быть)
- [x] 6.3 Проверить `python -m dev_helper_bot.main` (с заглушкой `TELEGRAM_BOT_TOKEN` или ожидаемой ошибкой о отсутствующем токене)
- [x] 6.4 Проверить `git status`: `.env` не отслеживается, старых `bot.py`/`config.py`/`llm/` в корне нет
- [ ] 6.5 Закоммитить перенос одним коммитом
