## 1. Инфраструктура тестов

- [x] 1.1 Добавить в `pyproject.toml` dev-зависимости (`pytest`, `pytest-asyncio`) через `[dependency-groups]` и настроить `[tool.pytest.ini_options]`: `asyncio_mode = "auto"`
- [x] 1.2 Установить dev-зависимости в окружение, убедиться что `pytest` из корня запускается и находит пустой набор без ошибок

## 2. Unit-тесты: обработчик бота (`tests/unit/test_main.py`)

- [x] 2.1 Добавить fake-объекты (`FakeBot` с записью `send_message`, `FakeMessage` с `.text`/`.chat.id`, `FakeLLM` с управляемым ответом/ошибкой) в `tests/conftest.py` или модуль теста
- [x] 2.2 Тест: текстовое сообщение → LLM получает один user-message с текстом, ответ уходит в тот же чат
- [x] 2.3 Тест: `LLMUnavailable` от LLM → в чат уходит человекочитаемое сообщение об ошибке, исключение не всплывает
- [x] 2.4 Тест `send_chunked`: 200 символов → одно сообщение; 9000 символов → три сообщения 4096+4096+808 в порядке текста

## 3. Unit-тесты: конфигурация (`tests/unit/test_config.py`)

- [x] 3.1 Тест `make_llm`: переменные `LLM_*` не заданы → OpenAI-совместимая реализация с дефолтными base_url и model (через `monkeypatch`)
- [x] 3.2 Тест `make_llm`: неизвестный `LLM_PROVIDER` → `ValueError` с понятным сообщением
- [x] 3.3 Тест `telegram_token`: токен не задан → `SystemExit` с сообщением про `.env` и BotFather

## 4. Integration-тесты: LLM-провайдер (`tests/integration/test_openai_compat.py`)

- [x] 4.1 Фикстура фейкового LLM-сервера: aiohttp-сервер на localhost (порт 0) с управляемым статусом/телом/задержкой и записью входящих запросов (включая заголовки)
- [x] 4.2 Тест: успешный ответ сервера → `complete` возвращает текст из `choices[0].message.content`; запрос содержит корректный JSON-payload
- [x] 4.3 Тест заголовка авторизации: с `api_key` → `Authorization: Bearer <key>`; без → заголовка нет
- [x] 4.4 Тест ошибок: HTTP 500 → `LLMUnavailable`; битый JSON (нет `choices`) → `LLMUnavailable`; задержка сервера при `timeout=0.2` → `LLMUnavailable` с сообщением о таймауте

## 5. Чистка неиспользуемых файлов

- [x] 5.1 Проверить grep-ом отсутствие импортов `dev_helper_bot.models` / `dev_helper_bot.services` / `dev_helper_bot.utils` в коде и тестах
- [x] 5.2 Удалить пустые заглушки `src/dev_helper_bot/models.py`, `services.py`, `utils.py`
- [x] 5.3 Удалить локальный мусор: корневой `__pycache__/` (байткод старой структуры), `src/dev_helper_bot.egg-info/`, `.pytest_cache/`

## 6. Верификация

- [x] 6.1 Полный прогон `pytest` из корня — зелёный, без внешних сервисов и секретов (переменные `TELEGRAM_BOT_TOKEN`/`LLM_*` не заданы)
- [x] 6.2 Переустановка `pip install -e .` и запуск импорта `python -c "import dev_helper_bot"` — пакет жив после удаления заглушек
- [x] 6.3 `openspec validate create-tests` — без ошибок
