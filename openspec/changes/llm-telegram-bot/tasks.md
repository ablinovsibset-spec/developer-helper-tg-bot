## 1. Каркас проекта

- [ ] 1.1 Создать `requirements.txt` с `aiogram` (3.x) и `python-dotenv`
- [ ] 1.2 Создать `.env.example` с `TELEGRAM_BOT_TOKEN`, `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` (с дефолтами для LM Studio)
- [ ] 1.3 Проверить, что `.env` уже покрыт `.gitignore` (репозиторий не пушит секреты)

## 2. Модуль LLM (шов LLMClient)

- [ ] 2.1 Создать пакет `llm/` с `__init__.py`, экспортирующим `make_llm`, `LLMClient`, `LLMUnavailable`
- [ ] 2.2 В `llm/base.py` определить `LLMClient` (Protocol с методом `async complete(messages) -> str`) и исключение `LLMUnavailable`
- [ ] 2.3 В `llm/openai_compat.py` реализовать `OpenAICompatibleClient`: async-метод `complete` через aiohttp к `/v1/chat/completions`, параметризация `base_url`/`model`/`api_key`, таймаут, возбуждение `LLMUnavailable` при ошибках соединения/HTTP/таймаута
- [ ] 2.4 В `config.py` реализовать `make_llm()`: чтение env, фабрика по `LLM_PROVIDER` (дефолт `openai_compatible`), явная ошибка при неизвестном поставщике, дефолты для LM Studio

## 3. Бот (aiogram)

- [ ] 3.1 В `bot.py` реализовать загрузку `.env` и инициализацию `Bot`/`Dispatcher` из `TELEGRAM_BOT_TOKEN`; понятная ошибка при отсутствии токена
- [ ] 3.2 Реализовать хендлер `@dp.message(F.text)`: формирование `messages=[{role:user, content:text}]`, вызов `await llm.complete(...)`, отправка ответа
- [ ] 3.3 Реализовать `send_chunked`: разбиение ответа длиннее 4096 символов на последовательные сообщения
- [ ] 3.4 Обернуть вызов LLM в try/except `LLMUnavailable`: отправка человекочитаемого сообщения об ошибке, поллинг продолжается
- [ ] 3.5 Запустить поллинг через `dp.start_polling(bot)` в `asyncio.run(main())`

## 4. Проверка

- [ ] 4.1 Поднять LM Studio сервер с моделью `Llama-3.2-3B-Instruct-4bit`, заполнить `.env` реальным `TELEGRAM_BOT_TOKEN`
- [ ] 4.2 Запустить `python bot.py` и отправить тестовое сообщение в Telegram — проверить возврат ответа LLM
- [ ] 4.3 Проверить one-shot: отправить два сообщения подряд — второе не должно содержать контекст первого
- [ ] 4.4 Проверить чанкинг: спровоцировать длинный ответ (>4096 символов) — проверить приход нескольких сообщений
- [ ] 4.5 Проверить graceful fallback: остановить LM Studio сервер, отправить сообщение — бот отвечает сообщением о недоступности и продолжает поллинг; перезапустить сервер, отправить сообщение — ответ снова приходит
