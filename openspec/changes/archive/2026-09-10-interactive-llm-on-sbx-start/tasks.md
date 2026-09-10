## 1. Interactive prompts in sbx-start

- [x] 1.1 В `scripts/sbx-start.sh` перед запуском бота добавить запросы `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` с дефолтами LM Studio (`http://host.docker.internal:${LLM_PORT:-1234}/v1`), `openai/gpt-oss-20b`, пустой ключ; Enter принимает дефолт; если переменная уже задана в окружении вызывающего shell — показать её как pre-fill (design D2)
- [x] 1.2 Для API key использовать скрытый ввод; в итоговых `say` печатать URL и model, для ключа — только `set`/`empty` (design D5)
- [x] 1.3 Передавать в `sbx exec` все три `-e LLM_BASE_URL`, `-e LLM_MODEL`, `-e LLM_API_KEY` (пустой ключ — явно), плюс существующий `OBS_WEB_HOST` (design D3); `.env` не трогать

## 2. Docs

- [x] 2.1 Обновить README: ежедневный запуск с интерактивным confirm, Enter×3 = LM Studio, пример ввода облачных значений с клавиатуры (design D6)

## 3. Verify

- [x] 3.1 Ручная проверка: `scripts/sbx-start.sh` + Enter×3 → в логе/поведении LM Studio + `openai/gpt-oss-20b`; `.env` не изменился
- [x] 3.2 Ручная проверка: ввод облачного URL/модели/ключа → бот ходит в облако; повторный старт с Enter×3 снова локальный стек (ключ из `.env` не подхватывается)
- [x] 3.3 `pytest` (hermetic) — регрессия не ожидается; прогон для уверенности
