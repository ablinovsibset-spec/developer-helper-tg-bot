## MODIFIED Requirements

### Requirement: Конфигурация только через окружение
Все параметры LLM (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`) SHALL читаться из переменных окружения / `.env` и MUST NOT захардкоживаться в исходном коде. Значения по умолчанию SHALL существовать в коде как дефолты для незаданных переменных и SHALL перекрываться окружением.

#### Scenario: Значения по умолчанию
- **WHEN** `LLM_BASE_URL` и `LLM_MODEL` не заданы
- **THEN** используются значения по умолчанию `https://routerai.ru/api/v1` и модель `openai/gpt-5.6-luna`

#### Scenario: Окружение перекрывает дефолт
- **WHEN** `LLM_BASE_URL` и `LLM_MODEL` заданы значениями локального сервера
- **THEN** запросы идут по заданным значениям, а не по дефолтам
