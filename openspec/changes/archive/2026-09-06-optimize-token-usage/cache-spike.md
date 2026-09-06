# Спайк D7: отдаёт ли LM Studio cached_tokens

- **Дата:** 2026-09-06
- **Модель:** gpt-oss-20b в LM Studio (localhost:1234), endpoint /v1/chat/completions
- **Метод:** два идентичных живых запроса подряд (одинаковый префикс ~114
  prompt-токенов, temperature 0), инспекция `usage` в ответе.

## Результат

`usage` содержит только:

```json
{
  "prompt_tokens": 114,
  "completion_tokens": 19,
  "total_tokens": 133,
  "completion_tokens_details": {"reasoning_tokens": 16}
}
```

Поля `prompt_tokens_details` нет ни в первом, ни во втором (повторном) запросе —
LM Studio не отдаёт `cached_tokens` вообще. В телеметрии `cached_tokens`
останется NULL («поставщик не отдал»), как и в baseline.

## Следствие для задачи (design D7)

- Кэш-скидка в аудите (obs-audit) реализуется, но для этой инсталляции
  эффективная стоимость будет равна сырой с честной пометкой «данные о кэше
  недоступны» — спека token-audit это явно разрешает.
- На цель −30% (по сырым входным токенам) не влияет.
