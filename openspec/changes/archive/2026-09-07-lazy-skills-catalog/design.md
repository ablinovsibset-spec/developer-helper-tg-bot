## Context

См. proposal.md — Why. Сейчас `load_skills` читает `skills/*.md` как
`stem → полный текст`, `build_system_prompt` клеит `## {name}\n{content}`
для каждого; отдельного инструмента активации нет (спека skills это
запрещала). Формат файлов выбран **A** (плоский `.md` + frontmatter).
Инструменты живут в `tools.py` + диспетчер `execute_tool_call` в
`agent.py`; скиллы грузятся один раз в `main`/`obs-benchmark` и
передаются в сборку запроса. PyYAML в зависимостях нет.

## Goals / Non-Goals

**Goals:**
- Catalog-only system prompt; тела — через `get_skill`.
- Fail-fast валидация frontmatter при старте.
- Сохранить стабильный кэшируемый префикс (каталог стабилен между
  сообщениями).
- Повторный замер харнессом для вклада в −30% observability.

**Non-Goals:**
- Layout Cursor (`skills/<name>/SKILL.md`), `reference.md`, scripts.
- Поля Cursor вроде `disable-model-invocation`.
- Горячая перезагрузка скиллов без рестарта.
- Принуждение `tool_choice` на `get_skill` (решение за моделью).
- Компакция dialog-истории.

## Decisions

### D1. Плоский файл + frontmatter (вариант A)
Оставляем `skills/<stem>.md`. Канон: `frontmatter.name == path.stem`.
Альтернатива B (каталоги + SKILL.md) отвергнута: четыре коротких скилла
без вложенных reference; лишняя миграция путей в README/спеке.

### D2. Модель данных Skill
`Skill(name: str, description: str, body: str)`. `load_skills` →
`dict[str, Skill]` по `name`. `build_system_prompt` принимает этот
словарь и рендерит catalog, не `body`. Тела для tool — lookup по имени.

### D3. Парсер frontmatter без новой зависимости
Разбор: документ начинается с `---\n` … `\n---\n`; между ними — YAML
только с ключами `name` и `description`. Реализация: минимальный
разборщик под эти два ключа (однострочные значения и `description: |` /
`>-` блок), либо `yaml.safe_load` **только если** уже появится PyYAML
по другой причине. Предпочтение — stdlib-only, без PyYAML.
Альтернатива «весь файл = body, description из первой строки» отвергнута:
нужен явный контракт name/description как в Agent Skills.

### D4. Формат catalog в system
После env-строк (reasoning, sandbox, memory) — секция вида:

```text
Скиллы: ниже каталог (name — description). Если description подходит
к запросу — вызови get_skill(name) и следуй телу. Не выдумывай шаги
скилла без загрузки.
- morning: …
- wttr-in-api: …
```

Порядок — sorted by name (детерминизм префикса). Пустой словарь — секции
нет. Альтернатива «JSON catalog» отвергнута: хуже читается слабыми
моделями; список маркеров достаточно.

### D5. get_skill — отдельный tool, in-memory
По образцу `read_file`: `GET_SKILL_TOOL_SPEC` + функция, принимающая
`Mapping[str, Skill]` (или Protocol). Не ходит в песочницу и не читает
диск на каждый вызов — согласованность с catalog при старте.
Диспетчер в `execute_tool_call`; регистрация в main и obs-benchmark.
Альтернатива «модель делает read_file по пути skills/…» отвергнута:
скиллы на хосте, не в контейнере; read_file — про песочницу.

### D6. Ошибки — текстом в tool result
Как exec/read_file: неизвестное имя / пустой аргумент → строка ошибки,
цикл живёт. Не бросать из диспетчера.

### D7. Descriptions (черновик содержания файлов)
Писать в третьем лице, WHAT + WHEN (триггеры). Ориентиры:

- `wttr-in-api` — погода через curl/wttr.in; когда просят погоду или
  другой скилл ссылается на wttr-in-api.
- `habr-feed` — топ Habr за 24ч через exec; когда нужен дайджест Habr
  или morning ссылается на habr-feed.
- `morning` — утренняя сводка (погода Минск → Habr → ≤4000); когда
  «доброе утро» / «утренняя сводка» / «что на сегодня»; сначала
  get_skill morning, затем связанные.
- `search-history` — обзор→поиск по прошлым беседам; когда ссылаются на
  прошлое (дополняет MEMORY_ENV_LINE деталями).

Тела: минимальные правки ссылок «см. секцию X» → «get_skill(X)».

### D8. Fail fast при старте
Невалидный файл → исключение/exit при загрузке в main (и в харнессе),
с путём файла и причиной. Пустой/отсутствующий каталог — по-прежнему OK.

### D9. Замер после внедрения
Харнесс с новой меткой (например `lazy-skills`), сравнение с `optimized`
и/или `benchmark` через obs-audit; цифры — в task-отчёт (gitignore
`task/` ок). Цель −30% от исходного baseline по сырым input — критерий
успеха change для observability, не блокер юнит-тестов.

## Risks / Trade-offs

- [Модель не вызывает get_skill] → сильные WHEN в description + явная
  инструкция в catalog; morning-сценарий в бенчмарке/ручной проверке;
  не используем tool_choice (non-goal).
- [Лишний ход съедает MAX_LLM_STEPS=8] → morning: +1–3 get_skill; бюджет
  шагов обычно достаточен; мониторить steps_exhausted в замере.
- [Слабый YAML-парсер ломает description] → зафиксировать в скиллах
  простой стиль (одна строка или `|` блок); тест на парсер.
- [Префикс короче → меньше cache write, но тело в mid-context] → на
  routerai/flex выигрыш по входу на каждом ходе важнее; cache по-прежнему
  на стабильном system+history до tool results.

## Migration Plan

Атомарно в одном PR: новые файлы скиллов с frontmatter + код загрузки +
tool. Откат — вернуть склейку полных тел (старый `build_system_prompt`)
и убрать tool из списка. Данных БД не трогаем. После merge — прогон
харнесса с меткой lazy-skills и короткий отчёт в `task/`.

## Open Questions

Нет блокирующих: формат A, канон name==stem, stdlib-парсер и in-memory
get_skill зафиксированы выше.
