## 1. Парсинг и catalog-промпт

- [x] 1.1 Ввести `Skill(name, description, body)` и парсер frontmatter (stdlib, design D3): `---` блок с `name`/`description`, тело после; ошибка при отсутствии полей или `name` ≠ `stem`
- [x] 1.2 Переписать `load_skills` → `dict[str, Skill]`; пустой/отсутствующий каталог по-прежнему `{}`
- [x] 1.3 `build_system_prompt`: env-строки + catalog name+description + инструкция `get_skill` (design D4); тела не включать; обновить `build_request_messages` под новый тип
- [x] 1.4 Unit-тесты: парсер (ok / нет description / name≠stem); catalog в промпте без тел; стабильность префикса без datetime; пустой каталог

## 2. Инструмент get_skill

- [x] 2.1 `GET_SKILL_TOOL_SPEC` + реализация lookup по in-memory словарю; неизвестное/пустое имя — текст ошибки без исключения (specs/get-skill-tool)
- [x] 2.2 Подключить в `execute_tool_call`, наборы tools в `main.py` и `obs-benchmark.py`; передать загруженные скиллы в шов исполнения
- [x] 2.3 Unit-тесты: успешное тело без frontmatter; неизвестное имя; согласованность с `load_skills`

## 3. Файлы скиллов

- [x] 3.1 Добавить frontmatter (name + description WHAT/WHEN, design D7) в `wttr-in-api.md`, `habr-feed.md`, `morning.md`, `search-history.md`
- [x] 3.2 В `morning.md` заменить отсылки к «секциям» на загрузку `wttr-in-api` / `habr-feed` через `get_skill`; тела без лишнего раздувания

## 4. Тесты интеграции и документация

- [x] 4.1 Обновить/добавить тесты под спеку test-suite: промпт содержит имена+description, не тела; цикл с вызовом `get_skill`
- [x] 4.2 Прогнать `pytest` (без docker) — зелёный
- [x] 4.3 Обновить README: формат frontmatter, catalog в system, инструмент `get_skill`

## 5. Замер observability

- [x] 5.1 Прогон харнесса с меткой `lazy-skills` (достаточно для сравнения с `optimized`/`benchmark`)
- [x] 5.2 Сравнение через `obs-audit`: сырые input-токены, success rate; краткий отчёт в `task/` (gitignore ок)
- [x] 5.3 Если −30% от baseline всё ещё не достигнуто — зафиксировать цифры и остаточный зазор в отчёте (без молчаливого «успеха»)
