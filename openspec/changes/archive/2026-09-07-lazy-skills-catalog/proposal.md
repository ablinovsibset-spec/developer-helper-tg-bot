## Why

Тройка оптимизаций токенов (read_file, компакция tool-выводов, дата/время
вне system) не дала −30% по сырым input-токенам: средний прогон даже вырос
(~8k → ~9k), а `prompt_chars` первого хода почти не изменился (~8160) —
полное тело всех скиллов (~7k символов) по-прежнему в каждом LLM-вызове.
Зарезервированная четвёртая оптимизация — lazy skills — активируется сейчас:
в system попадают только `name` и `description`, тело — по запросу. Это же
совпадает с подходом Agent Skills (frontmatter name/description).

## What Changes

- **BREAKING** для контракта промпта: системный промпт больше не содержит
  полные тексты скиллов — только каталог `name` + `description` и инструкцию
  загрузить тело через инструмент, когда description релевантен запросу.
- Формат файлов **A**: плоские `skills/*.md` с YAML frontmatter
  (`name`, `description`) и markdown-телом после разделителя.
- Канон имени: `frontmatter.name` MUST совпадать со `stem` файла; расхождение —
  ошибка загрузки при старте (fail fast).
- Новый инструмент `get_skill(name)`: возвращает тело скилла или понятную
  ошибку «нет такого», без прерывания агентного цикла.
- Четыре существующих скилла получают frontmatter с триггерными description
  (WHAT + WHEN); тело без смысловых изменений (кроме ссылок на «секцию» →
  «скилл / get_skill»).
- Тесты и README под новый контракт каталога и `get_skill`.
- Повторный замер харнессом (метка вроде `lazy-skills`) как доказательство
  вклада в цель −30% Части 3 observability — в скоупе задач change, не
  блокирует merge кода.

## Capabilities

### New Capabilities

- `get-skill-tool`: инструмент загрузки полного тела скилла по имени из
  каталога, известного модели через system-prompt catalog.

### Modified Capabilities

- `skills`: формат frontmatter; в system — только name+description (+
  инструкция активации); тело не инлайнятся; загрузка/валидация при старте;
  сценарии утренней сводки опираются на активацию через `get_skill`.
- `test-suite`: проверки загрузки скиллов утверждают наличие имён и
  description в системном промпте (не полных тел) и покрытие `get_skill`.

## Impact

- `skills/*.md` — frontmatter у всех четырёх файлов; правки перекрёстных
  ссылок morning → wttr/habr.
- `src/dev_helper_bot/skills.py` — парсинг frontmatter, структура Skill,
  сборка catalog-промпта вместо полной склейки тел.
- `src/dev_helper_bot/tools.py` — `GET_SKILL_TOOL_SPEC` + реализация.
- `src/dev_helper_bot/agent.py`, `main.py`, `scripts/obs-benchmark.py` —
  регистрация и исполнение `get_skill`.
- `tests/unit/test_skills.py` и связанные тесты агента/тулов; README.
- Зависимость: лёгкий YAML-парсер frontmatter (stdlib или уже имеющийся
  PyYAML — решить в design); новых runtime-сервисов нет.
