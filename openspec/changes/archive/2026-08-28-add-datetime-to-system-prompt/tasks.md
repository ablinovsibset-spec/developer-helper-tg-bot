# Tasks: add-datetime-to-system-prompt

## 1. Строка даты/времени в сборке промпта (skills.py)

- [x] 1.1 В `skills.py` добавить форматирование: кортеж русских дней недели (`WEEKDAYS`), функцию `datetime_line(now: datetime) -> str` → `Текущие дата и время: 2026-08-28 07:45 (четверг)` (формат `%Y-%m-%d %H:%M`)
- [x] 1.2 Расширить `build_system_prompt(skills, now: datetime | None = None)` и `system_prompt_from_dir(skills_dir, now=None)`: строка даты/времени идёт второй — после `Reasoning: medium`, перед секциями скиллов; без `now` используется `datetime.now()` (локальное время сервера)

## 2. Обновление системного сообщения в main.py

- [x] 2.1 Заменить `dp["system_prompt"] = system_prompt_from_dir(...)` на `dp["skills"] = load_skills(default_skills_dir())` (скиллы по-прежнему читаются один раз при старте)
- [x] 2.2 В `handle_text` перед добавлением user-сообщения собирать свежий промпт `build_system_prompt(skills, datetime.now())` и обновлять системное сообщение истории (`chat_history`): нет истории — создать `[system]`; есть — перезаписать `content` у `history[0]`, не трогая остальные сообщения

## 3. Тесты

- [x] 3.1 Обновить `tests/unit/test_skills.py`: точная строка `datetime_line` на фиксированном `datetime` (включая день недели, отсутствие секунд); порядок частей промпта (reasoning → дата/время → скиллы); `build_system_prompt({})` без скиллов содержит reasoning + дату/время; дефолт `now=None` не падает
- [x] 3.2 Обновить `tests/unit/test_main.py`: системное сообщение каждого запроса содержит префикс `Текущие дата и время:`; при смене времени (фейковые часы через `monkeypatch.setattr(dev_helper_bot.main, "datetime", ...)`) во втором сообщении диалога содержимое `history[0]` обновляется, а история (user/assistant/tool) сохраняется; после `/new` новый контекст начинается с системного сообщения с датой/временем

## 4. Финальная проверка

- [x] 4.1 Полный прогон `pytest` без сети и секретов; `openspec validate add-datetime-to-system-prompt`
- [x] 4.2 Ручной smoke: спросить у бота «какой сегодня день и сколько времени» — ответ без обращения к инструментам; в долгом диалоге время актуализируется
