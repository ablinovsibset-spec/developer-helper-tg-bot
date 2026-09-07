from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from dev_helper_bot.skills import (
    MEMORY_ENV_LINE,
    REASONING_EFFORT_LINE,
    SANDBOX_ENV_LINE,
    SKILLS_CATALOG_INTRO,
    WEEKDAYS,
    Skill,
    SkillLoadError,
    build_request_messages,
    build_system_prompt,
    datetime_line,
    load_skills,
    parse_skill_file,
    system_prompt_from_dir,
)


def skill_md(
    name: str,
    description: str,
    body: str,
) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"


def make_skills_dir(tmp_path: Path) -> Path:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "wttr-in-api.md").write_text(
        skill_md(
            "wttr-in-api",
            "Погода через wttr.in",
            "Правила wttr.in: format=3 и ?lang=ru\ncurl -s wttr.in",
        ),
        encoding="utf-8",
    )
    (skills_dir / "morning.md").write_text(
        skill_md(
            "morning",
            "Утренняя сводка",
            "Утро: погода Минск, события, сводка.",
        ),
        encoding="utf-8",
    )
    return skills_dir


def test_parse_skill_file_ok():
    skill = parse_skill_file(
        skill_md("morning", "Утренняя рутина", "Шаг 1. Погода"),
        "morning",
    )
    assert skill == Skill(
        name="morning",
        description="Утренняя рутина",
        body="Шаг 1. Погода\n",
    )


def test_parse_skill_file_missing_description():
    text = "---\nname: morning\n---\nтело\n"
    with pytest.raises(SkillLoadError, match="description"):
        parse_skill_file(text, "morning")


def test_parse_skill_file_name_must_match_stem():
    with pytest.raises(SkillLoadError, match="не совпадает"):
        parse_skill_file(
            skill_md("evening", "Вечер", "тело"),
            "morning",
        )


def test_load_skills_reads_md_files_with_stem_names(tmp_path):
    skills_dir = make_skills_dir(tmp_path)

    skills = load_skills(skills_dir)

    assert set(skills) == {"wttr-in-api", "morning"}
    assert skills["morning"].body.startswith("Утро: погода Минск")
    assert skills["morning"].description == "Утренняя сводка"


def test_load_skills_ignores_non_md_files(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "notes.txt").write_text("не скилл", encoding="utf-8")

    assert load_skills(skills_dir) == {}


def test_load_skills_empty_dir_is_allowed(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    assert load_skills(skills_dir) == {}


def test_load_skills_missing_dir_is_allowed(tmp_path):
    assert load_skills(tmp_path / "nope") == {}


def test_load_skills_fail_fast_on_bad_frontmatter(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "broken.md").write_text(
        "---\nname: broken\n---\nбез description\n", encoding="utf-8"
    )

    with pytest.raises(SkillLoadError, match="broken.md"):
        load_skills(skills_dir)


def test_datetime_line_exact_format_without_seconds():
    now = datetime(2026, 8, 28, 7, 45, 33)

    assert datetime_line(now) == (
        "Текущие дата и время: 2026-08-28 07:45 (пятница)"
    )


def test_datetime_line_weekday_for_each_day():
    monday = datetime(2026, 8, 24)
    assert [datetime_line(monday.replace(day=monday.day + i)) for i in range(7)] == [
        f"Текущие дата и время: 2026-08-{24 + i} 00:00 ({WEEKDAYS[i]})"
        for i in range(7)
    ]


def test_build_system_prompt_contains_catalog_not_bodies(tmp_path):
    skills = load_skills(make_skills_dir(tmp_path))

    prompt = build_system_prompt(skills)

    assert prompt.startswith(f"{REASONING_EFFORT_LINE}\n{SANDBOX_ENV_LINE}")
    assert SKILLS_CATALOG_INTRO in prompt
    assert "get_skill" in prompt
    assert "- morning: Утренняя сводка" in prompt
    assert "- wttr-in-api: Погода через wttr.in" in prompt
    assert "format=3" not in prompt
    assert "curl -s wttr.in" not in prompt
    assert "Утро: погода Минск" not in prompt


def test_build_system_prompt_has_no_datetime_line(tmp_path):
    """Дата/время ушли из системного промпта (design D3): префикс стабилен."""
    skills = load_skills(make_skills_dir(tmp_path))

    prompt = build_system_prompt(skills)

    assert "Текущие дата и время" not in prompt


def test_build_system_prompt_without_skills_is_reasoning_and_env():
    prompt = build_system_prompt({})

    assert prompt == (
        f"{REASONING_EFFORT_LINE}\n{SANDBOX_ENV_LINE}\n{MEMORY_ENV_LINE}"
    )
    assert SKILLS_CATALOG_INTRO not in prompt


def test_memory_env_line_present_in_prompt_with_skills(tmp_path):
    skills = load_skills(make_skills_dir(tmp_path))

    prompt = build_system_prompt(skills)

    assert MEMORY_ENV_LINE in prompt


def test_memory_env_line_present_in_prompt_with_empty_skills():
    prompt = build_system_prompt({})

    assert MEMORY_ENV_LINE in prompt


def test_memory_env_line_goes_right_after_sandbox_env_line():
    prompt = build_system_prompt({})

    assert prompt.index(SANDBOX_ENV_LINE) < prompt.index(MEMORY_ENV_LINE)


def test_system_prompt_from_dir_end_to_end(tmp_path):
    prompt = system_prompt_from_dir(make_skills_dir(tmp_path))

    assert prompt.startswith("Reasoning: medium")
    assert "Текущие дата и время" not in prompt
    assert "wttr-in-api" in prompt
    assert "morning" in prompt
    assert "curl -s wttr.in" not in prompt


def test_sandbox_env_line_reflects_persistent_state():
    """Промпт сообщает модели персистентность: файлы/пакеты переживают
    сообщения и /new; сброс — только пересозданием контейнера."""
    prompt = build_system_prompt({})

    assert "переживают сообщения" in prompt
    assert "/new" in prompt
    assert "пересозданием контейнера" in prompt
    # Прошлая формулировка «до конца сообщения» противоречит жителю.
    assert "до конца обработки" not in prompt


def test_repo_skills_load_with_frontmatter():
    """Реальные skills/*.md парсятся и не попадают телами в system."""
    from dev_helper_bot.skills import default_skills_dir

    skills = load_skills(default_skills_dir())
    assert {"morning", "wttr-in-api", "habr-feed", "search-history"} <= set(skills)
    prompt = build_system_prompt(skills)
    assert "get_skill" in prompt
    assert "format=j1" not in prompt
    assert skills["morning"].body.startswith("# Скилл: утренняя сводка")


# --- Сборка запроса: контекстная строка и стабильный префикс (design D3/D4) ---

T1 = datetime(2026, 8, 28, 7, 45)
T2 = datetime(2026, 8, 28, 7, 47)
SESSION_HISTORY = [
    {"role": "user", "content": "старый вопрос"},
    {"role": "assistant", "content": "старый ответ"},
]


def test_build_request_messages_places_datetime_in_current_user_message():
    messages = build_request_messages({}, [], "новый вопрос", T1)

    assert messages == [
        {"role": "system", "content": build_system_prompt({})},
        {
            "role": "user",
            "content": "Текущие дата и время: 2026-08-28 07:45 (пятница)"
            "\nновый вопрос",
        },
    ]


def test_build_request_messages_puts_history_before_current_message():
    messages = build_request_messages({}, SESSION_HISTORY, "вопрос", T1)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1] == SESSION_HISTORY[0]
    assert messages[2] == SESSION_HISTORY[1]
    assert messages[3]["content"].endswith("вопрос")


def test_build_request_messages_prefix_is_byte_stable_between_messages():
    first = build_request_messages({}, SESSION_HISTORY, "первое", T1)
    second = build_request_messages(
        {}, SESSION_HISTORY + [{"role": "user", "content": "первое"}], "второе", T2
    )

    # system + загруженная история идентичны — меняется только хвост
    assert second[:3] == first[:3]
    # Время обновилось в текущем сообщении, история не содержит строки времени
    assert "07:45" in first[3]["content"]
    assert "07:47" in second[4]["content"]
    assert "Текущие дата и время" not in second[1]["content"]
