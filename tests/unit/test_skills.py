from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dev_helper_bot.skills import (
    MEMORY_ENV_LINE,
    REASONING_EFFORT_LINE,
    SANDBOX_ENV_LINE,
    WEEKDAYS,
    build_system_prompt,
    datetime_line,
    load_skills,
    system_prompt_from_dir,
)


def make_skills_dir(tmp_path: Path) -> Path:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "wttr-in-api.md").write_text(
        "Правила wttr.in: format=3 и ?lang=ru", encoding="utf-8"
    )
    (skills_dir / "morning.md").write_text(
        "Утро: погода Минск, события, сводка.", encoding="utf-8"
    )
    return skills_dir


def test_load_skills_reads_md_files_with_stem_names(tmp_path):
    skills_dir = make_skills_dir(tmp_path)

    skills = load_skills(skills_dir)

    assert set(skills) == {"wttr-in-api", "morning"}
    assert skills["morning"] == "Утро: погода Минск, события, сводка."


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


def test_build_system_prompt_contains_reasoning_datetime_and_sections(tmp_path):
    skills = load_skills(make_skills_dir(tmp_path))
    now = datetime(2026, 8, 28, 7, 45)

    prompt = build_system_prompt(skills, now)

    assert prompt.startswith(
        f"{REASONING_EFFORT_LINE}\nТекущие дата и время: 2026-08-28 07:45 (пятница)"
    )
    assert SANDBOX_ENV_LINE in prompt
    assert "## wttr-in-api\nПравила wttr.in" in prompt
    assert "## morning\nУтро: погода Минск" in prompt


def test_build_system_prompt_without_skills_is_reasoning_datetime_and_env():
    prompt = build_system_prompt({}, datetime(2026, 8, 28, 7, 45))

    assert prompt == (
        "Reasoning: medium\nТекущие дата и время: 2026-08-28 07:45 (пятница)"
        f"\n{SANDBOX_ENV_LINE}"
        f"\n{MEMORY_ENV_LINE}"
    )


def test_memory_env_line_present_in_prompt_with_skills(tmp_path):
    skills = load_skills(make_skills_dir(tmp_path))

    prompt = build_system_prompt(skills, datetime(2026, 8, 28, 7, 45))

    assert MEMORY_ENV_LINE in prompt


def test_memory_env_line_present_in_prompt_with_empty_skills():
    prompt = build_system_prompt({})

    assert MEMORY_ENV_LINE in prompt


def test_memory_env_line_goes_right_after_sandbox_env_line():
    prompt = build_system_prompt({}, datetime(2026, 8, 28, 7, 45))

    assert prompt.index(SANDBOX_ENV_LINE) < prompt.index(MEMORY_ENV_LINE)


def test_build_system_prompt_default_now_does_not_crash():
    prompt = build_system_prompt({})

    assert prompt.startswith(REASONING_EFFORT_LINE + "\nТекущие дата и время: ")
    assert "(" in prompt  # день недели присутствует


def test_system_prompt_from_dir_end_to_end(tmp_path):
    prompt = system_prompt_from_dir(make_skills_dir(tmp_path), datetime(2026, 8, 28, 7, 45))

    assert prompt.startswith("Reasoning: medium")
    assert "Текущие дата и время: 2026-08-28 07:45 (пятница)" in prompt
    assert "wttr-in-api" in prompt
    assert "morning" in prompt


def test_sandbox_env_line_reflects_persistent_state():
    """Промпт сообщает модели персистентность: файлы/пакеты переживают
    сообщения и /new; сброс — только пересозданием контейнера."""
    prompt = build_system_prompt({}, datetime(2026, 8, 28, 7, 45))

    assert "переживают сообщения" in prompt
    assert "/new" in prompt
    assert "пересозданием контейнера" in prompt
    # Прошлая формулировка «до конца сообщения» противоречит жителю.
    assert "до конца обработки" not in prompt
