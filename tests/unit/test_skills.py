from __future__ import annotations

from pathlib import Path

from dev_helper_bot.skills import (
    REASONING_EFFORT_LINE,
    build_system_prompt,
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


def test_build_system_prompt_contains_reasoning_and_sections(tmp_path):
    skills = load_skills(make_skills_dir(tmp_path))

    prompt = build_system_prompt(skills)

    assert prompt.startswith(REASONING_EFFORT_LINE + "\n\n")
    assert "## wttr-in-api\nПравила wttr.in" in prompt
    assert "## morning\nУтро: погода Минск" in prompt


def test_build_system_prompt_without_skills_is_just_reasoning_line():
    assert build_system_prompt({}) == REASONING_EFFORT_LINE


def test_system_prompt_from_dir_end_to_end(tmp_path):
    prompt = system_prompt_from_dir(make_skills_dir(tmp_path))

    assert prompt.startswith("Reasoning: medium")
    assert "wttr-in-api" in prompt
    assert "morning" in prompt
