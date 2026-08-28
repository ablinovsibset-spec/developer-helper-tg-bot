from __future__ import annotations

from pathlib import Path

REASONING_EFFORT_LINE = "Reasoning: medium"
SKILLS_DIR_NAME = "skills"


def default_skills_dir() -> Path:
    """Каталог skills/ в корне репозитория (редактируется без переустановки пакета)."""
    return Path(__file__).resolve().parents[2] / SKILLS_DIR_NAME


def load_skills(skills_dir: Path) -> dict[str, str]:
    """Читает все .md-файлы каталога: имя без расширения → содержимое.

    Отсутствующий или пустой каталог — допустимое состояние: {} без ошибок.
    """
    if not skills_dir.is_dir():
        return {}
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(skills_dir.glob("*.md"))
    }


def build_system_prompt(skills: dict[str, str]) -> str:
    """Склейка системного промпта: Reasoning: medium + секции скиллов."""
    sections = [REASONING_EFFORT_LINE]
    for name, content in skills.items():
        sections.append(f"## {name}\n{content}")
    return "\n\n".join(sections)


def system_prompt_from_dir(skills_dir: Path) -> str:
    return build_system_prompt(load_skills(skills_dir))
