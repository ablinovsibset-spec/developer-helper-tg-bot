from __future__ import annotations

from datetime import datetime
from pathlib import Path

REASONING_EFFORT_LINE = "Reasoning: medium"
SKILLS_DIR_NAME = "skills"

SANDBOX_ENV_LINE = (
    "Окружение: команды инструмента exec выполняются в изолированном "
    "Linux-контейнере (Alpine), без доступа к файловой системе хоста. "
    "Контейнер долгоживущий: созданные файлы и установленные пакеты "
    "(pip install --user) переживают сообщения и сброс контекста /new; "
    "сброс состояния возможен только пересозданием контейнера. "
    "cd и переменные окружения между вызовами не сохраняются."
)

WEEKDAYS = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)


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


def datetime_line(now: datetime) -> str:
    """Строка даты/времени для промпта: дата, минуты и день недели."""
    return f"Текущие дата и время: {now:%Y-%m-%d %H:%M} ({WEEKDAYS[now.weekday()]})"


def build_system_prompt(skills: dict[str, str], now: datetime | None = None) -> str:
    """Склейка системного промпта: reasoning, дата/время, секции скиллов."""
    if now is None:
        now = datetime.now()
    sections = [
        f"{REASONING_EFFORT_LINE}\n{datetime_line(now)}\n{SANDBOX_ENV_LINE}"
    ]
    for name, content in skills.items():
        sections.append(f"## {name}\n{content}")
    return "\n\n".join(sections)


def system_prompt_from_dir(skills_dir: Path, now: datetime | None = None) -> str:
    return build_system_prompt(load_skills(skills_dir), now)
