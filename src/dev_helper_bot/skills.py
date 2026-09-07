from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dev_helper_bot.llm import Message

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

MEMORY_ENV_LINE = (
    "Память бесед: прошлые беседы этого чата хранятся и доступны через "
    "инструменты: list_sessions — обзор завершённых бесед (дата начала, "
    "число сообщений, превью), search_history — поиск по ключевому слову. "
    "Если пользователь ссылается на прошлое («что мы обсуждали», "
    "«помнишь, я спрашивал»), сначала вызови list_sessions, затем "
    "search_history по ключевому слову из превью. Не отвечай «не помню», "
    "не проверив эти инструменты."
)

SKILLS_CATALOG_INTRO = (
    "Скиллы: ниже каталог (name — description). Если description подходит "
    "к запросу — вызови get_skill(name) и следуй телу. Не выдумывай шаги "
    "скилла без загрузки."
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


@dataclass(frozen=True)
class Skill:
    """Скилл: метаданные для catalog-промпта и тело для get_skill."""

    name: str
    description: str
    body: str


class SkillLoadError(ValueError):
    """Некорректный файл скилла: fail-fast при загрузке каталога."""


def default_skills_dir() -> Path:
    """Каталог skills/ в корне репозитория (редактируется без переустановки пакета)."""
    return Path(__file__).resolve().parents[2] / SKILLS_DIR_NAME


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Отделяет YAML frontmatter от тела. Документ MUST начинаться с ---."""
    if text.startswith("---\r\n"):
        rest = text[5:]
    elif text.startswith("---\n"):
        rest = text[4:]
    else:
        raise SkillLoadError(
            "файл должен начинаться с YAML frontmatter между строками ---"
        )
    for sep in ("\n---\r\n", "\n---\n"):
        idx = rest.find(sep)
        if idx != -1:
            return rest[:idx], rest[idx + len(sep) :]
    if rest.startswith("---\n") or rest.startswith("---\r\n"):
        # Пустой frontmatter: --- сразу после открывающего
        return "", rest.split("\n", 1)[1] if "\n" in rest else ""
    raise SkillLoadError("нет закрывающего разделителя --- у frontmatter")


def _parse_scalar_block(lines: list[str], start: int, style: str) -> tuple[str, int]:
    """Читает block scalar (`|` или `>` / `>-`) начиная со строки после маркера."""
    collected: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        collected.append(line)
        i += 1
    indents = [
        len(ln) - len(ln.lstrip(" ")) for ln in collected if ln.strip()
    ]
    indent = min(indents) if indents else 0
    stripped = [(ln[indent:] if len(ln) >= indent else ln) for ln in collected]
    if style.startswith(">"):
        parts: list[str] = []
        paragraph: list[str] = []
        for ln in stripped:
            if not ln.strip():
                if paragraph:
                    parts.append(" ".join(paragraph))
                    paragraph = []
            else:
                paragraph.append(ln.strip())
        if paragraph:
            parts.append(" ".join(paragraph))
        return " ".join(p for p in parts if p).strip(), i
    return "\n".join(stripped).strip(), i

def _parse_frontmatter_fields(yaml_text: str) -> dict[str, str]:
    """Минимальный разбор name/description (stdlib, design D3)."""
    lines = yaml_text.splitlines()
    fields: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if ":" not in line:
            raise SkillLoadError(f"некорректная строка frontmatter: {line!r}")
        key, _, raw = line.partition(":")
        key = key.strip()
        if key not in ("name", "description"):
            raise SkillLoadError(
                f"неподдерживаемый ключ frontmatter: {key!r} "
                "(допустимы только name и description)"
            )
        value = raw.strip()
        if value in ("|", ">", ">-", "|-", "|+"):
            parsed, i = _parse_scalar_block(lines, i + 1, value)
            fields[key] = parsed
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        fields[key] = value
        i += 1
    return fields


def parse_skill_file(text: str, stem: str) -> Skill:
    """Разбирает markdown скилла: frontmatter + тело; name MUST == stem."""
    yaml_text, body = _split_frontmatter(text)
    fields = _parse_frontmatter_fields(yaml_text)
    name = (fields.get("name") or "").strip()
    description = (fields.get("description") or "").strip()
    if not name:
        raise SkillLoadError("отсутствует обязательное поле name")
    if not description:
        raise SkillLoadError("отсутствует обязательное непустое поле description")
    if name != stem:
        raise SkillLoadError(
            f"name {name!r} не совпадает со stem файла {stem!r}"
        )
    return Skill(name=name, description=description, body=body.lstrip("\n"))


def load_skills(skills_dir: Path) -> dict[str, Skill]:
    """Читает все .md-файлы каталога → dict[name, Skill].

    Отсутствующий или пустой каталог — допустимое состояние: {} без ошибок.
    Некорректный frontmatter — SkillLoadError с путём файла (fail fast).
    """
    if not skills_dir.is_dir():
        return {}
    skills: dict[str, Skill] = {}
    for path in sorted(skills_dir.glob("*.md")):
        try:
            skill = parse_skill_file(path.read_text(encoding="utf-8"), path.stem)
        except SkillLoadError as exc:
            raise SkillLoadError(f"{path}: {exc}") from exc
        skills[skill.name] = skill
    return skills


def datetime_line(now: datetime) -> str:
    """Контекстная строка даты/времени: дата, минуты и день недели."""
    return f"Текущие дата и время: {now:%Y-%m-%d %H:%M} ({WEEKDAYS[now.weekday()]})"


def build_system_prompt(skills: dict[str, Skill]) -> str:
    """Системный промпт: reasoning, окружение, catalog name+description.

    Тела скиллов не включаются — модель загружает их через get_skill.
    Без даты/времени: системный промпт байтово стабилен между сообщениями.
    """
    sections = [f"{REASONING_EFFORT_LINE}\n{SANDBOX_ENV_LINE}\n{MEMORY_ENV_LINE}"]
    if skills:
        lines = [SKILLS_CATALOG_INTRO]
        for name in sorted(skills):
            skill = skills[name]
            lines.append(f"- {skill.name}: {skill.description}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def system_prompt_from_dir(skills_dir: Path) -> str:
    return build_system_prompt(load_skills(skills_dir))


def build_request_messages(
    skills: dict[str, Skill],
    session_history: list[Message],
    user_text: str,
    now: datetime | None = None,
) -> list[Message]:
    """Единая сборка контекста запроса (design D4): system + загруженная
    история сессии + текущее user-сообщение с контекстной строкой времени.

    Контекстная строка даты/времени живёт только в текущем сообщении
    (design D3): не персистится в память, внутри прогона неизменна,
    между прогонами — свежая. Префикс «system + история» байтово
    стабилен, пока не меняются скиллы.
    """
    if now is None:
        now = datetime.now()
    return [
        {"role": "system", "content": build_system_prompt(skills)},
        *session_history,
        {"role": "user", "content": f"{datetime_line(now)}\n{user_text}"},
    ]
