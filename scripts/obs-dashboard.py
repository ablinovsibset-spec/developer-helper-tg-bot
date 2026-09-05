#!/usr/bin/env python3
"""Dashboard телеметрии агентных прогонов (change add-agent-observability).

Читает БД телеметрии (design D7) без запуска бота: режим агрегатов
(по умолчанию) и timeline одного прогона (--run <id>). Скрипт самодостаточен
— импортирует только stdlib и работает с файлом БД напрямую.

Запуск:
    python scripts/obs-dashboard.py                  # агрегаты по всем прогонам
    python scripts/obs-dashboard.py --label baseline # фильтр по метке
    python scripts/obs-dashboard.py --since 2026-09-01
    python scripts/obs-dashboard.py --run 42         # timeline прогона
    python scripts/obs-dashboard.py --db /path/to/obs.db

Путь к БД: --db, иначе OBS_DB_PATH, иначе дефолт на VM-локальном диске.
Пустая база и неизвестный run_id — понятные сообщения, не трейсбек.
Недоступные данные поставщика (кэшированные/reasoning-токены) выводятся
как «недоступно», а не как ноль (design D8).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections.abc import Iterable

DEFAULT_OBS_DB_PATH = "~/.local/share/dev-helper-bot/observability.db"

RUN_STATUS_LABELS = {
    "success": "успех",
    "steps_exhausted": "остановка по лимиту шагов",
    "validation_error": "ошибка валидации ответа",
    "llm_error": "ошибка LLM",
    None: "не завершён",
}

NOT_AVAILABLE = "недоступно"

AGGREGATES_SQL = """
SELECT
    COUNT(*),
    COUNT(status),
    COALESCE(SUM(input_tokens), 0),
    COALESCE(SUM(output_tokens), 0),
    SUM(cached_tokens),
    SUM(reasoning_tokens),
    COALESCE(SUM(estimated_cost), 0),
    COALESCE(SUM(llm_calls), 0),
    COALESCE(SUM(tool_calls), 0)
FROM runs
WHERE (:label IS NULL OR label = :label)
  AND (:since IS NULL OR started_at >= :since)
"""

STATUS_BREAKDOWN_SQL = """
SELECT status, COUNT(*) FROM runs
WHERE (:label IS NULL OR label = :label)
  AND (:since IS NULL OR started_at >= :since)
GROUP BY status ORDER BY COUNT(*) DESC
"""

CACHE_SQL = """
SELECT SUM(c.cached_tokens), SUM(c.input_tokens)
FROM llm_calls c JOIN runs r ON c.run_id = r.id
WHERE (:label IS NULL OR r.label = :label)
  AND (:since IS NULL OR r.started_at >= :since)
  AND c.cached_tokens IS NOT NULL
"""

REASONING_AVAILABLE_SQL = """
SELECT EXISTS(
    SELECT 1 FROM llm_calls c JOIN runs r ON c.run_id = r.id
    WHERE (:label IS NULL OR r.label = :label)
      AND (:since IS NULL OR r.started_at >= :since)
      AND c.reasoning_tokens IS NOT NULL
)
"""

TOP_TOOLS_SQL = """
SELECT tc.tool_name, SUM(tc.output_tokens), COUNT(*)
FROM tool_calls tc JOIN runs r ON tc.run_id = r.id
WHERE (:label IS NULL OR r.label = :label)
  AND (:since IS NULL OR r.started_at >= :since)
GROUP BY tc.tool_name
ORDER BY SUM(tc.output_tokens) DESC
LIMIT 5
"""

RUN_SQL = """
SELECT id, chat_id, label, started_at, finished_at, status,
       llm_calls, tool_calls, input_tokens, output_tokens,
       cached_tokens, reasoning_tokens, estimated_cost
FROM runs WHERE id = ?
"""

LLM_TURNS_SQL = """
SELECT id, turn_number, ok, latency_ms, input_tokens, output_tokens,
       cached_tokens, reasoning_tokens, estimated_cost, model
FROM llm_calls WHERE run_id = ? ORDER BY turn_number, id
"""

TOOL_CALLS_SQL = """
SELECT id, turn_number, tool_name, ok, duration_ms, output_tokens,
       input_size, output_size
FROM tool_calls WHERE run_id = ? ORDER BY turn_number, id
"""


def human_count(n: float | int | None) -> str:
    """Компактное число: 4_200_000 → «4.2M», 39_400 → «39.4k», 8200 → «8,200»."""
    if n is None:
        return NOT_AVAILABLE
    n = int(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 10_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:,}"


def human_float(n: float | None) -> str:
    if n is None:
        return NOT_AVAILABLE
    return f"{n:g}"


def human_cost(cost: float | None) -> str:
    if cost is None:
        return NOT_AVAILABLE
    if 0 < cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.2f}"


def human_duration(ms: float | None) -> str:
    if ms is None:
        return NOT_AVAILABLE
    if ms < 1000:
        return f"{ms:.0f}мс"
    return f"{ms / 1000:.1f}с"


def status_label(status: str | None) -> str:
    return RUN_STATUS_LABELS.get(status, status or "не завершён")


def render_aggregates(conn: sqlite3.Connection, label: str | None, since: str | None) -> str:
    params = {"label": label, "since": since}
    (total, finished, input_t, output_t, cached_t, reasoning_t,
     cost, llm_total, tool_total) = conn.execute(AGGREGATES_SQL, params).fetchone()

    if not total:
        hint = " Проверьте фильтры --label/--since." if label or since else ""
        return f"В базе телеметрии нет прогонов.{hint}"

    lines: list[str] = []
    lines.append(f"Прогонов: {total} (завершено {finished})")
    for status, count in conn.execute(STATUS_BREAKDOWN_SQL, params).fetchall():
        lines.append(f"  {status_label(status)}: {count}")

    lines.append("")
    lines.append("Токены:")
    lines.append(f"  Входные       {human_count(input_t)}")
    lines.append(f"  Выходные      {human_count(output_t)}")

    cached_sum, cached_input_sum = conn.execute(CACHE_SQL, params).fetchone()
    if cached_sum is None:
        lines.append("  Кэшированные  недоступно (поставщик не отдаёт)")
    else:
        lines.append(f"  Кэшированные  {human_count(cached_sum)}")

    (reasoning_available,) = conn.execute(REASONING_AVAILABLE_SQL, params).fetchone()
    lines.append(
        f"  Reasoning     {human_count(reasoning_t) if reasoning_available else 'недоступно (поставщик не отдаёт)'}"
    )

    lines.append("")
    lines.append(f"Стоимость (виртуальная): {human_cost(cost)}")

    if finished:
        tokens_per_run = (input_t + output_t) / finished
        lines.append("")
        lines.append("Средне на завершённый прогон:")
        lines.append(f"  Токены        {human_float(round(tokens_per_run, 1))}")
        lines.append(f"  LLM-ходы      {human_float(round(llm_total / finished, 1))}")
        lines.append(f"  Tool-вызовы   {human_float(round(tool_total / finished, 1))}")

    if cached_sum is not None and cached_input_sum:
        rate = cached_sum / cached_input_sum * 100
        lines.append("")
        lines.append(f"Cache hit rate: {rate:.0f}%")
    else:
        lines.append("")
        lines.append("Cache hit rate: недоступно (поставщик не отдаёт)")

    top = conn.execute(TOP_TOOLS_SQL, params).fetchall()
    lines.append("")
    if not top:
        lines.append("Вызовы инструментов: нет")
    else:
        grand_total = sum(tokens for _, tokens, _ in top) or 1
        lines.append("Топ инструментов (вклад в токены):")
        for name, tokens, count in top:
            share = tokens / grand_total * 100 if tokens else 0.0
            lines.append(
                f"  {name:<14} {human_count(tokens):>8} ток.  {share:.0f}%  "
                f"({count} вызов.)"
            )
    return "\n".join(lines)


def _llm_turn_line(row: tuple) -> str:
    (_id, _turn, ok, latency, input_t, output_t, cached, reasoning, cost, model) = row
    if not ok:
        return f"LLM      ошибка недоступности  {human_duration(latency)}"
    cached_part = (
        f"  кэш {human_count(cached)}" if cached is not None else ""
    )
    reasoning_part = (
        f"  reasoning {human_count(reasoning)}" if reasoning is not None else ""
    )
    return (
        f"LLM      in {human_count(input_t):>8}  out {human_count(output_t):>8}"
        f"{cached_part}{reasoning_part}  {human_duration(latency)}"
        f"  {human_cost(cost)}"
    )


def _tool_line(row: tuple) -> str:
    (_id, _turn, name, ok, duration, tokens, input_size, output_size) = row
    outcome = "успех" if ok else "ошибка"
    return (
        f"{name:<8} ~{human_count(tokens)} ток.  {human_duration(duration)}  "
        f"{outcome}  (арг. {input_size} симв., рез. {output_size} симв.)"
    )


def render_timeline(conn: sqlite3.Connection, run_id: int) -> str:
    run = conn.execute(RUN_SQL, (run_id,)).fetchone()
    if run is None:
        return f"Прогон #{run_id} не найден в базе телеметрии."

    (_id, chat_id, label, started, finished, status, llm_n, tool_n,
     input_t, output_t, cached, reasoning, cost) = run

    lines: list[str] = [
        f"Прогон #{run_id} — чат {chat_id}"
        + (f", метка {label!r}" if label else "")
        + f", {status_label(status)}",
        f"Начало {started} → {finished or '…'}"
        + f"  LLM-ходов: {llm_n if llm_n is not None else 0}"
        + f", tool-вызовов: {tool_n if tool_n is not None else 0}",
        f"Токены: in {human_count(input_t)}, out {human_count(output_t)},"
        f" кэш {human_count(cached)},"
        f" reasoning {human_count(reasoning)}  стоимость {human_cost(cost)}",
        "─" * 60,
    ]

    events: list[tuple[int, int, int, str]] = []  # (turn, kind_rank, id, line)
    for row in conn.execute(LLM_TURNS_SQL, (run_id,)).fetchall():
        events.append((row[1], 0, row[0], _llm_turn_line(row)))
    for row in conn.execute(TOOL_CALLS_SQL, (run_id,)).fetchall():
        events.append((row[1], 1, row[0], _tool_line(row)))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    for turn, _rank, _id, line in events:
        lines.append(f"Turn {turn:<3} {line}")
    if not events:
        lines.append("(событий нет)")
    return "\n".join(lines)


def resolve_db_path(explicit: str | None) -> str:
    return os.path.expanduser(explicit or os.getenv("OBS_DB_PATH") or DEFAULT_OBS_DB_PATH)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Агрегаты и timeline телеметрии agent-прогонов (obs.db)."
    )
    parser.add_argument("--db", help="путь к БД телеметрии (по умолчанию OBS_DB_PATH)")
    parser.add_argument("--label", help="фильтр прогонов по метке")
    parser.add_argument("--since", help="прогоны, начатые с даты (ISO, напр. 2026-09-01)")
    parser.add_argument("--run", type=int, help="timeline конкретного прогона по его id")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path = resolve_db_path(args.db)
    if not os.path.exists(db_path):
        print(f"БД телеметрии не найдена: {db_path}")
        print("Запустите бота — база создаётся при старте (OBS_DB_PATH).")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        if args.run is not None:
            print(render_timeline(conn, args.run))
        else:
            print(render_aggregates(conn, args.label, args.since))
    except sqlite3.Error as exc:
        print(f"Не удалось прочитать БД телеметрии ({db_path}): {exc}")
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
