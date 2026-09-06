#!/usr/bin/env python3
"""Аудит потребления токенов по БД телеметрии (change add-token-audit,
optimize-token-usage D5).

Отвечает на четыре вопроса Части 2 задачи observability (design D6/D7/D8),
формируя baseline для оптимизаций, и добавляет стоимость с кэш-скидкой:

  Q1  Топ инструментов      — оценочные токены вывода по tool_name
  Q2  Самый дорогой ход     — распределение input_tokens по turn_number
  Q3  Рост типов контекста  — доли типов по разбивке prompt_roles
  Q4  Повторные токены      — ре-отправки внутри прогона + межпрогонный слой
  Q5  Стоимость             — фактический биллинг поставщика (usage.cost,
                              ₽ как есть) основной строкой при наличии;
                              оценки по прайсу — сырая и эффективная
                              (кэш-скидка cached_tokens) с пометкой «оценка»

Как obs-dashboard.py: самодостаточный stdlib-скрипт (sqlite3), читает БД
без запуска бота. У dashborada другая роль (агрегаты и timeline) — аудит
отвечает на вопросы Части 2.

Запуск:
    python scripts/obs-audit.py                              # дефолтная БД
    python scripts/obs-audit.py --db ~/.local/share/dev-helper-bot/benchmark.db
    python scripts/obs-audit.py --label benchmark --since 2026-09-01
    python scripts/obs-audit.py --cached-price-multiplier 0.5

Пустая база и отсутствие файла — понятные сообщения, не трейсбек.
Записи без prompt_roles (созданные до миграции) не подставляются нулями:
секция Q3 честно сообщает о недоступности. Оценки chars/4 (как и в
телеметрии tool-вызовов) помечаются в выводе.

Цель −30% стоимости (задача observability, часть 3) считается по сырым
входным токенам (Q4); кэш-скидка в Q5 приводится справочно.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections.abc import Iterable

DEFAULT_OBS_DB_PATH = "~/.local/share/dev-helper-bot/observability.db"

DEFAULT_PRICE_INPUT_PER_M = 0.11
DEFAULT_PRICE_OUTPUT_PER_M = 0.60
DEFAULT_CACHED_PRICE_MULTIPLIER = 0.25
"""Дефолты виртуального прайса — те же, что у config.py бота; кэшированные
токены по отдельному прайсу (design D5): 0.25 × входного."""

CHARS_PER_TOKEN = 4
"""Та же оценка, что estimate_tool_tokens в телеметрии (design D7)."""

BAR_WIDTH = 24
"""Ширина ASCII-бара в символах (100% → BAR_WIDTH блоков)."""

NOT_AVAILABLE = "недоступно"

TYPE_SYSTEM = "System prompt"
TYPE_SESSION_HISTORY = "История сессии (ход 1: user+assistant)"
TYPE_DIALOG_HISTORY = "История диалога (assistant)"
TYPE_USER = "Текущая задача (user сверх хода 1)"
TYPE_TOOLS = "Выводы инструментов (вкл. файлы)"
CONTEXT_TYPES = (TYPE_SYSTEM, TYPE_SESSION_HISTORY, TYPE_DIALOG_HISTORY,
                 TYPE_USER, TYPE_TOOLS)


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


def bar(share_percent: float, width: int = BAR_WIDTH) -> str:
    """ASCII-бар: доля в процентах → width блоков «█»."""
    filled = round(share_percent / 100 * width)
    return "█" * max(filled, 0)


# --- Загрузка данных (одним проходом, дальше чистый python) ---


def load_runs(conn: sqlite3.Connection, label: str | None,
              since: str | None) -> list[dict]:
    sql = ("SELECT id, chat_id, label, started_at FROM runs "
           "WHERE (:label IS NULL OR label = :label) "
           "AND (:since IS NULL OR started_at >= :since) ORDER BY id")
    return [
        {"id": rid, "chat_id": chat_id, "label": run_label,
         "started_at": started}
        for rid, chat_id, run_label, started in
        conn.execute(sql, {"label": label, "since": since}).fetchall()
    ]


def load_llm_calls(conn: sqlite3.Connection, run_ids: list[int]) -> list[dict]:
    if not run_ids:
        return []
    placeholders = ",".join("?" * len(run_ids))
    sql = (
        f"SELECT run_id, turn_number, input_tokens, output_tokens, "
        f"cached_tokens, prompt_roles, billed_cost "
        f"FROM llm_calls WHERE run_id IN ({placeholders}) "
        f"ORDER BY run_id, turn_number, id"
    )
    calls = []
    for (run_id, turn, input_tokens, output_tokens, cached_tokens,
         prompt_roles, billed_cost) in conn.execute(sql, run_ids):
        roles = None
        if prompt_roles is not None:
            try:
                roles = json.loads(prompt_roles)
            except ValueError:
                roles = None  # битый JSON трактуется как отсутствие разбивки
        calls.append({"run_id": run_id, "turn": turn,
                      "input_tokens": input_tokens,
                      "output_tokens": output_tokens,
                      "cached_tokens": cached_tokens,
                      "roles": roles,
                      "billed_cost": billed_cost})
    return calls


def load_tool_calls(conn: sqlite3.Connection, run_ids: list[int]) -> list[dict]:
    if not run_ids:
        return []
    placeholders = ",".join("?" * len(run_ids))
    sql = (
        f"SELECT tool_name, output_tokens FROM tool_calls "
        f"WHERE run_id IN ({placeholders})"
    )
    return [
        {"tool_name": name, "output_tokens": tokens or 0}
        for name, tokens in conn.execute(sql, run_ids).fetchall()
    ]


def group_by_run(calls: list[dict]) -> dict[int, list[dict]]:
    runs: dict[int, list[dict]] = {}
    for call in calls:
        runs.setdefault(call["run_id"], []).append(call)
    return runs


# --- Q1: Топ инструментов по токенам (design D6, спека token-audit) ---


def render_q1(tool_calls: list[dict]) -> str:
    lines = ["Q1. Топ инструментов по токенам (оценка вывода, chars/4)",
             "─" * 60]
    if not tool_calls:
        lines.append("Нет вызовов инструментов под фильтрами — данных нет.")
        return "\n".join(lines)

    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for call in tool_calls:
        totals[call["tool_name"]] = (
            totals.get(call["tool_name"], 0) + call["output_tokens"]
        )
        counts[call["tool_name"]] = counts.get(call["tool_name"], 0) + 1

    grand_total = sum(totals.values())
    lines.append(f"Всего оценочных токенов вывода: {human_count(grand_total)}")
    for name, tokens in sorted(totals.items(), key=lambda kv: -kv[1]):
        share = tokens / grand_total * 100 if grand_total else 0.0
        lines.append(
            f"  {name:<16} {human_count(tokens):>8} ток.  {share:5.1f}%  "
            f"{bar(share)}  ({counts[name]} вызов.)"
        )
    return "\n".join(lines)


# --- Q2: Самый дорогой ход (спека token-audit) ---


def render_q2(calls: list[dict]) -> str:
    lines = ["Q2. Самый дорогой ход по входному контексту", "─" * 60]
    with_input = [c for c in calls if c["input_tokens"] is not None]
    if not with_input:
        lines.append("Нет LLM-вызовов с входными токенами под фильтрами — "
                     "данных нет.")
        return "\n".join(lines)

    by_turn: dict[int, list[int]] = {}
    for call in with_input:
        by_turn.setdefault(call["turn"], []).append(call["input_tokens"])

    averages = {turn: sum(vals) / len(vals) for turn, vals in by_turn.items()}
    if len(averages) == 1:
        (only_turn, only_avg), = averages.items()
        lines.append(
            f"Ход {only_turn}: среднее {human_count(only_avg)} input токенов "
            f"({len(by_turn[only_turn])} прогон.)"
        )
        lines.append("Все прогоны содержат по одному LLM-ходу — динамики "
                     "роста нет.")
        lines.append(f"Самый дорогий ход: {only_turn} "
                     f"({human_count(only_avg)} input токенов, единственный).")
        return "\n".join(lines)

    previous: float | None = None
    for turn in sorted(averages):
        avg = averages[turn]
        growth = (
            f"  рост к предыдущему {avg / previous * 100 - 100:+.0f}%"
            if previous is not None
            else ""
        )
        lines.append(
            f"  Ход {turn}:  среднее {human_count(avg):>8} input токенов  "
            f"({len(by_turn[turn])} прогон.){growth}"
        )
        previous = avg
    top_turn = max(averages, key=lambda t: averages[t])
    lines.append(
        f"Самый дорогой ход: {top_turn} "
        f"(в среднем {human_count(averages[top_turn])} input токенов)."
    )
    return "\n".join(lines)


# --- Q3: Рост типов контекста (design D8) ---


def run_type_chars(run_calls: list[dict]) -> dict[str, int] | None:
    """Символы по типам контекста для одного прогона (design D8).

    system → System prompt; user+assistant хода 1 → загруженная история
    сессии; assistant сверх хода 1 → история диалога; user сверх хода 1 →
    текущая задача; tool → выводы инструментов (вкл. файлы). None, если ни
    один ход прогона не имеет разбивки.
    """
    first = min((c["turn"] for c in run_calls), default=None)
    chars = dict.fromkeys(CONTEXT_TYPES, 0)
    has_roles = False
    for call in run_calls:
        roles = call["roles"]
        if roles is None:
            continue
        has_roles = True
        turn = call["turn"]
        chars[TYPE_SYSTEM] += roles.get("system", 0) or 0
        chars[TYPE_TOOLS] += roles.get("tool", 0) or 0
        user = roles.get("user", 0) or 0
        assistant = roles.get("assistant", 0) or 0
        if turn == first:
            chars[TYPE_SESSION_HISTORY] += user + assistant
        else:
            chars[TYPE_USER] += user
            chars[TYPE_DIALOG_HISTORY] += assistant
    return chars if has_roles else None


def render_q3(calls: list[dict]) -> str:
    lines = ["Q3. Рост типов контекста (по разбивке prompt_roles)", "─" * 60]
    with_roles = [c for c in calls if c["roles"] is not None]
    if not with_roles:
        lines.append(
            "Недоступно: записи LLM-вызовов созданы до появления разбивки "
            "по ролям (prompt_roles null)."
        )
        return "\n".join(lines)
    skipped = len(calls) - len(with_roles)
    if skipped:
        lines.append(f"(без разбивки, не учтено: {skipped} вызов.)")

    totals = dict.fromkeys(CONTEXT_TYPES, 0)
    for run_calls in group_by_run(calls).values():
        run_chars = run_type_chars(run_calls)
        if run_chars is None:
            continue
        for type_name, value in run_chars.items():
            totals[type_name] += value

    grand_total = sum(totals.values())
    lines.append(f"Всего символов контекста: {human_count(grand_total)}")
    for type_name in CONTEXT_TYPES:
        share = totals[type_name] / grand_total * 100 if grand_total else 0.0
        lines.append(
            f"  {type_name:<42} {share:5.1f}%  {bar(share)}"
        )

    # Тренд долей по ходам, усреднённый по прогонам (design D8)
    turn_shares: dict[int, dict[str, list[int]]] = {}
    for run_calls in group_by_run(calls).values():
        first = min((c["turn"] for c in run_calls), default=None)
        for call in run_calls:
            roles = call["roles"]
            if roles is None:
                continue
            chars = {
                TYPE_SYSTEM: roles.get("system", 0) or 0,
                TYPE_TOOLS: roles.get("tool", 0) or 0,
                TYPE_SESSION_HISTORY: (
                    (roles.get("user", 0) or 0) + (roles.get("assistant", 0) or 0)
                    if call["turn"] == first else 0
                ),
                TYPE_DIALOG_HISTORY: (
                    roles.get("assistant", 0) or 0 if call["turn"] != first else 0
                ),
                TYPE_USER: (
                    roles.get("user", 0) or 0 if call["turn"] != first else 0
                ),
            }
            call_total = sum(chars.values())
            if not call_total:
                continue
            shares = turn_shares.setdefault(call["turn"], {})
            for type_name, value in chars.items():
                shares.setdefault(type_name, []).append(value / call_total * 100)

    if turn_shares:
        lines.append("")
        lines.append("Тренд долей по ходам (среднее по прогонам с этим ходом):")
        for turn in sorted(turn_shares):
            shares = turn_shares[turn]
            parts = "  ".join(
                f"{type_name} {sum(values) / len(values):.0f}%"
                for type_name, values in shares.items()
            )
            lines.append(f"  ход {turn}: {parts}")
    return "\n".join(lines)


# --- Q4: Повторно отправляемые токены (design D7) ---


def render_q4(calls: list[dict]) -> str:
    lines = ["Q4. Повторно отправляемые токены", "─" * 60]
    by_run = group_by_run(calls)

    total_input = 0
    repeated_total = 0
    inter_run_tokens = 0
    runs_counted = 0
    for run_calls in by_run.values():
        with_input = [c for c in run_calls if c["input_tokens"] is not None]
        if not with_input:
            continue
        runs_counted += 1
        inputs = [c["input_tokens"] for c in with_input]
        total_input += sum(inputs)
        # Append-only история: каждый ход ре-отправляет предыдущий промпт —
        # повторными считаются входы всех вызовов, кроме последнего.
        repeated_total += sum(inputs[:-1])
        # Межпрогонный слой: (user+assistant) первого хода / 4 — оценка
        # сверху ре-отправки истории сессии между прогонами (design D7).
        first = run_calls[0]
        if first["roles"] is not None:
            inter_run_tokens += (
                (first["roles"].get("user", 0) or 0)
                + (first["roles"].get("assistant", 0) or 0)
            ) // CHARS_PER_TOKEN

    if not runs_counted:
        lines.append("Нет прогонов с входными токенами под фильтрами — "
                     "данных нет.")
        return "\n".join(lines)

    new_total = total_input - repeated_total
    repeated_share = repeated_total / total_input * 100 if total_input else 0.0
    lines.append(f"Всего входных:           {human_count(total_input):>9}")
    lines.append(f"Новая информация:        {human_count(new_total):>9}")
    lines.append(
        f"Повторная информация:    {human_count(repeated_total):>9}"
        f"  ({repeated_share:.0f}% входных — модель уже видела)"
    )
    lines.append("")
    lines.append(
        f"Межпрогонный слой (оценка сверху): ~{human_count(inter_run_tokens)} "
        f"токенов истории сессии, ре-отправляемой из прогона в прогон "
        f"(ход 1: user+assistant, chars/4)."
    )
    return "\n".join(lines)


# --- Q5: стоимость: фактический биллинг + оценки с кэш-скидкой (design D4
# pin-routerai-flex, спека token-audit) ---


def render_q5(
    calls: list[dict],
    price_input_per_m: float,
    price_output_per_m: float,
    cached_price_multiplier: float,
) -> str:
    """Стоимость в два слоя. Основной — фактический биллинг поставщика
    (сумма billed_cost, ₽ как есть), когда поставщик его отдаёт.
    Справочный — оценки по прайсу (сырая и эффективная с кэш-скидкой)
    с явной пометкой «оценка». Без фактических стоимостей основной
    строкой становится сырая оценка (прежнее поведение)."""
    lines = ["Q5. Стоимость: фактическая и оценки по прайсу", "─" * 60]
    with_input = [c for c in calls if c["input_tokens"] is not None]
    billed_values = [c["billed_cost"] for c in calls if c["billed_cost"] is not None]
    if not with_input and not billed_values:
        lines.append("Нет LLM-вызовов с входными токенами под фильтрами — "
                     "данных нет.")
        return "\n".join(lines)

    if billed_values:
        total_billed = sum(billed_values)
        lines.append(
            f"Фактический биллинг поставщика: {total_billed:.4f} ₽  "
            f"(сумма usage.cost по {len(billed_values)} вызовам; единицы "
            f"поставщика, без конверсии)"
        )
    if not with_input:
        return "\n".join(lines)

    total_input = sum(c["input_tokens"] or 0 for c in calls)
    total_output = sum(c["output_tokens"] or 0 for c in calls)
    total_cached = sum(c["cached_tokens"] or 0 for c in calls)
    has_cache_data = any(c["cached_tokens"] is not None for c in calls)

    cached_price = price_input_per_m * cached_price_multiplier
    raw_cost = (
        total_input * price_input_per_m + total_output * price_output_per_m
    ) / 1_000_000
    effective_cost = (
        (total_input - total_cached) * price_input_per_m
        + total_cached * cached_price
        + total_output * price_output_per_m
    ) / 1_000_000

    # При наличии фактического биллинга оценки уходят в справочный слой
    # с явной пометкой «оценка»; иначе сырая оценка — основная строка.
    raw_label = "Сырая оценка:" if billed_values else "Сырая стоимость:"
    effective_label = (
        "Эффективная оценка:" if billed_values else "Эффективная стоимость:"
    )
    lines.append(
        f"{raw_label:<20} ${raw_cost:.6f}  "
        f"(оценка: входные ${price_input_per_m:g}/1M, "
        f"выходные ${price_output_per_m:g}/1M, "
        f"все входные по полной цене)"
    )
    if has_cache_data:
        share = total_cached / total_input * 100 if total_input else 0.0
        lines.append(
            f"{effective_label:<20} ${effective_cost:.6f}  "
            f"(оценка: кэшировано {share:.1f}% входных по "
            f"${cached_price:g}/1M = {cached_price_multiplier:g}× входного)"
        )
        lines.append(f"Экономия кэша:         ${raw_cost - effective_cost:.6f}")
    else:
        lines.append(
            f"{effective_label:<20} ${effective_cost:.6f}  — данных о кэше "
            f"нет (cached_tokens не отдаётся поставщиком), равна сырой"
        )
    lines.append("Цель −30% считается по сырым входным токенам (Q4); "
                 "кэш-скидка и фактический биллинг — справочно.")
    return "\n".join(lines)


# --- Сборка отчёта ---


def render_audit(
    conn: sqlite3.Connection,
    label: str | None,
    since: str | None,
    *,
    price_input_per_m: float = DEFAULT_PRICE_INPUT_PER_M,
    price_output_per_m: float = DEFAULT_PRICE_OUTPUT_PER_M,
    cached_price_multiplier: float = DEFAULT_CACHED_PRICE_MULTIPLIER,
) -> str:
    runs = load_runs(conn, label, since)
    if not runs:
        hint = " Проверьте фильтры --label/--since." if label or since else ""
        return f"В базе телеметрии нет прогонов.{hint}"

    run_ids = [run["id"] for run in runs]
    llm_calls = load_llm_calls(conn, run_ids)
    tool_calls = load_tool_calls(conn, run_ids)

    header = [
        f"Аудит потребления токенов: прогонов {len(runs)}"
        + (f", метка {label!r}" if label else "")
        + (f", с {since}" if since else ""),
        "",
    ]
    return "\n".join(
        header
        + [
            render_q1(tool_calls),
            "",
            render_q2(llm_calls),
            "",
            render_q3(llm_calls),
            "",
            render_q4(llm_calls),
            "",
            render_q5(
                llm_calls,
                price_input_per_m,
                price_output_per_m,
                cached_price_multiplier,
            ),
        ]
    )


def resolve_db_path(explicit: str | None) -> str:
    return os.path.expanduser(explicit or os.getenv("OBS_DB_PATH")
                              or DEFAULT_OBS_DB_PATH)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Аудит потребления токенов по БД телеметрии "
                    "(четыре вопроса Части 2)."
    )
    parser.add_argument("--db", help="путь к БД телеметрии (по умолчанию OBS_DB_PATH)")
    parser.add_argument("--label", help="фильтр прогонов по метке")
    parser.add_argument("--since", help="прогоны, начатые с даты (ISO, напр. 2026-09-01)")
    parser.add_argument("--price-input-per-m", type=float,
                        default=DEFAULT_PRICE_INPUT_PER_M,
                        help=f"цена входных токенов $/1M для расчёта стоимостей "
                             f"(по умолчанию {DEFAULT_PRICE_INPUT_PER_M})")
    parser.add_argument("--price-output-per-m", type=float,
                        default=DEFAULT_PRICE_OUTPUT_PER_M,
                        help=f"цена выходных токенов $/1M "
                             f"(по умолчанию {DEFAULT_PRICE_OUTPUT_PER_M})")
    parser.add_argument("--cached-price-multiplier", type=float,
                        default=DEFAULT_CACHED_PRICE_MULTIPLIER,
                        help="доля входной цены для кэшированных токенов "
                             "в эффективной стоимости "
                             f"(по умолчанию {DEFAULT_CACHED_PRICE_MULTIPLIER})")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path = resolve_db_path(args.db)
    if not os.path.exists(db_path):
        print(f"БД телеметрии не найдена: {db_path}")
        print("Укажите путь через --db (например, БД бенчмарка), "
              "или запустите бота/бенчмарк — база создаётся при записи.")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        print(render_audit(
            conn,
            args.label,
            args.since,
            price_input_per_m=args.price_input_per_m,
            price_output_per_m=args.price_output_per_m,
            cached_price_multiplier=args.cached_price_multiplier,
        ))
    except sqlite3.Error as exc:
        print(f"Не удалось прочитать БД телеметрии ({db_path}): {exc}")
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
