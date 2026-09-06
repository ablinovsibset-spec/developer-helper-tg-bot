"""Форматирование чека для печати."""


def format_receipt(title: str, items: list[tuple[str, float]], total: float) -> str:
    """Чек: заголовок, строки «название — цена» и итог.

    >>> format_receipt("Кофейня", [("латте", 250)], 250)
    '== Кофейня ==\\nлатте — 250\\nИтого: 250'
    """
    lines = [f"== {title} =="]
    lines.extend(f"{name} — {price}" for name, price in items)
    lines.append(f"Итого: {total}")
    return "\n".join(lines)


def normalize_name(name: str) -> str:
    """Имя позиции с заглавной буквы без лишних пробелов."""
    return " ".join(name.split()).capitalize()
