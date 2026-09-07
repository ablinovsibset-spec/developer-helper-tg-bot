"""Арифметика чека: суммы и скидки."""


DISCOUNT_THRESHOLD = 1000
DISCOUNT_RATE = 0.05


def add(a: float, b: float) -> float:
    """Сумма двух чисел."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Разность двух чисел."""
    return a - b


def calculate_total(prices: list[float]) -> float:
    """Итог чека: сумма позиций.

    При сумме от DISCOUNT_THRESHOLD должна действовать скидка
    DISCOUNT_RATE (5%). Известный баг: скидка не применяется, функция
    возвращает полную сумму — см. падающий тест test_total_applies_discount.
    """
    total = sum(prices)
    # Баг: нет ветки `if total >= DISCOUNT_THRESHOLD: total *= 1 - DISCOUNT_RATE`
    return total
