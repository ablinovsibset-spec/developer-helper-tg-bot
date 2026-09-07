import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculator import DISCOUNT_RATE, DISCOUNT_THRESHOLD, add, calculate_total, subtract


class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)

    def test_total_without_discount(self):
        self.assertEqual(calculate_total([100, 200]), 300)

    def test_total_applies_discount(self):
        total = calculate_total([600, 500])
        expected = 1100 * (1 - DISCOUNT_RATE)
        self.assertEqual(total, expected)
        self.assertGreaterEqual(1100, DISCOUNT_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
