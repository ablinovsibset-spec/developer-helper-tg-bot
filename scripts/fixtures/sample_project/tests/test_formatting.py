import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formatting import format_receipt, normalize_name


class TestFormatting(unittest.TestCase):
    def test_format_receipt(self):
        receipt = format_receipt("Кофейня", [("латте", 250)], 250)
        self.assertIn("== Кофейня ==", receipt)
        self.assertIn("латте — 250", receipt)
        self.assertIn("Итого: 250", receipt)

    def test_normalize_name(self):
        self.assertEqual(normalize_name("  чёрный  кофе "), "Чёрный кофе")


if __name__ == "__main__":
    unittest.main()
