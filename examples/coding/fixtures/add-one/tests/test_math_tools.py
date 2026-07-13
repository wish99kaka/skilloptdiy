import unittest

from app.math_tools import add_one


class MathToolsTests(unittest.TestCase):
    def test_adds_one(self):
        self.assertEqual(add_one(2), 3)
        self.assertEqual(add_one(-1), 0)


if __name__ == "__main__":
    unittest.main()

