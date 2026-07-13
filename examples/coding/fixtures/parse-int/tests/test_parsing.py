import unittest

from app.parsing import parse_count


class ParsingTests(unittest.TestCase):
    def test_empty_count_defaults_to_zero(self):
        self.assertEqual(parse_count(""), 0)
        self.assertEqual(parse_count("7"), 7)


if __name__ == "__main__":
    unittest.main()

