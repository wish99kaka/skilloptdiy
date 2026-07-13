import unittest

from app.parsing import parse_bool


class ParsingTests(unittest.TestCase):
    def test_parses_common_boolean_strings(self):
        self.assertTrue(parse_bool("true"))
        self.assertTrue(parse_bool("YES"))
        self.assertFalse(parse_bool("false"))
        self.assertFalse(parse_bool("0"))


if __name__ == "__main__":
    unittest.main()

