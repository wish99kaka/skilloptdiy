import unittest

from app.parse import parse_int_list


class HiddenParseIntListTests(unittest.TestCase):
    def test_ignores_empty_and_malformed_tokens(self) -> None:
        self.assertEqual(parse_int_list(" 1, ,oops,2 "), [1, 2])

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(parse_int_list(""), [])


if __name__ == "__main__":
    unittest.main()
