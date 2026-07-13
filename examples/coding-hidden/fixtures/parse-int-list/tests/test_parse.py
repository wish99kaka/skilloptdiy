import unittest

from app.parse import parse_int_list


class ParseIntListTests(unittest.TestCase):
    def test_parses_comma_separated_ints(self) -> None:
        self.assertEqual(parse_int_list("1,2,3"), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
