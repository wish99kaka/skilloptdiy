import unittest

from app.urltools import parse_query


class UrlToolsTests(unittest.TestCase):
    def test_decodes_values_and_collects_repeated_keys(self):
        self.assertEqual(
            parse_query("tag=red&tag=blue&name=Ada%20Lovelace&empty="),
            {"tag": ["red", "blue"], "name": "Ada Lovelace", "empty": ""},
        )


if __name__ == "__main__":
    unittest.main()

