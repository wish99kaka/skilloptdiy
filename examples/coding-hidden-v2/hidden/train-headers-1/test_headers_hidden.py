import unittest
from app.headers import unique_headers


class HiddenHeaderTests(unittest.TestCase):
    def test_handles_blanks_and_third_duplicates(self):
        self.assertEqual(
            unique_headers(["", "  ", "Name", "name", "NAME"]),
            ["column", "column_2", "Name", "name_2", "NAME_3"],
        )

    def test_uses_unicode_casefold(self):
        self.assertEqual(unique_headers(["Straße", "STRASSE"]), ["Straße", "STRASSE_2"])
