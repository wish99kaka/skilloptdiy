import unittest
from app.headers import normalize_columns


class HiddenHeaderTests(unittest.TestCase):
    def test_handles_blanks_and_third_duplicates(self):
        self.assertEqual(
            normalize_columns(["", "  ", "Name", "name", "NAME"]),
            ["column", "column_2", "Name", "name_2", "NAME_3"],
        )

    def test_uses_unicode_casefold(self):
        self.assertEqual(normalize_columns(["Straße", "STRASSE"]), ["Straße", "STRASSE_2"])
