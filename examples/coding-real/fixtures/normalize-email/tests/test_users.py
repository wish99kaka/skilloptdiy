import unittest

from app.users import normalize_email


class UserTests(unittest.TestCase):
    def test_strips_and_lowercases_email(self):
        self.assertEqual(normalize_email(" Ada@Example.COM "), "ada@example.com")


if __name__ == "__main__":
    unittest.main()

