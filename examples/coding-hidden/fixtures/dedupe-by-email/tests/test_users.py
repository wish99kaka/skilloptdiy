import unittest

from app.users import dedupe_by_email


class DedupeByEmailTests(unittest.TestCase):
    def test_keeps_first_user_for_duplicate_email(self) -> None:
        users = [
            {"email": "a@example.com", "name": "first"},
            {"email": "a@example.com", "name": "second"},
            {"email": "b@example.com", "name": "third"},
        ]
        self.assertEqual(
            dedupe_by_email(users),
            [
                {"email": "a@example.com", "name": "first"},
                {"email": "b@example.com", "name": "third"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
