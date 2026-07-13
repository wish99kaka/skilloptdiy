import unittest

from app.users import dedupe_by_email


class HiddenDedupeByEmailTests(unittest.TestCase):
    def test_removes_non_adjacent_duplicates(self) -> None:
        users = [
            {"email": "a@example.com", "name": "first"},
            {"email": "b@example.com", "name": "second"},
            {"email": "a@example.com", "name": "third"},
        ]
        self.assertEqual(
            dedupe_by_email(users),
            [
                {"email": "a@example.com", "name": "first"},
                {"email": "b@example.com", "name": "second"},
            ],
        )

    def test_keeps_users_without_email_independently(self) -> None:
        users = [
            {"name": "first"},
            {"name": "second"},
            {"email": "a@example.com", "name": "third"},
            {"email": "a@example.com", "name": "fourth"},
        ]
        self.assertEqual(
            dedupe_by_email(users),
            [
                {"name": "first"},
                {"name": "second"},
                {"email": "a@example.com", "name": "third"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
