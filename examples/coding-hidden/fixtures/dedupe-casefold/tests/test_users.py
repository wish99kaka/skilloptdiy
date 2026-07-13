import unittest

from app.users import dedupe_emails


class DedupeEmailsTests(unittest.TestCase):
    def test_removes_duplicate_email(self) -> None:
        users = [
            {"email": "a@example.com", "name": "first"},
            {"email": "a@example.com", "name": "second"},
        ]
        self.assertEqual(dedupe_emails(users), [{"email": "a@example.com", "name": "first"}])


if __name__ == "__main__":
    unittest.main()
