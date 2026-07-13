import unittest

from app.users import dedupe_emails


class HiddenDedupeEmailsTests(unittest.TestCase):
    def test_compares_emails_case_insensitively(self) -> None:
        users = [
            {"email": "Ada@Example.com", "name": "first"},
            {"email": "ada@example.com", "name": "second"},
        ]
        self.assertEqual(dedupe_emails(users), [{"email": "Ada@Example.com", "name": "first"}])

    def test_preserves_users_without_email_independently(self) -> None:
        users = [{"name": "first"}, {"name": "second"}]
        self.assertEqual(dedupe_emails(users), users)


if __name__ == "__main__":
    unittest.main()
