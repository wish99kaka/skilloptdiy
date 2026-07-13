import unittest

from app.data import get_path


class GetPathTests(unittest.TestCase):
    def test_reads_dot_separated_path(self) -> None:
        data = {"user": {"profile": {"name": "Ada"}}}
        self.assertEqual(get_path(data, "user.profile.name"), "Ada")


if __name__ == "__main__":
    unittest.main()
