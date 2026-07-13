import unittest
from app.headers import unique_headers


class HeaderTests(unittest.TestCase):
    def test_suffixes_duplicate_headers(self):
        self.assertEqual(unique_headers(["Name", " name "]), ["Name", "name_2"])


if __name__ == "__main__":
    unittest.main()
