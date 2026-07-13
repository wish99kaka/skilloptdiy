import unittest
from app.headers import normalize_columns


class HeaderTests(unittest.TestCase):
    def test_suffixes_duplicate_headers(self):
        self.assertEqual(normalize_columns(["Region", " region "]), ["Region", "region_2"])


if __name__ == "__main__":
    unittest.main()
