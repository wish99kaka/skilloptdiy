import unittest

from app.versioning import compare_versions


class VersioningTests(unittest.TestCase):
    def test_compares_numeric_version_parts(self):
        self.assertEqual(compare_versions("1.10.0", "1.2.9"), 1)
        self.assertEqual(compare_versions("2.0", "2.0.0"), 0)
        self.assertEqual(compare_versions("3.0.1", "3.0.2"), -1)


if __name__ == "__main__":
    unittest.main()

