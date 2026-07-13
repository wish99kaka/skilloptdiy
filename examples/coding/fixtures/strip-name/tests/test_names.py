import unittest

from app.names import normalize_name


class NameTests(unittest.TestCase):
    def test_strips_before_lowercasing(self):
        self.assertEqual(normalize_name(" Ada "), "ada")


if __name__ == "__main__":
    unittest.main()

