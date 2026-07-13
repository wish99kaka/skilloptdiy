import unittest

from app.names import initials


class NameTests(unittest.TestCase):
    def test_returns_uppercase_initials(self):
        self.assertEqual(initials("Ada Lovelace"), "AL")
        self.assertEqual(initials("grace brewster hopper"), "GBH")


if __name__ == "__main__":
    unittest.main()

