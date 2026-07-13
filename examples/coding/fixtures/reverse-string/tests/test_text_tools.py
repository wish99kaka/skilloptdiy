import unittest

from app.text_tools import reverse_text


class TextToolsTests(unittest.TestCase):
    def test_reverses_text(self):
        self.assertEqual(reverse_text("abc"), "cba")


if __name__ == "__main__":
    unittest.main()

