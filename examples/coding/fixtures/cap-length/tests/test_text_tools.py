import unittest

from app.text_tools import cap_length


class TextToolsTests(unittest.TestCase):
    def test_caps_text_length(self):
        self.assertEqual(cap_length("abcdef", 3), "abc")


if __name__ == "__main__":
    unittest.main()

