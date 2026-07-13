import unittest

from app.text_tools import contains_word


class TextToolsTests(unittest.TestCase):
    def test_finds_space_separated_word(self):
        self.assertTrue(contains_word("red blue green", "blue"))


if __name__ == "__main__":
    unittest.main()

