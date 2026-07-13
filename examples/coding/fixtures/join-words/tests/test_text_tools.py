import unittest

from app.text_tools import join_words


class TextToolsTests(unittest.TestCase):
    def test_joins_words_with_spaces(self):
        self.assertEqual(join_words(["hello", "world"]), "hello world")


if __name__ == "__main__":
    unittest.main()

