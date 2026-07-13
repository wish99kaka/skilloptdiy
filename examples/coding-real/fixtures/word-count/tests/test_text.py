import unittest

from app.text import word_count


class TextTests(unittest.TestCase):
    def test_counts_words_not_characters(self):
        self.assertEqual(word_count("one two three"), 3)
        self.assertEqual(word_count(" spaced   words "), 2)


if __name__ == "__main__":
    unittest.main()

