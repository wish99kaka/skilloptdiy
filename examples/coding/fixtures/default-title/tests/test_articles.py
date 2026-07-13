import unittest

from app.articles import display_title


class ArticleTests(unittest.TestCase):
    def test_uses_default_title(self):
        self.assertEqual(display_title(""), "Untitled")
        self.assertEqual(display_title("Roadmap"), "Roadmap")


if __name__ == "__main__":
    unittest.main()

