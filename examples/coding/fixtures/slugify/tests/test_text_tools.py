import unittest

from app.text_tools import slugify


class TextToolsTests(unittest.TestCase):
    def test_replaces_spaces_with_hyphens(self):
        self.assertEqual(slugify("Hello World"), "hello-world")


if __name__ == "__main__":
    unittest.main()

