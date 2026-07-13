import unittest

from app.slug import normalize_slug


class SlugTests(unittest.TestCase):
    def test_replaces_spaces_with_hyphens(self) -> None:
        self.assertEqual(normalize_slug("Hello World"), "hello-world")
        self.assertEqual(normalize_slug("Skill Opt"), "skill-opt")


if __name__ == "__main__":
    unittest.main()
