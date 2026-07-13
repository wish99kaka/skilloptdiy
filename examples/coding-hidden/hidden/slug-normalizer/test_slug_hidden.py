import unittest

from app.slug import normalize_slug


class HiddenSlugTests(unittest.TestCase):
    def test_collapses_punctuation_and_repeated_spaces(self) -> None:
        self.assertEqual(normalize_slug("  Hello,   World! "), "hello-world")
        self.assertEqual(normalize_slug("Codex: API v3"), "codex-api-v3")

    def test_collapses_existing_separator_runs(self) -> None:
        self.assertEqual(normalize_slug("Already--Slug"), "already-slug")


if __name__ == "__main__":
    unittest.main()
