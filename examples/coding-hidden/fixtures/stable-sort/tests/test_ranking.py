import unittest

from app.ranking import sort_by_score


class SortByScoreTests(unittest.TestCase):
    def test_sorts_descending_by_score(self) -> None:
        rows = [{"name": "b", "score": 1}, {"name": "a", "score": 3}]
        self.assertEqual(
            sort_by_score(rows),
            [{"name": "a", "score": 3}, {"name": "b", "score": 1}],
        )


if __name__ == "__main__":
    unittest.main()
