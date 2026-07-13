import unittest

from app.ranking import sort_by_score


class HiddenSortByScoreTests(unittest.TestCase):
    def test_preserves_input_order_for_equal_scores(self) -> None:
        rows = [
            {"name": "first", "score": 2},
            {"name": "second", "score": 2},
            {"name": "third", "score": 1},
        ]
        self.assertEqual(sort_by_score(rows), rows)

    def test_treats_missing_score_as_zero(self) -> None:
        rows = [{"name": "missing"}, {"name": "positive", "score": 1}]
        self.assertEqual(sort_by_score(rows), [{"name": "positive", "score": 1}, {"name": "missing"}])


if __name__ == "__main__":
    unittest.main()
