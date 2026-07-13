import unittest
from app.grouping import group_sums


class GroupingTests(unittest.TestCase):
    def test_groups_numeric_values(self):
        rows = [{"team": "a", "points": 2}, {"team": "b", "points": 3}, {"team": "a", "points": 4}]
        self.assertEqual(group_sums(rows, "team", "points"), {"a": 6, "b": 3})


if __name__ == "__main__":
    unittest.main()
