import unittest
from app.grouping import rollup_amounts


class GroupingTests(unittest.TestCase):
    def test_groups_numeric_values(self):
        rows = [{"team": "a", "points": 3}, {"team": "b", "points": 3}, {"team": "a", "points": 4}]
        self.assertEqual(rollup_amounts(rows, "team", "points"), {"a": 7, "b": 3})


if __name__ == "__main__":
    unittest.main()
