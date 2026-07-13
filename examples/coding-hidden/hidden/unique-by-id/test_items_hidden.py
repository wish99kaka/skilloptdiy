import unittest

from app.items import unique_by_id


class HiddenUniqueByIdTests(unittest.TestCase):
    def test_removes_non_adjacent_duplicates(self) -> None:
        items = [
            {"id": "a", "name": "first"},
            {"id": "b", "name": "second"},
            {"id": "a", "name": "third"},
        ]
        self.assertEqual(
            unique_by_id(items),
            [{"id": "a", "name": "first"}, {"id": "b", "name": "second"}],
        )

    def test_keeps_items_without_id(self) -> None:
        items = [{"name": "first"}, {"name": "second"}, {"id": 1}, {"id": 1}]
        self.assertEqual(unique_by_id(items), [{"name": "first"}, {"name": "second"}, {"id": 1}])


if __name__ == "__main__":
    unittest.main()
