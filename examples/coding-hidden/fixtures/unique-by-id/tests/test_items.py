import unittest

from app.items import unique_by_id


class UniqueByIdTests(unittest.TestCase):
    def test_keeps_first_item_for_duplicate_id(self) -> None:
        items = [
            {"id": 1, "name": "first"},
            {"id": 1, "name": "second"},
            {"id": 2, "name": "third"},
        ]
        self.assertEqual(
            unique_by_id(items),
            [{"id": 1, "name": "first"}, {"id": 2, "name": "third"}],
        )


if __name__ == "__main__":
    unittest.main()
