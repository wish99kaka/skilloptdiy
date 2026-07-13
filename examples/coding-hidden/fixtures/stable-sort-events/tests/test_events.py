import unittest

from app.events import sort_events


class SortEventsTests(unittest.TestCase):
    def test_sorts_by_timestamp_ascending(self) -> None:
        events = [{"id": "b", "ts": 3}, {"id": "a", "ts": 1}]
        self.assertEqual(sort_events(events), [{"id": "a", "ts": 1}, {"id": "b", "ts": 3}])


if __name__ == "__main__":
    unittest.main()
