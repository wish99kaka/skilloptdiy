import unittest

from app.events import sort_events


class HiddenSortEventsTests(unittest.TestCase):
    def test_preserves_order_for_equal_timestamps(self) -> None:
        events = [{"id": "a", "ts": 1}, {"id": "b", "ts": 1}, {"id": "c", "ts": 2}]
        self.assertEqual(sort_events(events), events)

    def test_places_missing_timestamps_last(self) -> None:
        events = [{"id": "missing"}, {"id": "early", "ts": 1}]
        self.assertEqual(sort_events(events), [{"id": "early", "ts": 1}, {"id": "missing"}])


if __name__ == "__main__":
    unittest.main()
