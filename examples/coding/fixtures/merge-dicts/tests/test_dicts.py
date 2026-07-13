import unittest

from app.dicts import merge_dicts


class DictTests(unittest.TestCase):
    def test_right_values_override_left(self):
        self.assertEqual(merge_dicts({"a": 1}, {"a": 2, "b": 3}), {"a": 2, "b": 3})


if __name__ == "__main__":
    unittest.main()

