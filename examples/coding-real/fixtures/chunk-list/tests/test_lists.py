import unittest

from app.lists import chunk_list


class ListTests(unittest.TestCase):
    def test_chunks_list_into_fixed_sizes(self):
        self.assertEqual(chunk_list([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])


if __name__ == "__main__":
    unittest.main()

