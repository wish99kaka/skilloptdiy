import unittest
from app.dependencies import build_order


class DependencyTests(unittest.TestCase):
    def test_dependencies_come_first(self):
        graph = {"l": ["b"], "b": []}
        self.assertEqual(build_order(graph), ["b", "l"])


if __name__ == "__main__":
    unittest.main()
