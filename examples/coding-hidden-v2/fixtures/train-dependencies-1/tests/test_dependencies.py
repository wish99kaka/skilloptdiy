import unittest
from app.dependencies import dependency_order


class DependencyTests(unittest.TestCase):
    def test_dependencies_come_first(self):
        graph = {"k": ["a"], "a": []}
        self.assertEqual(dependency_order(graph), ["a", "k"])


if __name__ == "__main__":
    unittest.main()
