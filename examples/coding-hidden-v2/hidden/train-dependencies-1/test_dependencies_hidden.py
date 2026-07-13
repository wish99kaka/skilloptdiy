import unittest
from app.dependencies import dependency_order


class HiddenDependencyTests(unittest.TestCase):
    def test_includes_referenced_only_nodes_and_stable_ties(self):
        graph = {"build": ["core"], "docs": [], "test": ["core"]}
        self.assertEqual(dependency_order(graph), ["core", "build", "docs", "test"])

    def test_rejects_cycles(self):
        with self.assertRaises(ValueError):
            dependency_order({"a": ["b"], "b": ["a"]})
