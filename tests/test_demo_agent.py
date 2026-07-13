import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "examples/coding/demo_agent.py"


def load_demo_agent_module():
    spec = importlib.util.spec_from_file_location("demo_agent", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DemoAgentTests(unittest.TestCase):
    def test_exact_marker_still_triggers(self) -> None:
        module = load_demo_agent_module()

        self.assertTrue(module.should_apply_skill("Use a test-guided root-cause loop."))

    def test_semantic_equivalent_rule_triggers(self) -> None:
        module = load_demo_agent_module()
        skill = """When asked to fix code or failing tests without editing tests:
1. Analyze the test failure details to identify the root cause.
2. Locate the relevant source code file.
3. Make a minimal, targeted change to the source code.
"""

        self.assertTrue(module.should_apply_skill(skill))

    def test_external_model_workflow_rule_triggers(self) -> None:
        module = load_demo_agent_module()
        skill = """When asked to fix code without editing tests, follow this structured workflow:
1. Run the provided test command to identify test failures and their detailed error messages.
2. Analyze the test errors to pinpoint the root cause.
3. Locate the relevant source code file containing the function or logic causing the failure.
4. Make a small, targeted implementation change to address the root cause.
5. Re-run the test command to confirm the fix resolves the failures.
"""

        self.assertTrue(module.should_apply_skill(skill))

    def test_external_model_discrepancy_rule_triggers(self) -> None:
        module = load_demo_agent_module()
        skill = """When asked to fix code without editing tests:
1. Analyze the test failure output to understand the discrepancy between expected and actual results.
2. Locate the source code file containing the function, method, or logic responsible for the failing test.
3. Make a small, targeted implementation change to correct the behavior identified in the test failure.
4. Ensure your change directly addresses the issue shown in the test failure.
"""

        self.assertTrue(module.should_apply_skill(skill))

    def test_generic_skill_does_not_trigger(self) -> None:
        module = load_demo_agent_module()

        self.assertFalse(module.should_apply_skill("Make a small implementation change."))


if __name__ == "__main__":
    unittest.main()
