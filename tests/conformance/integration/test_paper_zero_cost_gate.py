from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[3]


class PaperZeroCostGateTests(unittest.TestCase):
    def test_complete_gate_process_has_network_guard_active(self) -> None:
        if os.environ.get("TEXTSKILL_ZERO_COST_GATE") != "1":
            self.skipTest("asserted by the complete M6 gate")

        with self.assertRaisesRegex(
            OSError,
            "M6 zero-cost gate blocks external network",
        ):
            socket.create_connection(("127.0.0.1", 9))

    def test_gate_python_processes_cannot_open_network_connections(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(ROOT / "scripts/zero_cost_guard"), str(ROOT))
        )

        completed = subprocess.run(
            (
                sys.executable,
                "-c",
                (
                    "import socket; "
                    "socket.create_connection(('127.0.0.1', 9))"
                ),
            ),
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("M6 zero-cost gate blocks external network", completed.stderr)

    def test_audit_only_gate_emits_a_machine_readable_pass_receipt(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                str(ROOT / "scripts/run_paper_zero_cost_gate.py"),
                "--audit-only",
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt["schema_version"], "paper-zero-cost-gate-v1")
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["external_calls"], 0)
        self.assertTrue(receipt["network_guard_active"])
        self.assertFalse(receipt["paid_experiment_executed"])
        self.assertFalse(receipt["paid_development_authorized"])
        self.assertEqual(receipt["prompt_count"], 18)
        self.assertEqual(
            receipt["test_targets"],
            ["tests/conformance", "tests/provenance"],
        )


if __name__ == "__main__":
    unittest.main()
