import hashlib
import unittest
from pathlib import Path

from textskill_optimizer.paper import ConsumedSplitRegistry


class ConsumedSplitRegistryTests(unittest.TestCase):
    def test_registered_attempts_and_receipts_are_immutable(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        registry = ConsumedSplitRegistry.load()

        self.assertGreater(len(registry.entries), 0)
        for entry in registry.entries:
            with self.subTest(split_id=entry.split_id):
                attempt = repo_root / entry.attempt_path
                receipt = repo_root / entry.receipt_path
                self.assertEqual(
                    hashlib.sha256(attempt.read_bytes()).hexdigest(),
                    entry.attempt_sha256,
                )
                self.assertEqual(
                    hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    entry.receipt_sha256,
                )


if __name__ == "__main__":
    unittest.main()
