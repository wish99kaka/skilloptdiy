import json
import sys
import tempfile
import unittest
from pathlib import Path

from textskill_optimizer.locked_eval import run_locked_archive, seal_directory


class LockedEvalTests(unittest.TestCase):
    def test_sealed_split_runs_once_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "test.jsonl").write_text('{"id":"locked-task"}\n', encoding="utf-8")
            archive = root / "test.enc"
            key = root / "test.key"
            lock = root / "test.lock.json"
            receipt = root / "receipt.json"

            sealed = seal_directory(
                source,
                archive,
                key,
                lock,
                details={"task_file": "test.jsonl", "task_count": 1},
            )
            command = [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,sys; "
                    "p=pathlib.Path(os.environ['CROSS_AGENT_TASKS']); "
                    "sys.exit(0 if 'locked-task' in p.read_text() else 3)"
                ),
            ]

            returncode = run_locked_archive(archive, key, lock, receipt, command)

            self.assertEqual(returncode, 0)
            self.assertEqual(sealed["details"]["task_count"], 1)
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt_payload["returncode"], 0)
            with self.assertRaisesRegex(RuntimeError, "already consumed"):
                run_locked_archive(archive, key, lock, receipt, command)

    def test_archive_tampering_is_rejected_before_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "test.jsonl").write_text("{}\n", encoding="utf-8")
            archive = root / "test.enc"
            key = root / "test.key"
            lock = root / "test.lock.json"
            receipt = root / "receipt.json"
            seal_directory(source, archive, key, lock)
            archive.write_bytes(archive.read_bytes() + b"tampered")

            with self.assertRaisesRegex(ValueError, "hash does not match"):
                run_locked_archive(archive, key, lock, receipt, [sys.executable, "-c", "pass"])

            self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
