import tempfile
import unittest
from pathlib import Path

from textskill_optimizer.io import load_tasks_jsonl, write_text


class IoTests(unittest.TestCase):
    def test_load_tasks_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            write_text(
                path,
                '{"id":"t1","input":"hello","expected":{"answer":"world"}}\n',
            )

            tasks = load_tasks_jsonl(path)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].id, "t1")
        self.assertEqual(tasks[0].expected, {"answer": "world"})


if __name__ == "__main__":
    unittest.main()

