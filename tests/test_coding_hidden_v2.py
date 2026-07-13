import json
import tempfile
import unittest
from pathlib import Path

from work.build_coding_hidden_v2 import (
    CONTRACT_TAG_VOCAB,
    FAMILY_FUNCTIONS,
    build_benchmark,
    render_benchmark_tasks,
    write_protocol_files,
    write_split,
)
from work.validate_coding_hidden_v2 import validate_benchmark


class CodingHiddenV2Tests(unittest.TestCase):
    def test_split_counts_and_family_coverage(self) -> None:
        tasks = render_benchmark_tasks()

        self.assertEqual(len(tasks["train"]), 10)
        self.assertEqual(len(tasks["selection"]), 10)
        self.assertEqual(len(tasks["test"]), 20)
        for split, expected_per_family in (("train", 1), ("selection", 1), ("test", 2)):
            counts = {family: 0 for family in FAMILY_FUNCTIONS}
            for task in tasks[split]:
                counts[task.family] += 1
            self.assertEqual(set(counts.values()), {expected_per_family})

    def test_development_manifests_resolve_to_generated_fixtures(self) -> None:
        tasks = render_benchmark_tasks()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_split(root, "train", tasks["train"])
            write_split(root, "selection", tasks["selection"])
            write_protocol_files(root, tasks)

            task_ids = set()
            for manifest_name in ("train.jsonl", "selection.jsonl"):
                for line in (root / manifest_name).read_text(encoding="utf-8").splitlines():
                    payload = json.loads(line)
                    self.assertNotIn(payload["id"], task_ids)
                    task_ids.add(payload["id"])
                    fixture = root / payload["metadata"]["repo"]
                    self.assertTrue(fixture.is_dir())
                    self.assertTrue((fixture / "app").is_dir())
                    self.assertTrue((fixture / "tests").is_dir())
                    contract_tags = payload["metadata"].get("contract_tags")
                    self.assertIsInstance(contract_tags, list)
                    self.assertTrue(contract_tags)
                    self.assertTrue(set(contract_tags).issubset(CONTRACT_TAG_VOCAB))

            self.assertEqual(len(task_ids), 20)
            protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["locked_test_tasks"], 20)
            self.assertEqual(protocol["family_count"], 10)
            self.assertEqual(protocol["contract_tags"], sorted(CONTRACT_TAG_VOCAB))
            self.assertIn("contract_macro_accuracy", protocol["scoring_unit"])

    def test_integrated_build_seals_test_and_validates_development_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark"
            key = root / "final-test.key"

            lock = build_benchmark(benchmark, key)
            report = validate_benchmark(benchmark)

            self.assertFalse((benchmark / "test.jsonl").exists())
            self.assertTrue((benchmark / "test.enc").is_file())
            self.assertEqual(lock["details"]["task_count"], 20)
            self.assertEqual(report["checked_development_tasks"], 20)
            self.assertTrue(report["all_initial_fixtures_fail"])


if __name__ == "__main__":
    unittest.main()
