import unittest

from work.build_coding_hidden_v2_targeted_baseline import filter_report


class BuildCodingHiddenV2TargetedBaselineTests(unittest.TestCase):
    def test_filter_report_recomputes_subset_scores(self) -> None:
        report = {
            "name": "selection",
            "results": [
                result("task-a", True),
                result("task-b", False),
                result("task-c", True),
            ],
        }

        filtered = filter_report(report, selection_task_ids={"task-a", "task-b"})

        self.assertEqual(filtered["pass_rate"], 0.5)
        self.assertEqual(filtered["average_score"], 0.5)
        self.assertEqual([item["task"]["id"] for item in filtered["results"]], ["task-a", "task-b"])

    def test_filter_report_requires_all_selected_ids(self) -> None:
        with self.assertRaises(ValueError):
            filter_report({"results": [result("task-a", True)]}, selection_task_ids={"task-a", "task-b"})


def result(task_id: str, success: bool) -> dict:
    return {
        "task": {
            "id": task_id,
            "metadata": {
                "benchmark_family": "allocation",
                "contract_tags": ["largest_remainder"],
            },
        },
        "score": {"success": success, "value": 1.0 if success else 0.0},
    }


if __name__ == "__main__":
    unittest.main()
