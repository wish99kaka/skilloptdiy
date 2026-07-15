import hashlib
import json
import unittest
from pathlib import Path

from textskill_optimizer.paper import OptimizerStage
from textskill_optimizer.paper.prompts import load_optimizer_prompt


EXPECTED = {
    OptimizerStage.REFLECT_FAILURE: (
        "skillopt/prompts/analyst_error.md",
        "40eaf561c7fdbdfc1c7a1febd9f7c3540270ca6d5e7e0a8798c7c90b8d303200",
    ),
    OptimizerStage.REFLECT_SUCCESS: (
        "skillopt/prompts/analyst_success.md",
        "0ffc8f85e13af2b14e1b9f3307a44d30ecf60e9d803ccb9190fdd9785e43fabc",
    ),
    OptimizerStage.MERGE_FAILURE: (
        "skillopt/prompts/merge_failure.md",
        "071313895da226840164ccda6e7f01d5f849ea128037d3f337cddab66ae799db",
    ),
    OptimizerStage.MERGE_SUCCESS: (
        "skillopt/prompts/merge_success.md",
        "6b816bcf5526d31164b779921f3c8c847541bf92fbe26c6a38b555eb3b753712",
    ),
    OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED: (
        "skillopt/prompts/merge_final.md",
        "b2c7b452032fbbe3b223e877b766ea6202775d685c3b32e68956a402e775b84b",
    ),
    OptimizerStage.RANK_TOP_L: (
        "skillopt/prompts/ranking.md",
        "8bf37725b7b328d8d54af7a8fb173179bdd0f02b57399ed2cc7a5b5d25edea8a",
    ),
}


class OfficialPaperPromptTests(unittest.TestCase):
    def test_bundled_fast_loop_prompts_match_locked_official_bytes(self) -> None:
        actual = {
            stage: hashlib.sha256(load_optimizer_prompt(stage).encode()).hexdigest()
            for stage in EXPECTED
        }

        self.assertEqual(actual, {stage: value[1] for stage, value in EXPECTED.items()})

    def test_source_lock_lists_every_reused_prompt_with_the_same_hash(self) -> None:
        source_lock = json.loads(
            (
                Path(__file__).parents[2] / "docs/papers/source-lock.json"
            ).read_text(encoding="utf-8")
        )
        reused = {
            item["path"]: item["sha256"]
            for item in source_lock["official_reference"]["reused_files"]
        }

        self.assertEqual(reused, dict(EXPECTED.values()))


if __name__ == "__main__":
    unittest.main()
