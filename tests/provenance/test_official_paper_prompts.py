import hashlib
import json
import unittest
from pathlib import Path

from textskill_optimizer.paper import OptimizerStage
from textskill_optimizer.paper.prompts import load_optimizer_prompt


EXPECTED = (
    (
        OptimizerStage.REFLECT_FAILURE,
        "patch",
        "skillopt/prompts/analyst_error.md",
        "40eaf561c7fdbdfc1c7a1febd9f7c3540270ca6d5e7e0a8798c7c90b8d303200",
    ),
    (
        OptimizerStage.REFLECT_SUCCESS,
        "patch",
        "skillopt/prompts/analyst_success.md",
        "0ffc8f85e13af2b14e1b9f3307a44d30ecf60e9d803ccb9190fdd9785e43fabc",
    ),
    (
        OptimizerStage.MERGE_FAILURE,
        "patch",
        "skillopt/prompts/merge_failure.md",
        "071313895da226840164ccda6e7f01d5f849ea128037d3f337cddab66ae799db",
    ),
    (
        OptimizerStage.MERGE_SUCCESS,
        "patch",
        "skillopt/prompts/merge_success.md",
        "6b816bcf5526d31164b779921f3c8c847541bf92fbe26c6a38b555eb3b753712",
    ),
    (
        OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
        "patch",
        "skillopt/prompts/merge_final.md",
        "b2c7b452032fbbe3b223e877b766ea6202775d685c3b32e68956a402e775b84b",
    ),
    (
        OptimizerStage.RANK_TOP_L,
        "patch",
        "skillopt/prompts/ranking.md",
        "8bf37725b7b328d8d54af7a8fb173179bdd0f02b57399ed2cc7a5b5d25edea8a",
    ),
    (
        OptimizerStage.PROPOSE_SLOW_UPDATE,
        "patch",
        "skillopt/prompts/slow_update.md",
        "7f6f97d2283a370ad072a91ad61e152551f4f4eb4f519c9a1c5c60679b4782f2",
    ),
    (
        OptimizerStage.UPDATE_META_SKILL,
        "patch",
        "skillopt/prompts/meta_skill.md",
        "b2bab7f7635cd2a94348de6adf15ceaa247ddabd0a2a8bb7c53eab5cfca7deba",
    ),
    (
        OptimizerStage.DECIDE_LEARNING_RATE,
        "patch",
        "skillopt/prompts/lr_autonomous.md",
        "667c78c7873a28fae0e6f47c5c6ed556d76dfe337c74da986757c9e4fadb1f0e",
    ),
    (
        OptimizerStage.REFLECT_FAILURE,
        "rewrite_from_suggestions",
        "skillopt/prompts/analyst_error_rewrite.md",
        "c393358b2be63d49bc8400944f17bcfb45f58aed5959cf6a46c2e1c1caefcd1b",
    ),
    (
        OptimizerStage.REFLECT_SUCCESS,
        "rewrite_from_suggestions",
        "skillopt/prompts/analyst_success_rewrite.md",
        "758bfe107d15ba1b5318f7191ea9e965fb09982b12599d62fee9db0d823f22f0",
    ),
    (
        OptimizerStage.MERGE_FAILURE,
        "rewrite_from_suggestions",
        "skillopt/prompts/merge_failure_rewrite.md",
        "e9cd2289d8515bca4561e52d3c8da14a2fe97d1b2e77d50f0d34f2fb257326c1",
    ),
    (
        OptimizerStage.MERGE_SUCCESS,
        "rewrite_from_suggestions",
        "skillopt/prompts/merge_success_rewrite.md",
        "2b293d9a16da199cade8bdade560b12553577ad66aacb0f9de3ae979aa150ab7",
    ),
    (
        OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
        "rewrite_from_suggestions",
        "skillopt/prompts/merge_final_rewrite.md",
        "6a79e19eebe2fbbe20f93a6b864c7c157388bbae168980154c0cdac452bbdfe4",
    ),
    (
        OptimizerStage.RANK_TOP_L,
        "rewrite_from_suggestions",
        "skillopt/prompts/ranking_rewrite.md",
        "2ff8c755188704352cb7dd5c57bbce66c506145a5a5cd212b8881bd09d608031",
    ),
    (
        OptimizerStage.REWRITE_SKILL,
        "rewrite_from_suggestions",
        "skillopt/prompts/rewrite_skill.md",
        "2da412ee1d92ab32e59f8a33e2816a2c315edef5e2906d7628d8d093ae0dc68a",
    ),
)


class OfficialPaperPromptTests(unittest.TestCase):
    def test_bundled_prompts_match_locked_official_bytes(self) -> None:
        actual = {
            path: hashlib.sha256(
                load_optimizer_prompt(stage, update_mode=mode).encode()
            ).hexdigest()
            for stage, mode, path, _ in EXPECTED
        }

        self.assertEqual(
            actual,
            {path: sha256 for _, _, path, sha256 in EXPECTED},
        )

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

        self.assertEqual(
            reused,
            {path: sha256 for _, _, path, sha256 in EXPECTED},
        )


if __name__ == "__main__":
    unittest.main()
