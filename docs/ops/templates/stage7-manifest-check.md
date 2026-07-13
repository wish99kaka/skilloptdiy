# SkillOpt Stage 7 Manifest Check

packet_id: stage7-locked-test-once-v1
stage: locked_test_once
manifest: work/skillopt_locked_manifest.stage7_v1.json
source_run_dir: runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop
created_at:
approved_to_run: false

## Readiness

status:
failed_check_count:
failed_checks:

## Pinned Values

selected_seed: seed-c
selected_skill_sha256: 93f6b57434c532626addf070f992faeede4ed9cd5982248c8eb632b5b559b290
selected_skill_bytes: 592
archive_sha256: d96afab4e903943c55773939d0def139be4f839cd008dffdb41b5db6fe967aad
expected_task_count: 20

## Checks

Record every check from `stage7_readiness.json`. All must be true.

## Proposed Command

```text
python3 work/skillopt_stage7.py run --manifest work/skillopt_locked_manifest.stage7_v1.json --confirmation CONSUME_LOCKED_TEST_ONCE
```

## Operator Notes


