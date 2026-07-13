# SkillOpt Stage 7 Operator README

> Completed historical procedure. Phase B consumed the locked test exactly once
> on 2026-07-13 and returned `0`. Do not run Phase A, Phase B, the controller,
> or the embedded locked evaluator again. Preserve this file only as an audit
> record for the archived packet and receipt.

## Purpose

Record how the `coding-hidden-v2` locked test was consumed exactly once and how
its compact, auditable held-out result was returned.

Stage 7 is irreversible. Phase A only validates the prepared manifest. Phase B
may run only after the requester explicitly approves Phase A.

## Source Of Truth

```text
source run: runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop
manifest: work/skillopt_locked_manifest.stage7_v1.json
packet id: stage7-locked-test-once-v1
```

The prepared manifest pins:

- selected seed: `seed-c`
- selection rule: task accuracy descending, contract macro descending, skill bytes ascending, seed ascending
- selected skill bytes: `592`
- selected skill SHA256: `93f6b57434c532626addf070f992faeede4ed9cd5982248c8eb632b5b559b290`
- locked archive SHA256: `d96afab4e903943c55773939d0def139be4f839cd008dffdb41b5db6fe967aad`
- expected locked task count: `20`
- locked task filename: `test.jsonl`
- attempts: `1`; task retries inside that attempt: `1`
- Coco target policy: local default, no model override

Do not edit or rebuild the manifest during either phase.

## Collaboration Contract

Phase A is read-only readiness review. It must not invoke `skillopt_stage7.py
run`, `textskill_optimizer.locked_eval`, Coco, or any command that creates a
locked receipt.

Phase B is the only locked attempt. Run the approved controller command once.
Never retry, even if the command fails, times out, or produces an incomplete
result. The receipt records the consumed attempt.

## Requester Prompts

Phase A:

```text
Read docs/ops/skillopt-stage7-operator-readme.md. Do Phase A only using
work/skillopt_locked_manifest.stage7_v1.json. Write manifest_check.md,
manifest_check.json, and stage7_readiness.json under
runs/operator-packets/stage7-locked-test-once-v1/. Do not run Phase B,
textskill_optimizer.locked_eval, Coco, or any long experiment. Stop after the
packet files and report approved_to_run plus the packet path.
```

Phase B, only after explicit approval:

```text
Read docs/ops/skillopt-stage7-operator-readme.md. Execute Phase B exactly once
for the approved work/skillopt_locked_manifest.stage7_v1.json. Never retry,
including after failure. Write result_packet.md and result_packet.json under
runs/operator-packets/stage7-locked-test-once-v1/. Do not paste raw logs. Stop
after the packet files.
```

## Output Files

Phase A:

```text
runs/operator-packets/stage7-locked-test-once-v1/manifest_check.md
runs/operator-packets/stage7-locked-test-once-v1/manifest_check.json
runs/operator-packets/stage7-locked-test-once-v1/stage7_readiness.json
```

Phase B:

```text
runs/operator-packets/stage7-locked-test-once-v1/result_packet.md
runs/operator-packets/stage7-locked-test-once-v1/result_packet.json
```

Use the Stage 7 templates under `docs/ops/templates/`.

## Phase A: Readiness Check

Run only:

```bash
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"
python3 work/skillopt_stage7.py check \
  --manifest work/skillopt_locked_manifest.stage7_v1.json \
  --out runs/operator-packets/stage7-locked-test-once-v1/stage7_readiness.json \
  --quiet
```

The output must be:

```text
stage7_check status=ready failed=0 missing=none
```

Confirm all readiness checks pass. Do not open or print the key; the controller
checks its pinned SHA256. Confirm these files are absent:

```text
<source-run>/locked_attempt.json
<source-run>/locked_receipt.json
<source-run>/locked_evaluation.json
<source-run>/locked_final_report.json
<source-run>/locked_usage_ledger.jsonl
```

Set `approved_to_run: true` only when the readiness report is `ready` with zero
failed checks and the manifest values match the pinned values above.

## Phase B: Consume Once

Run exactly this controller command once:

```bash
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"
python3 work/skillopt_stage7.py run \
  --manifest work/skillopt_locked_manifest.stage7_v1.json \
  --confirmation CONSUME_LOCKED_TEST_ONCE
```

Do not invoke the embedded command manually. Do not add task filters, health
checks, target-model overrides, retries, or alternate skills.

The controller removes inherited Coco binary, dry-run, extra-argument,
query/bash-timeout, YOLO, task-limit, and stale task-path overrides, then pins
`COCO_AGENT_TIMEOUT=360`. It also pins the absolute Python interpreter,
key-file hash, and Stage 7 code hashes. No whole-command kill timeout is used,
because killing the wrapper could orphan target-agent children and break receipt
semantics. These policies are part of the reviewed manifest.

After the command returns, read only compact projections:

```bash
jq '{status, selected_candidate:{seed:.selected_candidate.seed, skill_sha256:.selected_candidate.skill_sha256}, locked_result:{status:.locked_result.status, task_count:.locked_result.task_count, task_accuracy:.locked_result.task_accuracy, family_macro_accuracy:.locked_result.family_macro_accuracy, contract_macro_accuracy:.locked_result.contract_macro_accuracy}, execution, usage_scope, remaining_limitations}' \
  runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop/locked_final_report.json

jq '{archive_sha256, started_at, finished_at, returncode, error}' \
  runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop/locked_receipt.json
```

If `locked_attempt.json` exists, the attempt has started and must never be
retried. If the receipt or final report is missing or failed, record the failure
in the packet and stop.

## Stop Conditions

Stop before Phase B when any readiness check fails, the selected skill or
archive hash changes, the key file is missing, any attempt/receipt/result/final
output already exists, or the requester has not explicitly approved Phase B.

Stop after the first Phase B command returns, regardless of status. The locked
test is consumed once the receipt exists.

## Accounting And Claims

The same-run optimizer API total is `431247` tokens: executive `422280` and
one-shot `8967`. Target-agent tokens remain out of scope. The final report must
also state that Trace2Skill, TextGrad, GEPA, and EvoSkill are absent from the
local baseline matrix and that cross-model, cross-harness, and cross-benchmark
evidence remains future work.
