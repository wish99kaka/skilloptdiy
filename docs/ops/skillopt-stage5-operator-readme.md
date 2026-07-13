# SkillOpt Stage 5 Operator README

> Completed historical procedure. Stage 5 finished and its artifacts are
> frozen as contract-aware evidence. Do not execute the prompts or manifests in
> this file; retain them only for audit.

## Purpose

Run Stage 5 same-run baseline matrix with low Codex context cost.

The operator must produce small review files, not paste raw logs or large JSON. Codex reviews the files after the user reports that the operator is done.

## Source Of Truth

Read first:

- `docs/specs/skillopt-operator-handoff.md`
- `docs/specs/skillopt-experiment-runbook.md`
- `docs/specs/skillopt-paper-gap-plan.md`

Stage 4 passed at:

```text
runs/coding-hidden-v2-deepseek-full-selection-executive-only-stage4-v3-scorestop
```

Stage 5 accounting scope:

```text
optimizer API usage only
```

Do not claim paper-equivalent target-agent cost. Target-agent usage is out of scope unless a later stage explicitly enables it before execution.

## Collaboration Contract

Use two phases.

Phase A: manifest check only. Do not run the long experiment.

Phase B: execution. Run only after the requester explicitly approves the manifest check.

This prevents expensive accidental reruns with cached baselines, targeted filters, stale output directories, or non-comparable rows.

## Requester Prompt

For Phase A, send the operator:

```text
Read docs/ops/skillopt-stage5-operator-readme.md. Do Phase A only.
Use work/experiment_runner_manifest.same_run_baseline_matrix_stage5_v1_scorestop.json.
Create runs/operator-packets/stage5-same-run-baseline-matrix-v1-scorestop/manifest_check.md
and manifest_check.json. Do not run the long experiment. Stop after the manifest
check and tell me the packet path.
```

For Phase B, send only after Codex approves Phase A:

```text
Read docs/ops/skillopt-stage5-operator-readme.md. Execute Phase B for the
approved manifest only. Write result_packet.md and result_packet.json under
runs/operator-packets/stage5-same-run-baseline-matrix-v1-scorestop/. Do not
paste raw logs. Stop after writing the packet files.
```

## Output Files

For each operator task, create:

```text
runs/operator-packets/<packet-id>/manifest_check.md
runs/operator-packets/<packet-id>/manifest_check.json
runs/operator-packets/<packet-id>/result_packet.md
runs/operator-packets/<packet-id>/result_packet.json
```

Use this packet id:

```text
stage5-same-run-baseline-matrix-v1-scorestop
```

If Phase A has not been approved yet, leave `result_packet.*` absent.

## Phase A: Manifest Check

The manifest must satisfy every item:

- `experiment_stage == "same_run_baseline_matrix"`
- `runner_role == "mechanical_execution_only"`
- `timeout_seconds >= 86400`
- `immutable_controls.do_not_change_coco_model == true`
- command includes `--conditions no_skill,human_skill,one_shot,executive`
- command includes `--validation-confirmation-rounds` with value `>= 1`
- command includes `--early-stop-validation-score 1.0`
- command does not include `--baseline-summary`
- command does not include `--train-task-ids`
- command does not include `--selection-task-ids`
- command does not include `--task-contracts`
- command does not include `--task-limit`
- command does not override target model, Coco model, or scorer
- rows use the same seeds, task set, target model policy, scorer, confirmation-round policy, retry policy, and environment boundary
- run dir is fresh, or reports are regenerated before reading
- cost scope is recorded as `optimizer_api_usage_only`

Preferred manifest is already generated:

```text
work/experiment_runner_manifest.same_run_baseline_matrix_stage5_v1_scorestop.json
```

If it must be rebuilt, use:

```bash
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"
export EXTERNAL_LLM_API_KEY="$(awk -F'"' '/apiKey/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_BASE_URL="$(awk -F'"' '/baseUrl/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_MODEL="$(awk -F'"' '/model/ {print $4; exit}' /Users/bytedance/model_key)"
python3 work/skillopt_manifest_builder.py \
  --stage same_run_baseline_matrix \
  --out work/experiment_runner_manifest.same_run_baseline_matrix_stage5_v1_scorestop.json \
  --run-dir runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop \
  --timeout-seconds 86400 \
  --seed-workers 3 \
  --conditions no_skill,human_skill,one_shot,executive \
  --early-stop-validation-score 1.0
```

Manifest check command:

```bash
export EXTERNAL_LLM_API_KEY="$(awk -F'"' '/apiKey/ {print $4; exit}' /Users/bytedance/model_key)"
python3 work/skillopt_preflight.py \
  --manifest work/experiment_runner_manifest.same_run_baseline_matrix_stage5_v1_scorestop.json \
  --quiet
```

Write `manifest_check.md` and `manifest_check.json` using the templates in `docs/ops/templates/`.

For Stage 5, add these manifest-check fields if the template does not already contain them:

```text
stage_same_run_baseline_matrix
timeout_seconds_at_least_86400
conditions_include_no_skill_human_skill_one_shot_executive
baseline_summary_absent
rows_same_seed_task_model_scorer_confirmation_retry_environment
```

## Phase B: Execute Approved Manifest

Only after approval:

Before starting the workflow, verify the execution environment can create the run directory under `runs/`:

```bash
python3 -c 'from pathlib import Path; p=Path("runs/.stage5_write_probe"); p.mkdir(exist_ok=False); p.rmdir()'
```

If this fails, stop and write `result_packet.*` as an operational failure. Do not run the workflow in a sandbox that cannot write under `runs/`.

```bash
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"
export EXTERNAL_LLM_API_KEY="$(awk -F'"' '/apiKey/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_BASE_URL="$(awk -F'"' '/baseUrl/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_MODEL="$(awk -F'"' '/model/ {print $4; exit}' /Users/bytedance/model_key)"
python3 work/skillopt_workflow.py run-smoke \
  --manifest work/experiment_runner_manifest.same_run_baseline_matrix_stage5_v1_scorestop.json
```

Do not echo environment variables.

After completion, if reports are missing but `summary.json` exists:

```bash
python3 work/skillopt_workflow.py report \
  --run-dir runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop
```

Write `result_packet.md` and `result_packet.json` using the templates in `docs/ops/templates/`. If stale `result_packet.*` files already exist from a prior failed Phase B attempt, overwrite them after the new attempt and set `created_at` to the new attempt time.

## Allowed Reads

Use only these unless Codex asks for a specific deeper artifact:

```bash
cat <run-dir>/compact_status.txt

jq '{status:.decision.status, reason:.decision.reason, scale_up_allowed:.decision.scale_up_allowed, runner_status, smoke_gate, contract_effect}' \
  <run-dir>/decision.json

jq '{status, locked_test_recommended:.decision.locked_test_recommended, development_gate:.development_gate, anomaly_summary:.anomaly_summary, aggregates:.aggregates}' \
  <run-dir>/runner_report.json

jq '{status, reason, checks, contract_effect_audit:{status:.contract_effect_audit.status, reason:.contract_effect_audit.reason, effective_step_count:.contract_effect_audit.effective_step_count, protected_regression_count:.contract_effect_audit.protected_regression_count}}' \
  <run-dir>/smoke_gate_report.json

jq '{status, reason, required_step_count, effective_step_count, failed_effect_step_count, protected_regression_count, rejected_protected_regression_count}' \
  <run-dir>/contract_effect_audit.json
```

If failed:

```bash
sed -n '1,120p' <run-dir>/failure_delta_report.md
```

## Stop Conditions

Stop and write packet files when:

- preflight fails
- manifest contains `--baseline-summary`
- manifest contains any targeted filter
- manifest omits one of `no_skill`, `human_skill`, `one_shot`, or `executive`
- workflow exits nonzero
- required reports are missing
- smoke gate, contract effect audit, or development gate fails
- persistent anomaly count is nonzero
- rows are not comparable because seeds, task set, target model policy, scorer, confirmation rounds, retry policy, or environment boundary differ

Do not diagnose beyond the required packet unless Codex asks.
