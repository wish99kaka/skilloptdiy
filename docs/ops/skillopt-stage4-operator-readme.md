# SkillOpt Stage 4 Operator README

## Purpose

Run Stage 4 full-selection executive-only with low Codex context cost.

The operator must produce small review files, not paste raw logs or large JSON. Codex reviews the files after the user reports that the operator is done.

## Source Of Truth

Read first:

- `docs/specs/skillopt-operator-handoff.md`
- `docs/specs/skillopt-experiment-runbook.md`
- `docs/specs/skillopt-paper-gap-plan.md`

Stage 2 passed at:

```text
runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v5-guarded-net-retry7
```

Stage 3 accounting scope for Stage 4:

```text
optimizer API usage only
```

Do not claim paper-equivalent target-agent cost. Target-agent usage is out of scope unless a later stage explicitly enables it before execution.

Previous Stage 4 attempt to avoid:

```text
work/experiment_runner_manifest.full_selection_executive_only_stage4_v1.json
runs/coding-hidden-v2-deepseek-full-selection-executive-only-stage4-v1
runs/operator-packets/stage4-full-selection-executive-only-v1
work/experiment_runner_manifest.full_selection_executive_only_stage4_v2_timeout12h.json
runs/coding-hidden-v2-deepseek-full-selection-executive-only-stage4-v2-timeout12h
runs/operator-packets/stage4-full-selection-executive-only-v2-timeout12h
```

v1 hit the manifest builder's old `timeout_seconds=7200` limit before
`summary.json` was produced. v2 proved the run was progressing, but lacked
validation-score early stop and kept spending after seeds reached selection
score `1.0`. Treat both as operational/configuration evidence only; do not reuse
their manifests or partial run directories for the next Phase B.

Stage 4 cached full-development comparator:

```text
runs/coding-hidden-v2-deepseek-runner-v1/summary.json
```

Why this path is allowed for Stage 4:

- `benchmark == "coding-hidden-v2"`
- `development_only == true`
- `task_limit == null`
- conditions include `no_skill`, `human_skill`, and `one_shot`
- seeds are `seed-a`, `seed-b`, and `seed-c`
- best cached baseline is `human_skill` with mean `0.9`

Use it only as the Stage 4 cached score comparator. It is not same-run baseline
evidence and is not final cost evidence.

## Collaboration Contract

Use two phases.

Phase A: manifest check only. Do not run the long experiment.

Phase B: execution. Run only after the requester explicitly approves the manifest check.

This prevents expensive accidental reruns with targeted filters, stale output directories, or the wrong baseline.

## Requester Prompt

For Phase A, send the operator:

```text
Read docs/ops/skillopt-stage4-operator-readme.md. Do Phase A only.
Use runs/coding-hidden-v2-deepseek-runner-v1/summary.json as the full-development
baseline summary. Create
runs/operator-packets/stage4-full-selection-executive-only-v3-scorestop/manifest_check.md
and manifest_check.json. Do not run the long experiment. Stop after the manifest
check and tell me the packet path.
```

For Phase B, send only after Codex approves Phase A:

```text
Read docs/ops/skillopt-stage4-operator-readme.md. Execute Phase B for the
approved manifest only. Write result_packet.md and result_packet.json under
runs/operator-packets/stage4-full-selection-executive-only-v3-scorestop/. Do
not paste raw logs. Stop after writing the packet files.
```

## Output Files

For each operator task, create:

```text
runs/operator-packets/<packet-id>/manifest_check.md
runs/operator-packets/<packet-id>/manifest_check.json
runs/operator-packets/<packet-id>/result_packet.md
runs/operator-packets/<packet-id>/result_packet.json
```

Use a packet id such as:

```text
stage4-full-selection-executive-only-v3-scorestop
```

If Phase A has not been approved yet, leave `result_packet.*` absent.

## Phase A: Manifest Check

The manifest must satisfy every item:

- `experiment_stage == "full_selection_development"`
- `runner_role == "mechanical_execution_only"`
- `timeout_seconds >= 43200`
- `immutable_controls.do_not_change_coco_model == true`
- command includes `--conditions executive`
- command includes `--validation-confirmation-rounds` with value `>= 1`
- command includes `--early-stop-validation-score 1.0`
- command includes `--baseline-summary`
- command does not include `--train-task-ids`
- command does not include `--selection-task-ids`
- command does not include `--task-contracts`
- command does not include `--task-limit`
- command does not override target model, Coco model, or scorer
- run dir is fresh, or reports are regenerated before reading
- baseline summary is for full development comparison, not the targeted smoke baseline
- cost scope is recorded as `optimizer_api_usage_only`

Preferred manifest builder:

```bash
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"
export EXTERNAL_LLM_API_KEY="$(awk -F'"' '/apiKey/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_BASE_URL="$(awk -F'"' '/baseUrl/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_MODEL="$(awk -F'"' '/model/ {print $4; exit}' /Users/bytedance/model_key)"
python3 work/skillopt_manifest_builder.py \
  --stage full_selection_development \
  --out work/experiment_runner_manifest.full_selection_executive_only_stage4_v3_scorestop.json \
  --run-dir runs/coding-hidden-v2-deepseek-full-selection-executive-only-stage4-v3-scorestop \
  --timeout-seconds 43200 \
  --seed-workers 3 \
  --conditions executive \
  --early-stop-validation-score 1.0 \
  --baseline-summary runs/coding-hidden-v2-deepseek-runner-v1/summary.json
```

If the correct full-development baseline is unclear, stop after writing `manifest_check.*` and set `approved_to_run` to `false`.

Manifest check command:

```bash
python3 work/skillopt_preflight.py \
  --manifest work/experiment_runner_manifest.full_selection_executive_only_stage4_v3_scorestop.json \
  --quiet
```

Write `manifest_check.md` and `manifest_check.json` using the templates in `docs/ops/templates/`.

## Phase B: Execute Approved Manifest

Only after approval:

Before starting the workflow, verify the execution environment can create the
run directory under `runs/`:

```bash
python3 -c 'from pathlib import Path; p=Path("runs/.stage4_write_probe"); p.mkdir(exist_ok=False); p.rmdir()'
```

If this fails, stop and write `result_packet.*` as an operational failure. Do
not run the workflow in a sandbox that cannot write under `runs/`.

```bash
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"
export EXTERNAL_LLM_API_KEY="$(awk -F'"' '/apiKey/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_BASE_URL="$(awk -F'"' '/baseUrl/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_MODEL="$(awk -F'"' '/model/ {print $4; exit}' /Users/bytedance/model_key)"
python3 work/skillopt_workflow.py run-smoke \
  --manifest work/experiment_runner_manifest.full_selection_executive_only_stage4_v3_scorestop.json
```

Do not echo environment variables.

After completion, if reports are missing but `summary.json` exists:

```bash
python3 work/skillopt_workflow.py report \
  --run-dir runs/coding-hidden-v2-deepseek-full-selection-executive-only-stage4-v3-scorestop
```

Write `result_packet.md` and `result_packet.json` using the templates in `docs/ops/templates/`.
If stale `result_packet.*` files already exist from a prior failed Phase B
attempt, overwrite them after the new attempt and set `created_at` to the new
attempt time.

## Allowed Reads

Use only these unless Codex asks for a specific deeper artifact:

```bash
cat <run-dir>/compact_status.txt

jq '{status:.decision.status, reason:.decision.reason, scale_up_allowed:.decision.scale_up_allowed, runner_status, smoke_gate, contract_effect}' \
  <run-dir>/decision.json

jq '{status, locked_test_recommended:.decision.locked_test_recommended, development_gate:.development_gate, anomaly_summary:.anomaly_summary}' \
  <run-dir>/runner_report.json

jq '{status, reason, checks, contract_effect_audit:{status:.contract_effect_audit.status, reason:.contract_effect_audit.reason, effective_step_count:.contract_effect_audit.effective_step_count, protected_regression_count:.contract_effect_audit.protected_regression_count}}' \
  <run-dir>/smoke_gate_report.json

jq '{status, reason, required_step_count, effective_step_count, failed_effect_step_count, protected_regression_count}' \
  <run-dir>/contract_effect_audit.json
```

If failed:

```bash
sed -n '1,120p' <run-dir>/failure_delta_report.md
```

## Stop Conditions

Stop and write packet files when:

- preflight fails
- baseline summary is ambiguous
- manifest contains any targeted filter
- workflow exits nonzero
- required reports are missing
- smoke gate, contract effect audit, or development gate fails
- persistent anomaly count is nonzero

Do not diagnose beyond the required packet unless Codex asks.
