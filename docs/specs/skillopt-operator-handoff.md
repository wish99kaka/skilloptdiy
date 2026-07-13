# SkillOpt Operator Handoff

## Purpose

Run SkillOpt experiments without spending Codex context on long execution logs or large JSON artifacts.

Codex owns strategy, code changes, and result review. A runner agent or human operator owns mechanical execution and returns only the result packet defined here.

## Roles

- Planner/reviewer: chooses the next stage, reviews the result packet, and decides whether to change code, rerun, scale up, or stop.
- Operator: runs exactly the requested manifest/workflow, avoids analysis, and reports fixed fields only.

## Non-Negotiables

- Do not override the Coco target model.
- Do not run the consumed `coding-hidden-v2` locked test again. A future locked
  test requires a new untouched split and its own approved preflight.
- Do not print or write `EXTERNAL_LLM_API_KEY`.
- Do not inspect full raw artifacts unless the result packet is insufficient.
- Do not open full `result_checkpoint.json`, full `summary.json`, full `timing_events.jsonl`, or full per-task outputs.
- Do not treat cached baselines as final cost evidence.

## Standard Operator Flow

1. Confirm the manifest and intended stage with the requester.
2. Export Python 3.10 on PATH.
3. Inject optimizer environment from `/Users/bytedance/model_key` into the shell process only.
4. Run preflight.
5. Run the workflow command.
6. If the workflow already ran, regenerate reports with `work/skillopt_workflow.py report --run-dir <run-dir>`.
7. Produce the result packet below.
8. Stop. Do not diagnose unless asked.

## Stage-Specific READMEs

Use this document as the general contract. For a concrete stage, follow the
matching README under `docs/ops/`.

Completed historical handoff:

```text
docs/ops/skillopt-stage7-operator-readme.md
```

Its output templates are in:

```text
docs/ops/templates/
```

## Environment Setup

```bash
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"
export EXTERNAL_LLM_API_KEY="$(awk -F'"' '/apiKey/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_BASE_URL="$(awk -F'"' '/baseUrl/ {print $4; exit}' /Users/bytedance/model_key)"
export EXTERNAL_LLM_MODEL="$(awk -F'"' '/model/ {print $4; exit}' /Users/bytedance/model_key)"
python3 --version
```

Do not echo these environment variables.

## Commands

Preflight:

```bash
python3 work/skillopt_preflight.py --manifest <manifest> --quiet
```

Run the command requested by the planner. For targeted smoke manifests, use:

```bash
python3 work/skillopt_workflow.py run-smoke --manifest <manifest>
```

For later stages, use the exact workflow or runner command supplied with the manifest. Do not improvise flags.

Report existing run without rerunning agents:

```bash
python3 work/skillopt_workflow.py report --run-dir <run-dir>
```

## Result Packet

Return this exact shape. Use `unknown` only when the artifact is missing.

```text
run_dir:
manifest:
command:
decision.status:
decision.reason:
decision.scale_up_allowed:
runner.status:
runner.locked_test_recommended:
smoke.status:
smoke.reason:
contract_effect.status:
contract_effect.reason:
development_gate.passed:
executive_mean:
mean_delta:
seed_wins:
condition_means:
best_baseline_condition:
rows_comparable:
missing_paper_baselines:
protected_regression_count:
rejected_protected_regression_count:
effective_step_count:
optimizer_tokens:
compact_status:
failure_delta_excerpt:
operator_notes:
```

`failure_delta_excerpt` is required only when the workflow stops or is inconclusive. Keep it under 40 lines.

## Low-Context Read Commands

Use these instead of opening large artifacts:

```bash
cat <run-dir>/compact_status.txt

jq '{status:.decision.status, reason:.decision.reason, scale_up_allowed:.decision.scale_up_allowed, runner_status, smoke_gate, contract_effect}' \
  <run-dir>/decision.json

jq '{status, locked_test_recommended:.decision.locked_test_recommended, development_gate:.development_gate}' \
  <run-dir>/runner_report.json

jq '{status, reason, checks, contract_effect_audit:{status:.contract_effect_audit.status, reason:.contract_effect_audit.reason, effective_step_count:.contract_effect_audit.effective_step_count, protected_regression_count:.contract_effect_audit.protected_regression_count}}' \
  <run-dir>/smoke_gate_report.json

jq '{status, reason, required_step_count, effective_step_count, failed_effect_step_count, protected_regression_count}' \
  <run-dir>/contract_effect_audit.json
```

When failed:

```bash
sed -n '1,120p' <run-dir>/failure_delta_report.md
```

## When To Escalate Back To Codex

Escalate with the result packet when any of these happens:

- preflight fails
- workflow returns `stop`, `inconclusive`, or nonzero exit
- required artifacts are missing
- smoke gate fails
- contract effect audit fails
- development gate fails
- persistent anomaly count is nonzero
- operator suspects stale run artifacts

Do not perform root-cause analysis unless Codex asks for one specific artifact.

## Current Stage Boundary

Stage 2 targeted guarded smoke passed in:

```text
runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v5-guarded-net-retry7
```

Stage 3 is frozen as optimizer API usage only.

Stage 4 full-selection executive-only passed in:

```text
runs/coding-hidden-v2-deepseek-full-selection-executive-only-stage4-v3-scorestop
```

After report regeneration, Stage 4 is a scale-up candidate: runner complete,
smoke gate pass, contract effect audit pass, executive mean `1.0` versus cached
best baseline `0.9`, and seed wins `2/2`.

Stage 5 same-run baseline matrix passed in:

```text
runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop
```

Executive mean is `1.0` versus same-run human-skill mean `0.8667`; smoke,
development, and contract-effect gates passed. Stage 6 locked preflight is
`allowed`.

Stage 7 Phase A passed, and Phase B consumed the locked test exactly once on
2026-07-13. The controller returned `0` with `20/20` task accuracy and family
macro `1.0`; all 20 task executions returned `0` without timeout. The task
records had no contract tags, so the `unknown_contract=1.0` fallback must not be
presented as contract-generalization evidence.

Do not execute the Stage 7 handoff again. New execution work follows
`docs/specs/skillopt-paper-faithful-roadmap.md` and begins with Phase 0
provenance work, not another experiment.
