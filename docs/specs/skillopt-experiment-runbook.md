# SkillOpt Experiment Runbook

Date: 2026-07-13

## Purpose

This is the operational entry point for running SkillOpt experiments. It records the execution order, required artifacts, and stop conditions. It does not redefine experiment quality; all decisions must come from the mechanical gate reports and run artifacts.

Background and rationale live in:

- `docs/specs/skillopt-current-state.md`
- `docs/specs/skillopt-gap-epics.md`
- `docs/skillopt-executive-protocol.md`
- `docs/specs/skillopt-operator-handoff.md` for low-Codex-token execution handoffs.

## Fixed Boundaries

- Target agent: Coco local default. Do not override or modify the Coco model.
- Optimizer: an independent OpenAI-compatible editor model. It is injected through environment variables, not written into manifests.
- Benchmark: `coding-hidden-v2`.
- Targeted smoke: mechanism evidence only. It cannot prove full benchmark effectiveness.
- Cached baselines: allowed for smoke score comparison only. They are not cost evidence.
- Locked test: consumed once on 2026-07-13. Never run it again; the historical
  preflight and execution procedure below is retained for audit only.

## Environment Preflight

Use Python 3.10 or newer. On the current workstation, prefer:

```bash
export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"
python3 --version
```

Inject optimizer credentials only into the local shell process:

```bash
# Source these from /Users/bytedance/model_key without printing the key.
export EXTERNAL_LLM_API_KEY="<apiKey>"
export EXTERNAL_LLM_BASE_URL="<baseUrl>"
export EXTERNAL_LLM_MODEL="<model_id>"
```

Use the first previously verified model group from `/Users/bytedance/model_key` unless the experiment explicitly says otherwise. Do not write `apiKey` into manifests, reports, logs, or git.

Before a run:

```bash
python3 work/skillopt_preflight.py \
  --manifest work/experiment_runner_manifest.targeted_rejection_smoke_v5_guarded_net.json \
  --quiet
```

If this fails, fix the preflight failure before starting the experiment. If it passes but Codex reports DNS or network egress failure, run the same workflow from a normal local terminal and inspect the generated artifacts afterward.

## Operator Checklist

Use this checklist before every experiment. Stop at the first failed item.

When another agent or human operator runs the experiment for Codex review, use
`skillopt-operator-handoff.md` instead of pasting raw logs or large JSON files.
The completed Stage 7 handoff is archived in
`docs/ops/skillopt-stage7-operator-readme.md` and `runs/operator-packets/`.
Those instructions are historical and must not be executed again.

1. Confirm fixed boundaries: Coco local default is unchanged and the benchmark is `coding-hidden-v2`. Locked test is out of scope except for an explicitly approved Stage 7 Phase B.
2. Confirm Python is 3.10 or newer.
3. Inject `EXTERNAL_LLM_API_KEY`, `EXTERNAL_LLM_BASE_URL`, and `EXTERNAL_LLM_MODEL` from `/Users/bytedance/model_key` into the shell only.
4. Confirm the manifest stage matches the intended decision:
   - `mechanism_smoke` for targeted smoke.
   - `full_selection_development` for full-selection executive-only.
   - `same_run_baseline_matrix` for final same-run baseline comparison.
   - `locked_test_once` for the reviewed Stage 7 manifest only.
5. Run `work/skillopt_preflight.py --quiet` on the manifest.
6. For targeted smoke, run `work/skillopt_workflow.py run-smoke --manifest <manifest>`.
7. If a run already exists, run `work/skillopt_workflow.py report --run-dir <run-dir>` before reading results.
8. Read `decision.json` first, then `smoke_gate_report.json`, `contract_effect_audit.json`, and `failure_delta_report.md`.
9. Scale up only when `decision.json.decision.scale_up_allowed == true`.
10. For scale-up, ensure `validation_confirmation_rounds >= 1`.
11. For full-selection development, ensure `timeout_seconds >= 43200`.
12. For full-selection executive runs, ensure `--early-stop-validation-score 1.0` is present.
13. Run same-run baselines only after full-selection executive-only passes.
14. Run locked test only when `work/skillopt_locked_preflight.py` reports `allowed`.

## Baseline Preparation

Reuse a targeted cached baseline only when its selected task ids exactly match the smoke manifest.

Cached baseline is only a smoke score comparator. It is never valid evidence for final cost conclusions.

Current targeted baseline:

```text
runs/coding-hidden-v2-targeted-baseline-v1/summary.json
```

Current full-development cached comparator for Stage 4:

```text
runs/coding-hidden-v2-deepseek-runner-v1/summary.json
```

This full-development summary is allowed only as a Stage 4 cached score
comparator. It is not same-run baseline evidence and is not final cost evidence.

If the targeted task set changes, rebuild the baseline:

```bash
python3 work/build_coding_hidden_v2_targeted_baseline.py \
  --source-run-dir runs/coding-hidden-v2-deepseek-runner-v1 \
  --selection-task-ids "<comma-separated-selection-task-ids>" \
  --out runs/<new-targeted-baseline>/summary.json
```

Then generate or update the manifest through `work/skillopt_manifest_builder.py` so stage policy stays centralized.

## Targeted Smoke

Targeted smoke should run through the workflow wrapper:

```bash
python3 work/skillopt_workflow.py run-smoke \
  --manifest work/experiment_runner_manifest.targeted_rejection_smoke_v5_guarded_net.json
```

The workflow writes:

- `preflight_report.json`
- `runner_report.json`
- `smoke_gate_report.json`
- `contract_effect_audit.json`
- `failure_delta_report.json`
- `failure_delta_report.md`
- `compact_status.json`
- `compact_status.txt`
- `decision.json`

See the artifact reference table below for who writes each file and how to read it.

For an existing run, regenerate post-run reports without rerunning agents:

```bash
python3 work/skillopt_workflow.py report \
  --run-dir runs/<run-dir>
```

## Smoke Decision

Read `decision.json` first.

Allowed to consider scale-up only when all are true:

- `runner_report.json.status == "complete"`
- `smoke_gate_report.json.status == "pass"`
- `contract_effect_audit.json.status == "pass"`
- `decision.json.decision.scale_up_allowed == true`

Stop when any of these are true:

- runner failed or did not complete
- persistent anomaly count is nonzero
- no accepted step exists
- proposal targeting audit failed
- development gate failed
- contract effect audit failed
- decision says `stop` or `preflight_failed`

If the result is `inconclusive`, do not scale up. Inspect `failure_delta_report.md` and decide whether the next action is a proposal-generation change or a smaller diagnostic smoke.

## Full-Selection Development

Run full-selection executive-only only after targeted smoke passes the hardened smoke gate and contract effect audit.

Requirements:

- `experiment_stage == "full_selection_development"`
- `--conditions executive`
- `--baseline-summary <cached-baseline-summary>`
- `--validation-confirmation-rounds >= 1`
- Coco model remains local default and unmodified

Use `work/skillopt_manifest_builder.py` rather than copying a JSON manifest by hand.

## Same-Run Baseline Matrix

Run a complete same-run baseline matrix only after full-selection executive-only passes.

Requirements:

- `experiment_stage == "same_run_baseline_matrix"`
- conditions include `no_skill`, `human_skill`, `one_shot`, and `executive`
- `--validation-confirmation-rounds >= 1`
- `--early-stop-validation-score 1.0`
- `timeout_seconds >= 86400`
- no `--baseline-summary`
- no `--train-task-ids`, `--selection-task-ids`, `--task-contracts`, or `--task-limit`
- all rows use the same seeds, task set, target model policy, scorer, confirmation-round policy, retry policy, and environment boundary

Only this stage can support final cost and success-rate comparisons. Cached baselines are not cost evidence.

## Locked Test

Locked test is a one-attempt final evaluation. Do not run it from targeted smoke or from a cached-baseline-only conclusion.

Before any locked test:

```bash
python3 work/skillopt_locked_preflight.py \
  --run-dir runs/<candidate-run-dir>
```

Proceed only when the report status is `allowed`. A blocked report is the source of truth for missing evidence.

The locked preflight must pass all of these checks:

- `runner_report_present`
- `runner_complete`
- `development_gate_passed`
- `smoke_gate_passed`
- `contract_effect_passed`
- `no_persistent_anomaly`
- `actual_optimizer_usage_present`
- `locked_receipt_absent`

The following Stage 7 procedure is historical. It completed once on 2026-07-13;
do not run either command again. It is retained only to explain the archived
receipt and result:

```bash
python3 work/skillopt_stage7.py check \
  --manifest work/skillopt_locked_manifest.stage7_v1.json \
  --quiet
```

Proceed only when this check reports `ready` with zero failures and a separate
Phase B approval has been given. The single allowed execution command is:

```bash
python3 work/skillopt_stage7.py run \
  --manifest work/skillopt_locked_manifest.stage7_v1.json \
  --confirmation CONSUME_LOCKED_TEST_ONCE
```

The controller pins the deterministic selected skill, archive/key/code hashes,
lock metadata, Python runtime, full 20-task split, sanitized Coco local
environment, per-task timeout/retry policy, attempt marker, receipt, result,
usage ledger, and final report. It writes the attempt marker before launching
the locked wrapper. Never retry once that marker exists, even if the receipt is
missing. No whole-command kill timeout is used because it could orphan target
agent children before the receipt is written.

## Cross-Agent Evaluation

Coco, CCR Code, and Kilo can be used for transfer and robustness evidence after same-target improvement is proven. They do not replace the same-target Coco development gate, and their votes do not make locked test eligible.

Label cross-agent results as transfer evidence, not as the primary SkillOpt optimization result.

## Artifact Reference

Read high-level gate artifacts before raw per-candidate artifacts. Raw artifacts are for diagnosis after a gate stops or reports inconclusive.

| Artifact | Writer | Read When | Decision Use |
| --- | --- | --- | --- |
| `preflight_report.json` | `work/skillopt_preflight.py` or workflow | Before a run starts, or when workflow returns `preflight_failed` | Confirms Python, manifest, stage policy, env passthrough, baseline summary, out dir, and task ids. |
| `summary.json` | `work/run_coding_hidden_v2_matrix.py` | After runner completion, mainly through reports | Source of rows, aggregate metrics, development gate, cached-baseline markers, and optimizer usage. Do not use cached rows for cost. |
| `runner_report.json` | `work/experiment_runner.py` or workflow | Immediately after a run, before deeper diagnosis | Confirms run status, anomaly summary, development gate, and locked-test recommendation state. |
| `decision.json` | `work/skillopt_workflow.py` | First artifact to read after workflow | Single operator decision: `scale_up_candidate`, `stop`, `inconclusive`, or `preflight_failed`. |
| `smoke_gate_report.json` | `work/skillopt_smoke_gate.py` or workflow | After targeted smoke or report regeneration | Combines runner completion, proposal audit, accepted-step, development-gate, anomaly, and contract-effect checks. |
| `contract_effect_audit.json` | `work/skillopt_contract_effect_audit.py` or workflow | Whenever smoke gate is not pass, or before scale-up | Separates metadata compliance from real targeted-contract effect and protected-contract regression. |
| `failure_delta_report.md` | `work/skillopt_failure_delta_report.py` or workflow | When smoke stops or is inconclusive | Human-readable failure diagnosis: seed outcomes, evidence-guided rejected steps, regressions, unchanged failed contracts, and next action hints. |
| `failure_delta_report.json` | `work/skillopt_failure_delta_report.py` or workflow | When writing automation or auditing exact counts | Machine-readable version of the failure diagnosis. |
| `compact_status.txt` | `work/skillopt_compact_status.py` or workflow | During handoff or quick status checks | Compact run status, seed progress, optimizer usage, and slow validation gates. |
| `compact_status.json` | `work/skillopt_compact_status.py` or workflow | When another script needs compact status | Machine-readable compact status. |
| `proposals.jsonl` | `textskill_optimizer.command_editor` through matrix runner | Only after proposal audit or effect audit needs diagnosis | Proposal log. Inspect evidence source, targeted contracts, protected contracts, cooldown metadata, and rejected-buffer context. |
| `rejected_buffer.jsonl` | executive optimizer | When optimizer keeps repeating weak edits or proposal evidence looks wrong | Shows rejected candidates passed into later optimizer calls, including validation failure reasons and contract evidence. |
| `selection_*_gate.json` | executive optimizer validation gate | When a candidate is rejected or contract effect audit fails | Per-candidate acceptance record, including current/candidate scores, contract evidence, and contract policy guard. |
| `usage_ledger.jsonl` | coding runner and command editor | For usage accounting and optimizer-call verification | Confirms actual optimizer API usage. Same-run cost claims must come from same-run conditions, not cached baselines. |
| `timing_events.jsonl` | executive optimizer | When run time, queueing, or validation latency is unclear | Separates candidate write, validation start, task start/finish, gate write, and validation finish events. |
| `runner_stdout.txt` / `runner_stderr.txt` | experiment runner | When runner failed before producing summary or reports | Diagnose command, network, timeout, or external editor failures. Do not treat these as gate evidence. |
| `locked_receipt.json` | locked evaluation command | After final locked test attempt only | Proves the one-attempt locked evaluation has been consumed. Its absence is required before locked test. |
| `locked_attempt.json` | `work/skillopt_stage7.py run` | Immediately before the locked subprocess starts | Irreversibly records that Phase B began, including failures that occur before `locked_eval` can write a receipt. |
| `work/skillopt_locked_manifest.stage7_v1.json` | `work/skillopt_stage7.py prepare` | Before Stage 7 Phase A | Pins selected skill, archive commitment, key path, receipt/result paths, full task count, and the exact one-attempt command. |
| `locked_evaluation.json` | `work/run_coding_hidden_v2_locked_eval.py` | After the one locked attempt | Records selected-skill identity and task, family, and contract held-out scores for all 20 tasks. |
| `locked_final_report.json` | `work/skillopt_stage7.py run` | After the one locked attempt | Joins receipt, held-out result, same-run development evidence, usage scope, and remaining paper-gap limitations. |

## Current Snapshot

Latest completed same-run baseline matrix:

```text
runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop
```

Result: runner complete, smoke and contract-effect gates passed, executive mean
`1.0` against same-run human-skill mean `0.8667`, mean delta `+0.1333`, seed
wins `3/3`, and no accepted protected regression.

Locked preflight is `allowed`. Stage 7 manifest preparation selects `seed-c`
and the readiness check is `ready` with zero failures. Next: run Phase A only
through `docs/ops/skillopt-stage7-operator-readme.md`; do not consume the locked
test until its packet is reviewed and Phase B is explicitly approved.
