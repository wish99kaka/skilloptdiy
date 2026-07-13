---
id: PLAN-skillopt-paper-gap
sources:
  - skillopt-paper-gap-goals.md
  - skillopt-current-state.md
  - skillopt-experiment-runbook.md
---

# SkillOpt Paper Gap Plan

> Historical plan. This document describes the completed contract-aware
> Stage 4–7 campaign and is no longer the active implementation roadmap. Use
> `docs/specs/skillopt-paper-faithful-roadmap.md` for all new work. The Stage 7
> locked test was consumed once on 2026-07-13 and must not be rerun.

## Purpose

Turn `skillopt-paper-gap-goals.md` into execution order. The plan closes the paper gap by proving behavior-level improvement before expanding evidence volume.

## Decision Principle

Run only the smallest experiment that can change the next decision. The current blocker is not missing SkillOpt loop mechanics; it is whether evidence-guided proposals reliably change Coco target-agent behavior without regressing protected contracts.

## Stage 1: Restore Guarded Smoke Operability

Goal: make v5 guarded smoke or its successor produce fresh, interpretable decision artifacts.

Run through the workflow path in `skillopt-experiment-runbook.md` after injecting optimizer environment from the local key source:

```bash
python3 work/skillopt_preflight.py \
  --manifest work/experiment_runner_manifest.targeted_rejection_smoke_v5_guarded_net.json \
  --quiet

python3 work/skillopt_workflow.py run-smoke \
  --manifest work/experiment_runner_manifest.targeted_rejection_smoke_v5_guarded_net.json
```

Freshness requirement:

- Use a fresh `out_dir`, or explicitly regenerate reports for an existing run before reading it.
- Do not interpret stale artifacts from `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v5-guarded`, which failed before `summary.json`.

Pass condition:

- `preflight_report.json` exists and passed.
- `summary.json`, `runner_report.json`, `decision.json`, `smoke_gate_report.json`, `contract_effect_audit.json`, `compact_status.json`, and `compact_status.txt` exist.
- Runner status is complete.
- Endpoint or DNS failure is absent.
- If the runner completes and the smoke gate stops or is inconclusive, `failure_delta_report.json` and `failure_delta_report.md` exist. If the runner fails before `summary.json`, `decision.json` and `runner_report.json` must explain why failure-delta artifacts cannot be meaningful.

Stop condition:

- The external model endpoint cannot be resolved or called. This is operational failure, not algorithm evidence.
- Required workflow artifacts are missing or stale.

Decision unlocked:

- Whether the guarded candidate mechanism can be evaluated at all.

## Stage 2: Pass Targeted Guarded Smoke

Goal: verify that the guarded proposal path fixes the v4 failure mode.

Pass condition:

- `runner_report.json.status == "complete"`.
- `smoke_gate_report.json.status == "pass"`.
- `contract_effect_audit.json.status == "pass"`.
- `contract_effect_audit.json.protected_regression_count == 0`.
- `contract_effect_audit.json.effective_step_count >= 1`.
- Every accepted required evidence-guided record improves a targeted priority contract without protected-contract regression.
- Rejected required evidence-guided regressions are reported as diagnostics and do not enter the deployed skill.
- `decision.json.decision.scale_up_allowed == true`.
- At least one accepted step exists.

Stop condition:

- Development gate fails.
- Contract effect audit fails.
- Proposal targeting metadata passes but targeted contracts do not improve.
- A protected or anti-regression contract regresses.

Decision unlocked:

- Whether it is rational to spend on full-selection executive-only.

If this stage fails:

- Do not expand benchmark coverage.
- Do not run locked test.
- Change proposal generation, ranking, or candidate acceptance so candidates behave like minimal contract improvements: one measurable target improvement, no protected-contract regression.
- Before rerunning external agents after such a change, run focused tests:

```bash
python3 -m pytest \
  tests/test_skillopt_contract_effect_audit.py \
  tests/test_contract_rejection_evidence.py \
  tests/test_development_gate.py \
  tests/test_skillopt_smoke_gate.py \
  tests/test_executive_optimizer.py
```

## Stage 3: Freeze Accounting And Reporting Scope

Goal: decide what cost evidence will be collected before scale-up spends target-agent time.

Pass condition:

- The next scale-up manifest or run report declares whether final cost claims cover optimizer API usage only, executor proxy accounting, target-agent usage, or a named subset.
- If target-agent usage will be claimed, the collection path is enabled before Stage 4.
- If target-agent usage remains out of scope, final reports must say optimizer-only or executor-proxy scope and must not claim paper-equivalent cost-per-point.

Stop condition:

- The team cannot say what usage scope the next same-run comparison will support.
- Cost comparison would mix same-run executive rows with cached baseline artifacts.

Decision unlocked:

- Whether Stage 4 and Stage 5 can produce usable cost evidence instead of merely accuracy evidence.

## Stage 4: Run Full-Selection Executive-Only

Goal: prove the executive path works on the full `coding-hidden-v2` selection scope.

Requirements:

- `experiment_stage == "full_selection_development"`.
- `--conditions executive`.
- `--validation-confirmation-rounds >= 1`.
- `--early-stop-validation-score 1.0`.
- Coco remains local default and unmodified.
- Use the full current `coding-hidden-v2` development split: 10 train tasks and 10 selection tasks.
- Do not pass `--train-task-ids` or `--selection-task-ids`.
- Use `runs/coding-hidden-v2-deepseek-runner-v1/summary.json` as the cached full-development score comparator.
- Cached baselines may be used only as score comparators at this stage.

Pass condition:

- Hardened development gate passes against the best eligible baseline.
- Contract effect audit passes for accepted evidence-guided steps.
- Contract macro does not regress.
- Required seed wins pass.
- Accepted edits have gate records and proposal records.
- No persistent anomaly is present.

Stop condition:

- Executive fails to beat the best eligible baseline by the required margin.
- Contract effect audit fails.
- Contract macro regresses.
- Seed wins miss the gate.
- Accepted edits are not auditable.
- Full-selection manifest uses targeted filters.

Decision unlocked:

- Whether same-run baselines are worth running.

## Stage 5: Run Same-Run Baseline Matrix

Goal: replace cached smoke comparison with final-ready same-run evidence.

Requirements:

- `experiment_stage == "same_run_baseline_matrix"`.
- Conditions include at least `no_skill`, `human_skill`, `one_shot`, and `executive`.
- If Trace2Skill, TextGrad, GEPA, or EvoSkill are not run, reports must state that baseline coverage remains below the paper.
- All rows use the same seeds, task set, target model policy, scorer, confirmation-round policy, retry policy, and environment boundary.
- `--validation-confirmation-rounds >= 1`.
- Coco remains local default and unmodified.
- No cached baseline rows are used for final claims.

Pass condition:

- Same-run rows confirm executive improvement over the best eligible baseline.
- Usage reports match the accounting scope frozen in Stage 3.
- Reports identify any missing paper baselines explicitly.
- Regenerated `smoke_gate_report.json` and `contract_effect_audit.json` are present for the candidate run directory.

Stop condition:

- Same-run baseline invalidates the executive advantage.
- Cost or usage scope is ambiguous.
- Rows are not comparable because seeds, task set, model policy, scorer, confirmation rounds, retry policy, or environment boundary differ.

Decision unlocked:

- Whether the project has enough development evidence for locked preflight.

## Stage 6: Run Locked Preflight

Goal: decide whether the one-attempt locked test is allowed.

Run only after Stage 5 passes, using the Stage 5 same-run baseline matrix run directory:

```bash
python3 work/skillopt_locked_preflight.py \
  --run-dir runs/<stage-5-same-run-baseline-matrix-dir>
```

Pass condition:

- Locked preflight reports `allowed`.
- The run directory contains regenerated `runner_report.json`, `smoke_gate_report.json`, `contract_effect_audit.json`, actual optimizer usage, and no prior locked receipt.

Stop condition:

- Any required preflight check is blocked: runner report, development gate, smoke gate, contract effect audit, persistent anomaly, actual optimizer usage, or prior locked receipt.

Decision unlocked:

- Whether the locked test may be consumed.

## Stage 7: Consume Locked Test Once And Report Held-Out Result

Goal: close the same-target `coding-hidden-v2` paper gap with final held-out evidence.

Requirements:

- Locked preflight from Stage 6 reports `allowed`.
- Candidate selection is fixed before locked evaluation using task accuracy descending, contract macro accuracy descending, skill bytes ascending, then seed ascending.
- The selected skill path and SHA256, archive commitment, lock metadata, sanitized Coco environment policy, Python runtime, code commitments, external key path and hash, attempt marker, receipt/result/usage paths, expected task count, per-task timeout/retry policy, and exact command are pinned in a reviewed Stage 7 manifest.
- Phase A validates the manifest without decrypting or evaluating the locked split; Phase B requires separate explicit approval.
- The locked test is consumed exactly once.
- The final report cites the locked receipt and the Stage 5 same-run baseline matrix.

Pass condition:

- Locked evaluation completes and writes its receipt.
- Final held-out report states the selected skill, locked score, same-run development evidence, usage scope, and any remaining baseline-coverage limitations.

Stop condition:

- Locked preflight is blocked.
- Locked receipt already exists.
- Locked attempt marker already exists, even if the receipt is missing.
- Selected skill, archive commitment, command, or expected task count differs from the approved Stage 7 manifest.
- Locked result cannot be tied back to the Stage 5 candidate run.
- The first locked command fails. Record the consumed attempt and do not retry.

Decision unlocked:

- Whether the repository can claim a final same-target `coding-hidden-v2` result.

## Stage 8: Expand Toward Paper-Scale Evidence

Goal: broaden evidence after the same-target Coco loop works.

Allowed expansions:

- Cross-agent transfer.
- Cross-harness transfer.
- Cross-benchmark transfer.
- Additional benchmark families.
- Ablations for learning rate, rejected buffer, slow/meta update, and proposal guards.
- Final paper-gap report.

Constraint:

- Transfer and expansion evidence cannot replace the same-target Coco development gate or the locked held-out result.
- Exploratory expansion can begin after Stage 5 passes if clearly labeled, but paper-level final claims require the Stage 7 held-out result.

## Final Outcome And Successor

Stage 5 completed cleanly in:

```text
runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop
```

- runner complete, no timeout, no persistent anomaly
- smoke gate pass
- contract effect audit pass
- executive mean `1.0` versus same-run human-skill mean `0.8667`
- mean delta `+0.1333`
- seed wins `3/3`
- accepted protected regressions `0`
- same-run optimizer API usage `431247` tokens: executive `422280`, one-shot `8967`
- target-agent token usage remains out of scope

Stage 6 locked preflight reports:

```text
locked_preflight status=allowed failed=0 missing=none
```

Stage 7 selected `seed-c` with development score `1.0`, contract macro
`1.0`, `592` bytes, and skill SHA256
`93f6b57434c532626addf070f992faeede4ed9cd5982248c8eb632b5b559b290`.
The approved manifest was:

```text
work/skillopt_locked_manifest.stage7_v1.json
```

Phase A passed with zero readiness failures. Phase B then consumed the locked
test exactly once on 2026-07-13 and completed with return code `0`:

- locked task accuracy: `20/20 = 1.0`
- family macro accuracy: `1.0`
- selected skill and archive hashes match the approved manifest
- all 20 task executions returned `0`; none timed out
- attempt, receipt, evaluation, final report, and usage ledger are present

The locked task records contain no contract tags. The reported
`unknown_contract=1.0` is therefore not contract-generalization evidence. This
result is frozen as historical `contract-aware-extension-v1` evidence and must
never be rerun or relabeled as paper-faithful.

The active next action is Phase 0 of
`docs/specs/skillopt-paper-faithful-roadmap.md`: freeze provenance and establish
the repository baseline before implementing the isolated paper-faithful
protocol.
