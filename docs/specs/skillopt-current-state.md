# SkillOpt Current State

Date: 2026-07-13

## Objective

Build a paper-faithful SkillOpt implementation and produce claims whose scope is determined by algorithm conformance, data provenance, and measured evidence. Preserve the current contract-aware optimizer as a separate extension rather than treating it as the paper baseline.

## Frozen Decisions

- Target agent: Coco local default. Do not override or modify the Coco model.
- Optimizer: external OpenAI-compatible model editor.
- Historical extension benchmark: `coding-hidden-v2`.
- Locked test: the reviewed Stage 7 attempt completed once on 2026-07-13. Never rerun this split or reuse it for a paper-faithful held-out claim.
- Active roadmap: `docs/specs/skillopt-paper-faithful-roadmap.md`.
- Source precedence: paper first, official implementation second, local extensions third.
- Current token accounting scope: project executor usage and optimizer API usage. Coco/CCR/Kilo token usage is explicitly out of scope for now.
- Targeted smoke runs are mechanism checks only. They validate proposal/rejection/control-loop behavior on selected tasks; they do not prove full `coding-hidden-v2` benchmark effectiveness.
- Cached baselines may be used for low-cost smoke score comparison only. They must not be used for final cost comparisons because their duration and usage are artifact-derived, not same-run measurements.

## Local Paper Source

- Paper PDF: `docs/papers/skillopt-2605.23904.pdf`
- Paper index: `docs/papers/README.md`

## Implemented

- External editor path through `examples/coding/openai_compatible_skill_editor.py`.
- Executive optimizer with bounded atomic edits, learning-rate schedule, rejected buffer, validation gate, slow update, meta skill, checkpoint, and early stop.
- `coding-hidden-v2` builder/validator with contract tags and contract macro accuracy.
- Experiment runner with manifest validation, start/status/wait/report, no Coco model override, and compact reporting.
- Auditable development gate persisted in `summary.json` and mirrored in `runner_report.json`.
- Contract-aware validation gate evidence for new executive runs, including current/candidate contract breakdowns, per-contract deltas, and rejected-buffer metadata.
- External editor prompt now receives compact `contract_rejection_evidence` and requires proposal metadata for targeted contracts, evidence source, and expected behavior change.
- Proposal logs now include a non-execution `proposal_targeting_audit` that flags candidates missing targeted-contract metadata when contract evidence is available.
- External editor now normalizes proposal metadata to `evidence_source=contract_rejection_evidence` when compact contract evidence is available, so source attribution is deterministic instead of model-dependent.
- Executive rejected buffer now uses the recent persisted rejection pool across epochs, not only the current epoch, so later optimizer calls can learn from earlier validation-gate failures.
- Contract rejection evidence now derives a `proposal_policy` with anti-regression contracts and cooldown contracts; proposal audit checks `protected_contracts` and `cooldown_override` metadata for evidence-guided proposals.
- Evidence-guided proposal policy now also prefers one priority contract per proposal unless previously passing priority contracts are explicitly protected.
- Executive ranking now locally penalizes repeated generic contract-audit advice before merging atomic edits, so repeated "verify all contracts" style edits lose priority against more specific alternatives.
- Outcome-aware proposal guards now require evidence-guided proposals to protect currently passing priority contracts, and locally penalize retargeting a recently regressed or unchanged-failed contract unless the proposal declares a new evidence-backed mechanism.
- Matrix runner supports executive-only runs with cached baseline summary rows for low-cost smoke validation.
- Smoke gate script reads existing run artifacts and emits a pass/stop/inconclusive scale-up decision without calling external agents.
- Smoke gate and failure-delta scripts support `--quiet` one-line summaries to avoid dumping full JSON into the control-agent context.
- Compact status script `work/skillopt_compact_status.py` summarizes run completion, seed progress, optimizer token usage, and slow validation gates from artifacts without process-table scans or raw log expansion.
- Development gate schema v2 now includes task-accuracy margin, seed wins, contract-macro non-regression, and optional critical-contract regression blocking.
- Contract effect audit `work/skillopt_contract_effect_audit.py` now checks whether evidence-guided candidates improved targeted contracts without regressing protected or anti-regression contracts.
- Smoke gate now requires proposal metadata audit, contract effect audit, accepted steps, and the hardened development gate before recommending scale-up.
- Runner manifests now label `experiment_stage`; `validation_confirmation_rounds=0` is accepted only for `mechanism_smoke`.
- Mechanical runbook and tooling now provide `work/skillopt_preflight.py`, `work/skillopt_manifest_builder.py`, `work/skillopt_workflow.py`, `work/skillopt_locked_preflight.py`, and centralized `work/skillopt_stage_policy.py`.
- New executive runs append `timing_events.jsonl` with `candidate_written`, `validation_started`, `task_started`, `task_finished`, `validation_gate_written`, and `validation_finished` events. Compact status prefers these events over file mtime gaps, so future runs can separate real agent/scorer time from queueing, resume gaps, and artifact write delays.
- Matrix runner supports targeted train/selection task filters for harder, auditable smoke experiments.
- Targeted smoke runs can disable slow update with `--disable-slow-update` to validate the fast rejected-buffer loop at lower token and time cost.
- Targeted cached baseline summary can be built from existing full-run per-task reports without rerunning baseline agents.
- Recovery path for interrupted executive runs.
- Usage ledger and reports for optimizer calls and executor proxy accounting.
- Adaptive external-agent validation work exists for Coco/CCR/Kilo, but the current SkillOpt target comparison is still Coco.
- Stage 7 tooling now deterministically selects the final development candidate, pins its SHA256 and the locked archive commitment in a manifest, validates readiness without decryption, requires an explicit one-attempt confirmation, and joins the receipt and held-out scores into a final report.
- The locked Coco evaluator requires `CROSS_AGENT_TASKS` from `textskill_optimizer.locked_eval`, rejects partial splits, and records compact task and family metrics plus contract tags when present. The consumed locked split had no contract tags, so its contract metric is only an `unknown_contract` fallback.

## Latest Development Result

Run directory: `runs/coding-hidden-v2-deepseek-runner-v1`

| condition | task accuracy mean | contract macro mean | runs |
| --- | ---: | ---: | ---: |
| no_skill | 0.8000 | 0.7037 | 3 |
| one_shot | 0.8000 | 0.7037 | 3 |
| human_skill | 0.9000 | 0.8519 | 3 |
| executive | 0.8667 | 0.8025 | 3 |

Executive did improve over no skill, but did not beat the human skill baseline.

Executive accepted steps:

- seed-a: 0 accepted / 3 total
- seed-b: 0 accepted / 3 total
- seed-c: 0 accepted / 3 total

Decision: do not run locked test from this state. This is now encoded in `summary.json.development_gate` and `runner_report.json.development_gate`.

## Current Stage Evidence

Stage 5 same-run baseline matrix:

```text
runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop
```

- runner complete; no persistent anomaly
- smoke gate and contract effect audit pass
- executive mean `1.0`; human-skill and one-shot means `0.8667`; no-skill mean `0.8333`
- executive delta over best same-run baseline `+0.1333`; seed wins `3/3`
- accepted protected regressions `0`; rejected protected regressions `2`
- optimizer API total `431247` tokens: executive `422280`, one-shot `8967`
- target-agent tokens remain out of scope

Stage 7 selected `seed-c`: development task accuracy `1.0`, development
contract macro `1.0`, `592` bytes, SHA256
`93f6b57434c532626addf070f992faeede4ed9cd5982248c8eb632b5b559b290`.
The one-attempt locked run completed on 2026-07-13 with return code `0` and
task accuracy `20/20 = 1.0`. The locked test evaluated only the selected
SkillOpt artifact, not same-run held-out baselines, and its task records had no
contract tags. Preserve this as `contract-aware-extension-v1` historical
evidence, not as a paper-faithful or held-out baseline-superiority claim.

## Main Gap Versus Paper

The contract-aware same-target loop and its one locked attempt are complete,
but the paper-faithful implementation and evidence campaign have not begun.

Highest-impact remaining gaps:

1. The current executive algorithm is not paper-faithful: selection contract diagnostics and benchmark-specific mechanisms feed optimization, merge/ranking semantics differ, buffer and slow/meta lifecycles differ, and the paper-default run was not exercised.
2. A separate `paper-faithful-v1` engine, profile, conformance suite, and claim-provenance layer are missing.
3. The consumed `coding-hidden-v2` split cannot support a future paper-faithful held-out claim; a fresh split or official benchmark is required.
4. The local matrix lacks several paper baselines, ablations, benchmark/model/harness breadth, and target-agent token accounting.

## Latest Smoke Result

Run directory: `runs/coding-hidden-v2-deepseek-executive-smoke-v1`

Outcome: stop. Do not scale up from this run.

- runner status: complete
- persistent anomalies: 0
- executive rows: 3
- accepted steps: 0
- executive task accuracy: 1.0000
- best cached baseline: `human_skill`, 1.0000
- development gate: failed because the executive delta was 0.0000 and seed wins were 0/2 required
- proposal audit: not triggered because each seed had only the first proposal and therefore no prior rejected-buffer contract evidence

Interpretation: this smoke configuration was too saturated and too shallow. With `task_limit=1`, cached baselines already score 1.0000, so the development gate cannot show a +0.05 improvement. With only one proposal per seed, the targeted-contract audit cannot trigger because rejected evidence is only produced after the first rejection.

## Revised Smoke Design

Prepared artifacts:

- Targeted cached baseline: `runs/coding-hidden-v2-targeted-baseline-v1/summary.json`
- Revised manifest: `work/experiment_runner_manifest.targeted_executive_smoke.json`

Targeted selection tasks:

- `coding-hidden-v2-selection-allocation-2`
- `coding-hidden-v2-selection-headers-2`

Targeted train tasks:

- `coding-hidden-v2-train-allocation-1`
- `coding-hidden-v2-train-headers-1`

Why this is better:

- Cached human baseline on the selected tasks is 0.5000, not saturated.
- The selected contracts include `largest_remainder`, `input_validation`, `stable_order`, and `unicode_casefold`.
- The revised run uses two train batches in one epoch, so a second proposal can see rejected-buffer contract evidence from the first rejected candidate.
- `early_stop_rejection_limit=0`, so the run does not stop before the second proposal.

## Latest Targeted Smoke Result

Run directory: `runs/coding-hidden-v2-deepseek-targeted-executive-smoke-v2`

Outcome: inconclusive for scale-up. The development gate passed, but the proposal evidence-source audit was not triggered.

- runner status: complete
- persistent anomalies: 0
- accepted steps: 4 total
- executive task accuracy mean: 1.0000
- best cached baseline: `human_skill`, 0.5000
- mean delta: +0.5000
- development gate: passed with 2 required seed wins
- proposal audit: not triggered because no proposal was generated after a rejection carrying contract evidence
- slow update: disabled for this smoke
- optimizer model API usage in this run: 30,210 actual tokens, all from `reflect`

Cost comparison against v1:

- v1 optimizer model API usage: 66,382 actual tokens
- v1 slow meta update usage: 30,639 actual tokens
- v2 optimizer model API usage: 30,210 actual tokens
- reduction: 36,172 actual tokens, about 54.5%

Interpretation: disabling slow update is appropriate for this smoke stage because the run still passed the development gate while using materially fewer optimizer tokens. However, v2 does not prove the run-level evidence-source path because the accepted/rejected sequence never created a later proposal that saw contract rejection evidence. The code-level enforcement and unit tests cover that case, but the experiment protocol still needs one rejection-triggering smoke before treating proposal-source compliance as empirically exercised.

## Latest Rejection-Triggering Smoke Result

Run directory: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v1`

Failure-delta report:

- JSON: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v1/failure_delta_report.json`
- Markdown: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v1/failure_delta_report.md`

Outcome: stop. Do not scale up from this run.

- runner status: complete
- persistent anomalies: 0
- proposal audit: passed
- proposal records with contract evidence: 8
- failed required proposal-audit records: 0
- accepted steps: 2 total
- executive task accuracy mean: 0.6667
- best cached baseline: `human_skill`, 0.5000
- mean delta: +0.1667
- development gate: failed because executive won 1 seed and 2 seed wins are required
- slow update: disabled
- optimizer model API usage in this run: 80,373 actual tokens, all from `reflect`

Interpretation: the evidence-source mechanism is now empirically exercised. Proposals generated after real contract-evidence rejections used `evidence_source=contract_rejection_evidence` and targeted priority contracts. Scale-up is still blocked because evidence-guided edits did not improve enough seeds. The current bottleneck is no longer metadata compliance; it is proposal effectiveness after contract evidence.

Failure-delta interpretation:

- primary blocker: `proposal_effectiveness_after_contract_evidence`
- evidence-guided rejected steps: 7/8
- top evidence-guided regressions: `largest_remainder` 5, `input_validation` 1, `stable_order` 1
- top unchanged failed evidence-guided contracts: `input_validation` 4, `stable_order` 4, `unicode_casefold` 1
- seed outcomes: seed-a tied human baseline, seed-b beat human baseline, seed-c tied human baseline

Decision: do not rerun blindly. The next optimizer change should add contract anti-regression and repeated-target penalties, then rerun a smoke of the same class.

## Latest Policy-Hardened Rejection Smoke Result

Run directory: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v2`

Failure-delta report:

- JSON: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v2/failure_delta_report.json`
- Markdown: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v2/failure_delta_report.md`

Outcome: stop. Do not scale up from this run.

- runner status: complete
- persistent anomalies: 0
- proposal audit: passed
- proposal records with contract evidence: 8
- failed required proposal-audit records: 0
- accepted steps: 1 total
- executive task accuracy mean: 0.5000
- best cached baseline: `human_skill`, 0.5000
- mean delta: 0.0000
- development gate: failed because mean delta was below +0.0500 and executive won 1 seed where 2 are required
- optimizer model API usage in this run: 92,215 actual tokens, all from `reflect`

Policy interpretation:

- The new anti-regression/cooldown policy was exercised in proposal logs.
- `largest_remainder` regressions dropped from 5 in v1 to 0 in v2, so the anti-regression direction had a real effect.
- However, evidence-guided rejected steps worsened from 7/8 to 8/8.
- The dominant failure shifted from regression to no improvement: `largest_remainder` unchanged failed 8 times, `input_validation` 7 times, `stable_order` 7 times.
- Accepted steps dropped from 2 to 1 and optimizer API usage increased from 80,373 to 92,215 tokens.

Decision: do not keep tightening metadata-only proposal policy. The next change should alter proposal generation and ranking toward one small, measurable, single-contract edit at a time, with local de-duplication of semantically repeated contract-audit advice.

## Latest Single-Contract Rejection Smoke Result

Run directory: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v3`

Failure-delta report:

- JSON: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v3/failure_delta_report.json`
- Markdown: `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v3/failure_delta_report.md`

Outcome: stop. Do not scale up from this run.

- runner status: complete
- persistent anomalies: 0
- proposal audit: passed
- proposal records with contract evidence: 6
- failed required proposal-audit records: 0
- accepted steps: 1 total
- executive task accuracy mean: 0.5000
- best cached baseline: `human_skill`, 0.5000
- mean delta: 0.0000
- development gate: failed because mean delta was below +0.0500 and executive won 1 seed where 2 are required
- optimizer model API usage in this run: 89,866 actual tokens, all from `reflect`

Operational note: the first v3 attempt timed out in the external editor at 300 seconds before `summary.json` was produced. The resume manifest `work/experiment_runner_manifest.targeted_rejection_smoke_v3_resume.json` completed the same run directory with only the external editor timeout raised to 600 seconds. It did not override or modify the Coco target model.

Failure-delta interpretation:

- primary blocker: `proposal_effectiveness_after_contract_evidence`
- evidence-guided rejected steps: 6/6
- top evidence-guided regressions: `largest_remainder` 3
- top unchanged failed evidence-guided contracts: `input_validation` 6, `stable_order` 6, `largest_remainder` 3
- seed outcomes: seed-a tied human baseline, seed-b beat human baseline, seed-c lost to human baseline

Decision: single-contract targeting and local generic-duplicate penalties did not improve over v2. Metadata compliance is not the bottleneck. The next change should be outcome-aware: protect already-passing contracts for any evidence-guided proposal, and penalize or reject repeated targeting of a contract that recently produced `unchanged_failed` or `regressed` outcomes unless the proposal states a genuinely new mechanism.

## Outcome-Aware Guard Implementation

Implemented locally on 2026-06-30:

- `proposal_policy.protected_priority_contracts` is now emitted for priority contracts that the current skill was at least partially passing.
- `proposal_targeting_audit` now fails evidence-guided proposals that omit those protected priority contracts, even when the proposal targets only one contract.
- Executive local ranking now penalizes proposals that retarget a recently regressed or unchanged-failed contract without `cooldown_override` or another explicit new-mechanism declaration.
- The OpenAI-compatible editor prompt now tells the optimizer to treat `cooldown_override` as the new evidence-backed mechanism for retargeting recently failed contracts.
- v4 smoke manifest prepared at `work/experiment_runner_manifest.targeted_rejection_smoke_v4.json`.

Verification:

- `python3 -m pytest tests` passed: 240 tests after protocol hardening and mechanical workflow tooling.
- v4 timeout600 smoke completed at `runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v4-timeout600`.
- v4 result: runner complete, proposal targeting passed, accepted step present, development gate failed because executive won 1 seed where 2 are required, and contract effect audit failed due protected or anti-regression contract regression. Do not scale up from v4.

## Protocol Hardening Addendum

External review identified a real protocol gap: the current development gate is still task-accuracy-first, while contract-level evidence is only recorded and used for optimizer feedback. To avoid accepting superficial task-accuracy gains that hide contract regressions, the development protocol must be hardened before any scale-up claim.

Implemented locally on 2026-06-30:

- Development gate combines `task_accuracy_mean`, `contract_macro_mean`, and optional critical-contract regression checks.
- Contract macro is a hard gate with first-version threshold `contract_macro_delta >= 0.0` versus the best baseline.
- Critical contract regression blocks when criteria specify `critical_contracts`.
- Proposal audit remains a metadata/evidence-use check only; contract effect audit separately verifies targeted-contract improvement and protected-contract non-regression.
- `validation_confirmation_rounds=0` is allowed only for `mechanism_smoke` runner manifests. Scale-up manifests must use at least `validation_confirmation_rounds >= 1`.
- Full benchmark claims require a full-selection executive-only run before any complete same-run baseline matrix.
- Final cost claims require same-run baselines. Cached baselines are not cost evidence.
- Locked test remains one-time and blocked until full-selection, proposal audit, contract effect audit, and hardened development gate all pass.

Recheck of the latest completed targeted rejection smoke (`runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v3`) under the new artifact-only audits:

- `proposal_targeting_audit`: pass, 0/6 required records failed.
- `contract_effect_audit`: fail, 3/6 evidence-guided steps improved a target, but 3 protected/anti-regression regressions were detected.
- `smoke_gate`: stop, because the development gate still failed; effect audit is also failed.

Recheck of the outcome-aware v4 timeout600 smoke (`runs/coding-hidden-v2-deepseek-targeted-rejection-smoke-v4-timeout600`):

- runner status: complete
- proposal targeting audit: pass
- accepted step present: true
- executive task accuracy mean: 0.6667
- best cached baseline: `human_skill`, 0.5000
- mean delta: +0.1667
- contract macro delta: +0.1667
- development gate: failed because executive won 1 seed where 2 are required
- contract effect audit: fail because a protected or anti-regression contract regressed
- smoke gate: stop
- optimizer model API usage: 86,261 actual tokens

## Root Cause Hypothesis

The optimizer receives enough trajectory evidence and now satisfies proposal metadata audits, but the generated edits still do not reliably change target-agent behavior. Evidence-guided proposals are either no-ops on the failed contract or improve one contract while regressing another already-passing contract. The failed contracts in the latest completed run are concentrated in:

- `largest_remainder`
- `input_validation`
- `stable_order`

Generic rules such as "verify every contract clause" are already present and no longer create measurable improvement.

## Historical Contract-Aware Plan

The completed work below belongs to the contract-aware extension. Its detailed
historical Epic breakdown remains in `docs/specs/skillopt-gap-epics.md`.

Completed:

- Add a machine-readable development gate verdict to matrix summary and runner report.
- Make validation and rejected-buffer records contract-aware for new executive runs.
- Feed contract-aware rejection evidence into the optimizer prompt/meta skill payload.
- Add a cheap proposal audit hook for missing targeted-contract metadata when contract evidence is available.
- Add executive-only matrix mode, cached baseline import, and an executive smoke manifest.
- Add `work/skillopt_smoke_gate.py` so smoke output has a fixed interpretation.
- Run the first executive smoke and record that it stops: 0 accepted steps, saturated cached baseline, targeting audit not triggered.
- Add targeted task filters, targeted baseline builder, and revised targeted executive smoke manifest.
- Run revised targeted smoke and record that it stops with positive signal: 2 accepted steps and +0.1667 mean delta, but only 1/2 required seed wins and one proposal audit failure.
- Tighten editor metadata so contract-evidence proposals emit `evidence_source=contract_rejection_evidence`.
- Add a targeted-smoke option to disable slow update.
- Rerun targeted smoke with slow update disabled and record that it passes the development gate with 4 accepted steps, but remains scale-up-inconclusive because the proposal evidence-source audit was not triggered.
- Persist rejected-buffer evidence across epochs and add a rejection-triggering targeted smoke manifest.
- Run rejection-triggering targeted smoke and record that proposal audit passes, but development gate still fails with 1/2 required seed wins.
- Add `work/skillopt_failure_delta_report.py` and generate v3 JSON/Markdown failure-delta reports.
- Add evidence-guided proposal policy constraints for anti-regression and repeated-target cooldown metadata.
- Rerun rejection-triggering smoke with policy hardening and record that proposal audit passes but development gate regresses to mean delta 0.
- Add single-contract proposal audit and local semantic duplicate penalties for repeated generic contract-audit advice.
- Rerun rejection-triggering smoke with single-contract targeting and duplicate penalties; record that proposal audit still passes, but the development gate remains failed with mean delta 0 and 1/2 required seed wins.
- Add outcome-aware proposal guards for protected priority contracts and recent failed-target retargeting.
- Run v4 targeted rejection smoke after injecting `EXTERNAL_LLM_API_KEY`; result stops because seed wins remain insufficient and contract effect audit fails on protected/anti-regression regression.
- Harden evidence-guided candidate acceptance so protected-contract regression is blocked before or at selection acceptance.
- Add mechanical preflight, manifest builder, workflow wrapper, stage policy, locked-test preflight, and the operational runbook.
- Implement hardened development gate with contract-macro non-regression and critical-contract regression support.
- Add contract effect audit and wire it into smoke gate.
- Label mechanism-smoke manifests and block zero-confirmation non-smoke runner manifests.
- Re-evaluate v3 artifacts with contract effect audit; result remains stop because development gate failed and effect audit failed.

Active next work:

1. Use `docs/specs/skillopt-paper-faithful-roadmap.md` as the canonical plan.
2. Preserve the Stage 5/7 run as immutable `contract-aware-extension-v1`
   evidence and do not rerun its locked test.
3. Establish a reproducible repository baseline commit and pin the paper and
   official reference versions.
4. Implement the paper profile, selection/test firewall, claim provenance, and
   conformance tests before adding the new engine.
5. Build the isolated paper patch core, then epoch buffer, gated slow update,
   and optimizer-only meta skill.
6. Run zero-cost conformance before any paid smoke or new benchmark campaign.

## Practical Review Rule

Before any new experiment, answer these questions from this spec:

1. What is the target agent, and are we changing it?
2. What is the validation gate?
3. What new evidence will this run produce that the current run does not?
4. What is the stop condition?

If the answers are unclear, do not run the experiment yet.
