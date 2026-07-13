# SkillOpt Gap Epics

Date: 2026-06-30

> Historical Epic record for the completed contract-aware campaign. It is not
> the active roadmap and its locked-test conditions are not authorization to
> rerun the consumed split. Use `skillopt-paper-faithful-roadmap.md` for new
> work.

## Purpose

Turn the current paper-gap analysis into implementation Epics that move the project from "SkillOpt-style pipeline works" to "SkillOpt-style optimization result is defensible."

## Decision Principle

Optimize for evidence quality before experiment volume.

The current root problem is not that we need more runs. The current run already shows the important failure: executive improves over no skill but does not beat human skill, and it has zero accepted optimization steps. More full runs without better verdicts and failure attribution would mostly increase cost.

## Non-Goals

- Do not change or override Coco's model.
- Do not run locked test until development gate passes.
- Do not expand to paper-scale benchmark coverage before the optimization loop produces accepted edits.
- Do not count Coco/CCR/Kilo token usage in the primary accounting scope yet.

## Epic 1: Auditable Development Gate

Status: Complete on 2026-06-26.

### Goal

Make the "should we proceed to locked test?" decision machine-readable, reproducible, and visible in every compact report.

### Why

The latest `summary.json` contains rows, aggregate metrics, and usage, but not a first-class development-gate verdict. We can infer that locked test is not recommended, but inference is weaker than an explicit protocol artifact.

### Stories

#### E1.S1: Development Gate Decision Model

As the experiment controller, I need one canonical function that computes the development gate verdict from rows/aggregate/acceptance criteria, so every report makes the same locked-test decision.

Implementation tasks:

- Define a reusable development gate payload.
- Include best baseline condition, baseline score, executive score, score delta, required delta, seed wins, required wins, and pass/fail reason.
- Make `locked_test_recommended` derive from this gate, not from duplicated logic.

Acceptance criteria:

- A failing executive-vs-human result produces a failed gate with an explicit reason.
- A passing result produces a passed gate and `locked_test_recommended=true`.
- Missing executive or missing baseline rows produce a failed gate, not an exception.

#### E1.S2: Matrix Summary Persists Gate Verdict

As a reviewer, I need `summary.json` to contain the development gate verdict, so I do not have to infer the decision from raw rows.

Implementation tasks:

- Add `development_gate` to matrix `summary.json`.
- Add top-level `locked_test_recommended` to matrix `summary.json`.
- Persist the acceptance criteria used to compute the verdict.

Acceptance criteria:

- New matrix summaries contain `development_gate` and `locked_test_recommended`.
- The existing recovered run can be re-reported or regenerated to show the same verdict.
- Unit tests fail if a completed summary omits the gate verdict.

#### E1.S3: Runner Report Mirrors Gate Verdict

As the mechanical runner consumer, I need `runner_report.json` to surface the same gate verdict plus anomaly blocking, so status checks and final review agree.

Implementation tasks:

- Read the summary gate when present.
- Fall back to computing the gate for older summaries.
- Keep anomaly checks as an additional blocker on top of the development gate.
- Report distinct states for missing summary, development gate failed, anomaly blocked, and locked test recommended.

Acceptance criteria:

- `runner_report.json` exposes the same `development_gate` fields as `summary.json`.
- Persistent anomalies can block `locked_test_recommended` without rewriting the development score verdict.
- Existing runner tests cover pass, fail, missing summary, and anomaly-blocked cases.

### Acceptance Criteria

- `runs/.../summary.json` has a stable `development_gate` object.
- `locked_test_recommended` is true only when executive beats the selected baseline by required mean delta and required seed wins.
- Reports distinguish "missing summary", "gate failed", and "gate passed".
- Existing unit tests pass.

### Exit Gate

A completed development run can answer, without human inference: "Should locked test run now?"

## Epic 2: Contract-Aware Rejection Evidence

Status: Implementation complete on 2026-06-26. Applies to newly generated validation gates and recovered results whose gate files contain contract evidence. Existing old gate files remain readable but do not receive fabricated contract deltas.

### Goal

Make validation rejections tell the optimizer which behavioral contracts failed, regressed, or failed to improve.

### Why

Current rejected feedback mostly says `selection_not_improved`. That is too coarse. The latest aggregate gap is concentrated in `largest_remainder`, `input_validation`, and `stable_order`; the optimizer needs this structure as evidence, not as a human side note.

### Stories

#### E2.S1: Contract Evidence Model

As the optimizer, I need a compact contract-level comparison between current and candidate validation reports, so rejection evidence says what behavior changed instead of only saying the score did not improve.

Implementation tasks:

- Compute contract breakdown from an evaluation report.
- Compute per-contract candidate-minus-current deltas.
- Identify top negative contracts and top no-improvement contracts.
- Keep the payload compact and JSON-serializable.

Acceptance criteria:

- A candidate regression on one contract appears as a negative delta.
- A candidate that leaves a failed contract unchanged appears in no-improvement contracts.
- Reports without contract tags fall back to a stable `unknown_contract` bucket.

#### E2.S2: Gate and Rejected Buffer Persistence

As the next optimizer call, I need rejected candidates to carry contract evidence, so the editor can target the actual validation blockers.

Implementation tasks:

- Add contract evidence to `ValidationGateDecision.to_dict()`.
- Write the evidence into `selection_*_gate.json`.
- Include the evidence in history metadata.
- Include the evidence in rejected-buffer metadata for fast and slow updates.

Acceptance criteria:

- Every validation-gate rejection has `metadata.validation_gate.contract_evidence`.
- The standalone gate JSON has the same contract evidence.
- Accepted candidates also retain contract evidence for audit.

#### E2.S3: Recovery and Backward Compatibility

As a reviewer of interrupted or old runs, I need missing contract evidence to be handled explicitly, so recovery does not pretend unavailable evidence exists.

Implementation tasks:

- Preserve contract evidence when gate files already contain it.
- For old gate files without evidence, leave a clear absence rather than fabricating precise deltas.
- Add tests that old artifacts still load and new artifacts include evidence.

Acceptance criteria:

- Existing old runs remain readable.
- Recovered new runs keep contract-aware gate metadata when available.
- Recovery-generated results do not crash when old gates lack contract evidence.

### Acceptance Criteria

- Every validation-gate rejection can explain which contracts blocked acceptance.
- Rejected buffer includes compact contract evidence, not full raw reports.
- Recovery results retain enough contract evidence to remain useful for optimizer feedback.
- Existing reports remain backward compatible for older runs.

### Exit Gate

After a rejected candidate, the next optimizer call receives enough structured evidence to target the actual failing contracts.

## Epic 3: Targeted Optimizer Proposal Mechanism

Status: Complete on 2026-06-26. Contract evidence is surfaced to the external editor, proposals are prompted to declare targeted contracts, and proposal logs include a non-execution targeting audit.

### Goal

Make the editor generate small, evidence-backed edits targeted at the contracts that block validation improvement.

### Why

The current optimizer tends to rediscover generic process rules like "verify every contract clause." Those rules are already present and produced zero accepted steps. The next mechanism must force proposals to explain how they address recorded contract deltas.

### Stories

#### E3.S1: Contract Evidence Payload

As the external editor, I need contract-aware rejection evidence surfaced as a compact first-class payload, so I do not have to infer validation blockers from deep rejected-buffer metadata.

Implementation tasks:

- Extract recent contract evidence from rejected buffer.
- Summarize priority contracts from top negative and no-improvement deltas.
- Send this summary in reflect and slow-meta-update user payloads.
- Keep raw rejected buffer for audit, but make the compact summary the primary signal.

Acceptance criteria:

- Prompt payload includes `contract_rejection_evidence` when rejected buffer has validation-gate contract evidence.
- Payload remains empty/explicitly unavailable for old rejected buffers.
- Summary does not include raw task outputs or full candidate reports.

#### E3.S2: Targeted Proposal Prompt and Metadata

As the optimizer controller, I need proposals to declare which validation contracts they target, so we can audit whether the editor used the evidence.

Implementation tasks:

- Update external editor system prompt to prioritize contract deltas over generic repeated advice.
- Require proposal metadata: `targeted_contracts`, `evidence_source`, and `expected_behavior_change`.
- Update meta skill to avoid repeating generic full-contract-audit edits unless contract evidence supports that direction.
- Add tests with synthetic rejected evidence.

Acceptance criteria:

- With contract evidence available, prompt tells the editor to target at least one priority contract or return no proposals.
- Prompt forbids copying contract tags into a benchmark answer sheet.
- Tests verify targeted metadata is requested and the compact evidence appears in payload.

#### E3.S3: Candidate Audit Hook

As a future experiment reviewer, I need to detect proposals that ignore contract evidence, so low-cost smoke runs can fail fast before full external-agent evaluation.

Implementation tasks:

- Add proposal-log fields or validation helpers that expose targeted-contract metadata.
- Define a non-execution audit for "no targeted contract despite available evidence."
- Keep this as an audit first, not a hard rejection, until smoke data shows it is safe.

Acceptance criteria:

- Proposal logs can answer whether a proposal used contract-aware evidence.
- The audit can fail a smoke review without changing target-agent execution.
- Hard rejection is deferred until we have evidence it does not suppress useful proposals.

### Acceptance Criteria

- Optimizer prompts explicitly include failing contract names and deltas.
- Candidate metadata names targeted contracts.
- Generic duplicate process edits are discouraged when they do not address the current contract gap.
- The mechanism still forbids hidden-answer leakage and task-specific answer rules.

### Exit Gate

A smoke run produces at least one candidate that is visibly targeted at a known failing contract, even before we know whether it passes.

## Epic 4: Cost-Controlled Verification Loop

Status: Protocol, smoke-gate support, failure attribution, proposal-policy hardening, contract-aware development gate, contract effect audit, and scale-up validation-strength checks are complete through local implementation. The first external-agent executive smoke completed and correctly stopped: no accepted steps, saturated cached baseline, and no triggered targeted-contract audit. The first revised targeted smoke completed with positive signal but stopped: 2 accepted steps, +0.1667 mean delta, 1/2 required seed wins, and one proposal audit failure. The second revised targeted smoke disabled slow update and passed the development gate with 4 accepted steps and +0.5000 mean delta, but remained scale-up-inconclusive because no proposal was generated after a contract-evidence rejection. The rejection-triggering targeted smoke passed proposal audit with 8 required evidence-backed proposal records and 0 audit failures, but stopped because the development gate failed with 1/2 required seed wins. Failure-delta attribution identified proposal effectiveness after contract evidence as the blocker. The anti-regression/cooldown policy was then exercised in a second rejection-triggering smoke; proposal audit still passed, but development performance regressed to mean delta 0.0000. Single-contract targeting plus local semantic duplicate penalties were then exercised in v3; proposal audit still passed, but development performance remained at mean delta 0.0000 with 1/2 required seed wins. Outcome-aware v4 completed and correctly stopped: proposal targeting passed and an accepted step existed, but the development gate still failed with 1/2 required seed wins and the contract effect audit failed on protected or anti-regression regression. Evidence-guided candidate guards now reject protected-contract regressions before or at selection acceptance, and mechanical workflow tooling/runbook now centralizes preflight, smoke reporting, effect audit, failure delta, compact status, and locked-test preflight.

### Goal

Run only the smallest experiment that can change the decision at each stage.

### Why

Full external-agent experiments are expensive because each task copies a repo, invokes an agent CLI, lets the agent inspect/edit/test, then runs the scorer. We should not pay that cost until the mechanism has a plausible chance to create accepted edits.

### Stories

#### E4.S1: Executive-Only Matrix Mode

As the experiment controller, I need to run only executive rows, so smoke validation does not rerun no_skill, human_skill, or one_shot baselines.

Implementation tasks:

- Add `--conditions` to the matrix runner.
- Support `--conditions executive`.
- Persist selected conditions in the experiment manifest.

Acceptance criteria:

- The matrix can run only executive rows.
- Existing full-matrix behavior remains the default.

#### E4.S2: Cached Baseline Comparison

As a reviewer, I need an executive-only summary to compare against cached same-protocol baselines, so the normal development gate remains usable.

Implementation tasks:

- Add `--baseline-summary`.
- Import non-executive rows from that summary.
- Mark imported rows with `cached_baseline=true`.
- Exclude cached rows from current-run usage accounting.
- Build targeted cached baseline summaries from existing per-task reports when a smoke run filters selection tasks.

Acceptance criteria:

- Summary contains cached baseline rows plus fresh executive rows.
- Development gate can compare executive against best cached baseline.
- Current-run usage excludes cached baseline ledgers.
- A targeted smoke baseline is computed on the same selected task ids as the executive smoke.

#### E4.S3: Smoke Manifest and Stop Conditions

As the operator, I need a ready manifest and clear stop conditions, so a smoke run has an interpretation before it starts.

Implementation tasks:

- Add `work/experiment_runner_manifest.executive_smoke.json`.
- Keep target model immutable.
- Use existing `task_limit=1` smoke baseline summary for the first artifact-flow smoke.
- Add a revised targeted executive smoke manifest using known failing selection contracts and two train batches in one epoch.
- Define stop conditions:
  - no proposal with targeted-contract metadata: stop before full run
  - no accepted step: do not scale up
  - accepted step but development gate failed: inspect contract deltas
  - development gate passed: consider small multi-seed executive run
- Add a read-only smoke gate script that emits `pass`, `stop`, `inconclusive`, or `missing_artifacts` from existing run artifacts.
- Add a smoke-only `--disable-slow-update` option so fast rejected-buffer behavior can be tested without paying slow meta-update cost.
- Make the executive rejected buffer persist across epochs so a later proposal can learn from earlier validation-gate rejection evidence.
- Add `work/experiment_runner_manifest.targeted_rejection_smoke.json` to exercise the real gate-to-buffer-to-proposal path.
- Add `work/skillopt_failure_delta_report.py` to summarize seed outcomes, evidence-guided step outcomes, contract regressions, unchanged failed contracts, and next actions without rerunning agents.
- Derive evidence-guided `proposal_policy` with anti-regression contracts and cooldown contracts, and audit `protected_contracts` / `cooldown_override` metadata.
- Derive single-contract targeting requirements and audit broad multi-contract proposals unless previously passing priority contracts are protected.
- Penalize repeated generic contract-audit advice locally before atomic edit merging.
- Require protection of currently passing priority contracts for any evidence-guided proposal, not only for broad multi-contract proposals.
- Penalize or reject repeated targeting of contracts that recently produced `unchanged_failed` or `regressed` validation outcomes unless the proposal states a new, evidence-backed mechanism.

Acceptance criteria:

- Manifest validates with the mechanical runner.
- Manifest does not override Coco model.
- Smoke output can be judged through normal `summary.json`, `runner_report.json`, and proposal log audit.
- Missing proposal logs cannot be treated as a pass.
- A saturated baseline and zero accepted steps must stop scale-up even if proposal targeting audit is not triggered.
- Revised smoke manifest validates and does not override Coco model.
- Revised targeted smoke must not scale up unless both proposal audit and seed-win development gate pass.
- Slow update can be disabled for smoke runs without changing the target agent or cached-baseline protocol.
- Rejection-triggering smoke must produce at least one required proposal-audit record before it can validate the contract-evidence path.
- Failure attribution must distinguish metadata/evidence-use failure from evidence-guided proposal-effectiveness failure.
- Proposal audit must fail evidence-guided proposals that omit anti-regression protection or retarget cooldown contracts without a new evidence-backed override.
- Proposal audit must flag broad multi-priority proposals that do not protect previously passing priority contracts.
- Local ranking must prefer specific non-duplicate edits over repeated generic contract-audit advice.
- Outcome-aware guards must reduce avoidable validation spend on proposals that repeat a recently failed target or risk regressing an already-passing priority contract.

#### E4.S4: Contract-Aware Development Gate

Status: Complete locally on 2026-06-30.

As the experiment reviewer, I need the development gate to reject superficial task-accuracy gains that hide contract regressions.

Implementation tasks:

- Extend `work/development_gate.py` criteria with `contract_macro_margin` and `critical_contract_regression_policy`.
- Keep `task_accuracy_mean` margin and seed wins as required criteria.
- Add `contract_macro_mean` delta against the selected best baseline.
- Add a no-critical-contract-regression check using aggregate contract breakdown and known priority contracts from validation evidence when available.
- Persist the expanded criteria and blocked reasons in both `summary.json.development_gate` and `runner_report.json.development_gate`.
- Update smoke gate and compact status summaries to surface contract-gate failures without dumping full JSON.

Acceptance criteria:

- A run with higher task accuracy but lower contract macro fails the development gate.
- A run with equal task accuracy and a critical priority-contract regression fails the development gate.
- Existing old summaries remain readable with schema-version-aware fallback.
- The gate output states whether failure came from task margin, seed wins, contract macro, or critical contract regression.

#### E4.S5: Contract Effect Audit

Status: Complete locally on 2026-06-30.

As the optimizer controller, I need to know whether evidence-guided edits changed the intended validation contracts, not only whether proposal metadata was filled.

Implementation tasks:

- Add a read-only contract effect audit over `selection_*_gate.json` and proposal logs.
- For each evidence-guided accepted or rejected candidate, compare targeted contracts, protected contracts, and contract deltas.
- Treat accepted evidence-guided candidates as the hard gate: an accepted step is effective only when at least one targeted priority contract improves and no protected or anti-regression contract regresses.
- Keep rejected candidate protected regressions as non-blocking search-quality diagnostics, because rejected edits do not enter the deployed skill.
- Integrate the audit into smoke gate output as a separate check from `proposal_targeting_audit`.
- Keep metadata audit and effect audit separate in reports.

Acceptance criteria:

- An accepted proposal with valid metadata but unchanged targeted contracts fails effect audit.
- An accepted proposal that improves the target but regresses a protected contract fails effect audit.
- A rejected proposal that regresses a protected contract is counted in rejected diagnostics without failing the effect audit by itself.
- Failure-delta reports can distinguish metadata compliance failure from contract-effect failure.

#### E4.S6: Scale-Up Validation Strength

Status: Complete locally on 2026-06-30.

As the experiment operator, I need cheap smoke runs and scale-up runs to have different evidence strength without mixing their conclusions.

Implementation tasks:

- Document `validation_confirmation_rounds=0` as mechanism-smoke-only.
- Add or update scale-up manifests to use `validation_confirmation_rounds >= 1`.
- Preserve order-balanced paired confirmation when confirmation rounds are enabled.
- Label run artifacts with `mechanism_smoke`, `full_selection_development`, or `same_run_baseline_matrix`.

Acceptance criteria:

- Scale-up manifests cannot silently use `validation_confirmation_rounds=0`.
- Reports clearly state that targeted smoke cannot prove full benchmark effectiveness.
- Accepted steps from scale-up runs include confirmation evidence, not only a single initial improvement.

### Acceptance Criteria

- We can test optimizer changes without rerunning no_skill/human_skill/one_shot.
- A smoke run has a clear pass/fail interpretation.
- A full multi-seed run is only triggered after smoke evidence shows at least one accepted or strongly targeted candidate and passes contract-aware gate checks.
- A benchmark-level claim is not made from targeted smoke.

### Exit Gate

The next experiment is bounded by a decision it can actually change. The first smoke shows the current `task_limit=1` configuration cannot change the development decision. The first revised targeted smoke shows accepted edits are possible, but scale-up remained blocked by seed wins and metadata compliance. The second revised targeted smoke shows slow update can be disabled for smoke at materially lower optimizer-token cost while still passing the development gate. The rejection-triggering smoke proves that proposals generated after contract evidence use `evidence_source=contract_rejection_evidence`, but scale-up remains blocked because evidence-guided edits did not pass the development gate. The failure-delta report narrowed policy work to anti-regression and repeated-target handling. Smoke validation shows that metadata-level anti-regression/cooldown constraints reduce some regressions but do not create improvements; they also increase optimizer token cost. Single-contract targeting plus local duplicate penalties did not improve the development gate. Outcome-aware v4 improved aggregate score but still stopped on seed wins and protected-contract regression. The next guarded smoke remains mechanism-only until the hardened gates pass.

## Epic 5: Paper-Grade Evidence Expansion

### Goal

Only after the development loop produces accepted improvements, expand evidence toward paper-level credibility.

### Why

The paper-level claim requires more than one benchmark family and one harness. But expanding before the local optimization loop works would hide root causes under variance and cost.

### Stories

- E5.1: Define minimum development pass criteria for locked test.
- E5.2: Add a post-pass locked-test protocol checklist.
- E5.3: Add a second benchmark or task family only after `coding-hidden-v2` development gate passes.
- E5.4: Add optional cross-agent transfer evaluation after same-agent improvement is proven.
- E5.5: Produce a final comparison report with no_skill, one_shot, human_skill, executive, and relevant ablations.
- E5.6: Rerun same-run full baseline matrix for final cost and success-rate conclusions; cached baselines are smoke-only score comparators.

### Acceptance Criteria

- Locked test is run only from a documented pass state.
- Cross-agent evidence is labeled as transfer evidence, not same-target optimization evidence.
- Final report separates mechanism validation, benchmark result, and transfer result.
- Final cost comparison uses same-run baselines, not cached baseline artifacts.
- Full-selection executive-only passes before complete same-run baseline matrix.

### Exit Gate

The project can make a narrow but defensible claim about SkillOpt-style skill optimization.

## Dependency Order

1. Epic 1 first: otherwise we cannot trust experiment decisions.
2. Epic 2 second: otherwise rejected feedback remains too vague.
3. Epic 3 third: otherwise the optimizer cannot use the new evidence.
4. Epic 4 fourth: otherwise validation is too costly.
5. Epic 5 last: only after local development evidence passes.

## Superseded Final Planned Story

The final pre-completion recommendation was to run guarded v5 targeted smoke.
That run and the later Stage 4–7 campaign are complete. Do not execute this
historical recommendation; all new work follows
`skillopt-paper-faithful-roadmap.md`.

Historical reason: v4 showed that outcome-aware proposal metadata and local
ranking were insufficient because the run improved aggregate score while
failing seed wins and regressing a protected or anti-regression contract.
