---
id: SPEC-skillopt-paper-faithful-roadmap
status: active
date: 2026-07-13
normative_source:
  - ../papers/skillopt-2605.23904.pdf@arxiv:2605.23904v2
reference_implementation:
  - https://github.com/microsoft/SkillOpt/tree/e4ea6a6771e797ef820cdd8bfea64c57e0481065
source_lock:
  - ../papers/source-lock.json
supersedes_as_active_plan:
  - skillopt-paper-gap-plan.md
  - skillopt-gap-epics.md
---

# SkillOpt Paper-Faithful Roadmap

## 1. Objective

Build and evaluate an implementation that is faithful to the SkillOpt paper.
The paper is the specification; the Microsoft implementation is a source of
reusable code and a reference for ambiguities, not the acceptance standard.

The work has four program-level evidence levels. These are conclusions over
result bundles, not the per-artifact `claim_class` taxonomy in Section 12:

1. `paper_mechanism_conformant`: the implementation follows Algorithm 1 and
   the Appendix contracts.
2. `fresh_local_efficacy`: the conformant implementation beats eligible
   baselines on a new untouched local held-out set.
3. `partial_paper_reproduction`: at least one paper benchmark is run with the
   published split, scorer, configuration, target, and applicable baselines.
4. `paper_scope_replication`: the benchmark, model, harness, baseline,
   ablation, and transfer breadth approaches the paper's full evidence scope.

Do not use a higher claim label until every lower-level provenance and evidence
gate has passed.

## 2. Source Precedence

Decisions use this order:

1. Paper Algorithm 1, Section 3, and Appendix C.
2. Paper experiment protocol and benchmark-specific overrides.
3. The official implementation, only where the paper is ambiguous.
4. Local engineering choices that do not alter algorithm semantics.

When official code conflicts with the paper, record the divergence and follow
the paper. Two known examples are already confirmed:

- Official `main` defaults slow update to force-accept, while the paper requires
  the slow candidate to pass the selection gate.
- Official `max_analyst_rounds=3` is configured but does not execute a real
  teacher-refinement loop.

Pin the official reference commit or release and record file checksums before
porting code. Preserve its MIT attribution.

## 3. Current Evidence Boundary

### 3.1 Frozen contract-aware result

The following run is historical `contract-aware-extension-v1` evidence:

```text
runs/coding-hidden-v2-deepseek-same-run-baseline-matrix-stage5-v1-scorestop
```

Its development result remains valid for that protocol:

- executive mean: `1.0`
- best same-run baseline mean: `0.8667`
- seed wins: `3/3`
- optimizer API usage: `422280` executive tokens and `431247` total tokens

Stage 7 consumed the locked test once on 2026-07-13 and completed successfully:

- task accuracy: `20/20 = 1.0`
- family macro: `1.0`
- receipt return code: `0`

This result must not be relabeled as paper-faithful because its optimization
used selection contract feedback, benchmark-specific mechanism rules, paired
confirmation, contract guards, two epochs, unit batch/minibatch sizes, constant
LR 2, disabled slow update, and early stopping. The locked run evaluated only
the SkillOpt artifact, not held-out baselines. Its test tasks also had no
contract tags, so `unknown_contract=1.0` is not contract-generalization evidence.

### 3.2 Permanent consequences

- Never rerun the `coding-hidden-v2` locked test.
- Preserve its attempt, receipt, result, usage ledger, and final report.
- Do not use its skill, meta skill, rejected buffer, selection cache, or test
  result as initialization or selection evidence for the paper engine.
- Future paper-faithful held-out claims require a new untouched split or an
  official paper benchmark test protocol.

## 4. Protocol Separation

Maintain three explicit protocols:

| Protocol | Purpose | Status |
| --- | --- | --- |
| `legacy` | Original minimal optimizer and historical replay | Compatibility only |
| `contract-aware` | Current executive optimizer with contract feedback and guards | Supported extension |
| `paper-faithful-v1` | Paper Algorithm 1 and Appendix behavior | New mainline |

The paper package must not import `contract_evidence`,
`contract_rejection_evidence`, or the old executive optimizer. Existing
extension behavior stays testable, but cannot silently enter a paper run.

Every new run must record a `protocol_id`. Existing artifacts are immutable and
must not be rewritten into a new schema or claim class.

## 5. Target Architecture

Create a separate injected-backend engine:

```text
textskill_optimizer/paper/
  config.py
  types.py
  backend.py
  prompts/
  reflection.py
  aggregate.py
  ranking.py
  update.py
  scheduler.py
  buffer.py
  epoch_memory.py
  engine.py
```

Reuse only infrastructure without optimizer semantics:

- task runner and scorer interfaces
- usage ledger and timing events
- generic manifest and artifact I/O
- environment isolation and failure reporting
- locked evaluation primitives for a future new split

Add `--protocol paper` without changing the existing meaning of
`--protocol executive`. Consider changing the default only after the paper
engine reaches its conformance milestone.

## 6. Official Code Reuse Boundary

### Reuse selectively

- Appendix prompt contracts.
- Patch schema and per-edit apply-report structure.
- Failure/success reflection grouping.
- Hierarchical failure merge, success merge, and failure-priority final merge.
- Optimizer-driven ranking and top-L clipping.
- Scheduler, rewrite, slow-update, and meta-skill algorithm skeletons.

All model calls must be adapted to an injected `OptimizerBackend`; do not bring
over global model configuration.

### Do not import wholesale

- Official `trainer.py`, environment packages, or dataloaders.
- WebUI, Sleep, provider backends, or plugin shells.
- Force-accept slow update.
- Soft/mixed gates, skill-aware appendix, or full-rewrite-minibatch mode.
- Silent merge concatenation or ranking truncation fallbacks.

In paper mode, failed semantic merge/ranking must be retried according to the
recorded provider policy and then reject/skip the step without changing the
skill. It must not masquerade as a successful optimizer-model decision.

## 7. Normative Conformance Invariants

### N1. Data boundary

- Train supplies trajectory evidence.
- Selection supplies only a benchmark-native scalar score and accept/reject
  decision to the controller.
- Selection tasks, contract tags, per-item results, and diagnostics never enter
  reflection, merge, ranking, buffer, slow, or meta payloads.
- Test runs only after candidates and evaluation policy are frozen.

### N2. State

Persist current and best skills/scores separately, the selection score cache,
epoch step buffer, meta skill, scheduler state, data plan, and epoch snapshots.
Every epoch starts with an empty step buffer.

### N3. Rollout and reflection

- Each optimization step collects `A` rollout batches.
- Failures and successes are separated and divided into reflection minibatches.
- They use distinct analyst prompt contracts.
- Teacher refinement is real semantic refinement, capped at three rounds, and
  distinct from transport retries or schema repair.

### N4. Merge and ranking

The required call order is:

```text
merge_failure
merge_success
merge_final_failure_prioritized
rank_and_clip_to_L
```

Merge and ranking are optimizer-model decisions, not local string
deduplication. Source type and support count remain auditable. Meta guidance is
available to reflection, merge, and ranking.

### N5. Text updates

Paper patch mode uses `append`, `insert_after`, `replace`, and `delete`. Apply
edits sequentially, protect the slow-update field, and emit one apply result per
edit. Every candidate must be replayable from its current skill, ranked patch,
and apply report.

### N6. Selection gate

- Accept only when `candidate_score > current_score`; reject ties.
- Use a skill-hash score cache.
- Paired confirmation, contract guards, soft/mixed gates, and force-accept are
  extension behavior and forbidden in the default paper profile.
- Slow candidates use the same strict gate.

### N7. Epoch buffer

Record train-observed failure patterns for every step. For rejected candidates,
also record the attempted edits and score drop. Later optimizer calls in the
same epoch may see this context; the next epoch must not. Resume must reconstruct
the buffer without adding selection diagnostics.

### N8. Slow and meta

- Skip real slow/meta updates in epoch 1.
- From epoch 2, compare adjacent epoch-end skills on the same sampled train
  tasks and classify improvement, regression, persistent failure, and stable
  success.
- Slow guidance lives in a protected target-side field and must pass selection.
- Meta skill is optimizer-only and affects future reflection, merge, and
  ranking; it is never sent to the target runner or exported as part of the
  deployed skill.

## 8. Default Paper Profile

```yaml
profile: paper-faithful-v1
protocol_id: paper-faithful-v1
epochs: 4
split_seed: 42
default_split_ratio: 2:1:7

rollout_batch_size: 40
accumulation: 1
reflection_minibatch_size: 8
merge_batch_size: 8
analyst_workers: 16
max_analyst_rounds: 3

update_mode: patch
learning_rate: 4
learning_rate_floor: 2
learning_rate_schedule: cosine

selection_gate:
  enabled: true
  metric: benchmark_native
  comparator: strict_greater

rejected_buffer:
  enabled: true
  scope: epoch

slow_update:
  enabled: true
  start_epoch: 2
  sample_size: 20
  selection_gated: true

meta_skill:
  enabled: true
  start_epoch: 2
  initial: empty
  target_visible: false

early_stop: false
```

Paired/confirmation gating, contract guards, soft or mixed gates, force-accept,
selection feedback, and benchmark-specific prompt controls are not disabled
paper fields. They are absent from the schema, and their presence makes a
profile non-conformant.

`paper-faithful-v1` is frozen to the values above. M1's allowed override
registry is empty: a deviation is rejected as `unregistered_profile_override`.
A future benchmark-specific override becomes valid only after its exact field,
value rule, and paper citation are added to the versioned contract and tests;
free-form run-manifest justification is not authority by itself.

## 9. Execution Roadmap

The pre-M0 findings were measured against repository commit
`91c00b9c582e48b077c9282f4ccc80db26341653` on 2026-07-13. The table below
records the status of the repository revision that contains it. The WP sections
in Section 13 are the only executable plan; this table is only a status index.

| Phase | Status | Work package |
| --- | --- | --- |
| 0. Freeze and provenance | Completed | WP0 |
| 1. Spec and profile | Completed | WP1 |
| 2. Data firewall | Completed | M2 / WP1B |
| 3. Patch fast core | Completed | M3 / WP2 |
| 4. Epoch state | Completed | M4 / WP3 |
| 5. Slow/meta | Completed | M4 / WP3 |
| 6. Mechanism completeness | Completed | M5 / WP3 |
| 7. Zero-cost conformance | Completed | WP4 |
| 8–11. Development evidence | Next | WP5 |
| 12–13. Held-out evidence and breadth | Blocked by Phase 11 | WP6 |

## 10. Conformance Test Program

Create separate suites:

```text
tests/conformance/unit/
tests/conformance/integration/
tests/conformance/mechanism/
tests/provenance/
tests/experiment/
tests/extensions/contract_aware/
```

Required gates include:

- paper defaults and forbidden extension controls
- no imports from contract-aware modules
- selection sentinel and test-access firewalls
- failure/success partitioning and real three-round refinement
- hierarchical merge call order and mandatory optimizer ranking
- top-L budget and deterministic patch replay
- apply report completeness and protected-region preservation
- strict tie rejection and selection cache behavior
- epoch buffer reset and resume reconstruction
- epoch-1 slow/meta skip and gated slow update
- meta visibility only in future optimizer stages
- same-run matrix protocol equality
- one-factor ablation validation
- consumed-split and immutable-receipt policy

Existing tests for selection contract feedback, paired confirmation, persistent
cross-epoch rejected history, contract guards, targeting/cooldown policies, and
hardcoded mechanism anchors move conceptually under the extension suite. Their
behavior must not become paper defaults.

## 11. Evidence Program

### 11.1 Cost ladder

Spend in this order:

1. Scripted target and optimizer: zero external calls.
2. One-seed mechanism smoke.
3. One full-profile pilot seed.
4. Three confirmatory development seeds.
5. Core causal ablations.
6. One atomic held-out matrix.
7. Benchmark/model/harness breadth.

Set each paid stage's call-count and wall-time cap from dry-run or pilot
estimates with a 1.25–1.5 safety factor. Those two limits are fail-closed.
Target-agent and experiment-optimizer token projections and actual usage remain
mandatory audit fields, but are excluded from experiment cost and go/no-go
decisions. The resource optimization target is the Codex development task's
own token usage, which is not an experiment receipt metric.

### 11.2 Baselines

Start with same-run `no_skill`, human skill, and one-shot LLM skill. Add
Trace2Skill and EvoSkill when the harness can be matched. Add TextGrad and GEPA
only where their execution mode is applicable. Record unavailable comparisons
as `-`; do not mix unmatched protocols into a ranking.

### 11.3 Ablations

After the frozen default method passes development, run single-factor cases for:

- rejected buffer off
- meta off with slow retained
- meta and slow both off
- bounded/cosine versus constant, linear, autonomous, and unbounded LR
- train size, rollout batch size, reflection minibatch size, and edit budget
- slow sample size
- patch versus `rewrite_from_suggestions`

Use three seeds for causal claims. A one-seed ablation is exploratory only.

### 11.4 Fresh final evidence

The recommended paper anchor is SearchQA because its official split is
available and its direct-chat runner/scorer is the lowest integration cost.
Use a reduced development slice only for smoke, then use the paper split and
configuration for a reproduction claim.

For a local coding claim, create `coding-hidden-v3` before optimization. A
recommended starting design is 100 structurally distinct tasks across 10
families with a deterministic `20/10/70` split. Test tasks must not be mere
renames or numeric variants, must retain family/contract metadata, and must be
sealed and committed by hash before paper-faithful development begins.

The final controller evaluates all frozen default, baseline, and published
ablation artifacts in one atomic test matrix. It must not evaluate SkillOpt
first and append baselines later.

## 12. Claim Provenance

Every result bundle records:

- artifact claim class, optional program evidence level, and `protocol_id`
- paper version and official reference commit
- local code commit and upstream-deviation manifest
- profile, prompt, and skill hashes
- split manifest, scorer, runner, harness, and environment hashes
- student and optimizer model identifiers and reasoning settings
- seeds, retries, schedules, provider/runtime versions
- optimizer and target calls/tokens, wall time, and cost scope
- test exposure history, archive commitment, attempt, and receipt

Allowed claim classes are:

```text
mechanism_test
development_result
contract_aware_extension
paper_faithful_development
paper_faithful_heldout
paper_scale_reproduction
```

The two taxonomy axes map as follows:

| Artifact `claim_class` | Allowed non-null `evidence_level` |
| --- | --- |
| `mechanism_test` | `paper_mechanism_conformant` |
| `development_result`, `contract_aware_extension`, `paper_faithful_development` | none |
| `paper_faithful_heldout` | `fresh_local_efficacy` |
| `paper_scale_reproduction` | `partial_paper_reproduction`, `paper_scope_replication` |

An evidence level remains null until its measured gate passes; lineage naming
alone never upgrades evidence.

A profile other than `paper-faithful-v1` cannot create a
`paper_faithful_*` claim. A split already consumed by another protocol cannot
create a new paper-faithful held-out claim.

## 13. Next Execution Plan

### 13.1 Pre-M0 review baseline

The project-wide review on 2026-07-13 established these pre-M0 facts:

- The reproducible repository baseline is complete at commit
  `91c00b9c582e48b077c9282f4ccc80db26341653`, tagged
  `contract-aware-extension-v1`, with a clean worktree and `origin/main` at the
  same commit.
- `python3 -m pytest tests -q` passed `258` tests plus `2` subtests, while the
  default command collected benchmark fixtures and failed. M0 resolved this by
  setting pytest `testpaths` to `tests`.
- The README extraction and coding offline examples passed, while the existing
  `executive` protocol could exit successfully without optimizing when given a
  built-in full-skill editor. M0 resolved this with an explicit editor
  capability check and non-atomic response failure.
- `paper-faithful-v1` was specified but had no package, CLI protocol, or
  conformance suite. M1 resolved the contract/package gap; the engine and CLI
  protocol remain intentionally unavailable until later gates pass.
- Stage 7 is consumed historical `contract-aware-extension-v1` evidence. Its
  `20/20` result cannot be relabeled or reused for a paper-faithful held-out
  claim.
- No paid experiment is authorized until Phase 7 zero-cost conformance passes.

### 13.2 Work packages and gates

Execute one work package at a time. Do not begin a package until its declared
dependencies pass.

#### WP0 — Close engineering and provenance baseline

Status: completed on 2026-07-13. Dependencies: none.

1. Pin the normative paper as arXiv `2605.23904v2` and record the tracked PDF
   SHA256 `87f7f0f323b1671e9202b3ebb1596e909e507c71ecd1b360b0075a5ee1727fe3`.
2. Pin the official reference to Microsoft SkillOpt `v0.2.0`, commit
   `e4ea6a6771e797ef820cdd8bfea64c57e0481065`. Record checksums for every
   reused source/prompt and maintain an upstream-deviation manifest. A newer
   upstream version requires an explicit re-pin; `main` is never a valid
   reference by itself.
3. Add pytest `testpaths = ["tests"]`, reproducible development dependencies,
   and CI for the supported Python versions. Establish measured coverage before
   choosing a coverage threshold; do not invent an arbitrary number.
4. Add editor capability validation so `--protocol executive` fails before
   execution when its editor cannot produce atomic edits. Do not expand the old
   optimizer to solve paper requirements.
5. Correct historical spec/runbook wording and mark the Stage 4–7 operator
   procedures completed. Never modify the raw Stage 5/7 artifacts.

Exit gate:

- default and scoped test commands run the same intended suite and pass
- CI reproduces the local gate from a clean environment
- official, paper, local, prompt, and reused-source identities are immutable
- incompatible executive/editor combinations return a clear non-zero error
- no active document instructs an operator to consume `coding-hidden-v2`

#### WP1 — Define the paper contract before implementation

Status: completed on 2026-07-13. Dependencies: WP0.

1. Add the machine-readable `paper-faithful-v1` profile and reject all forbidden
   extension overrides.
2. Add claim taxonomy, lineage schema, consumed-split registry, and Algorithm 1
   event schema.
3. Scaffold `textskill_optimizer.paper` with an injected `OptimizerBackend` and
   independent paper edit/state types.
4. Write import-firewall, selection-sentinel, test-access, strict-tie, and
   consumed-split tests before implementing the engine.

Exit gate: a fake run can be classified and rejected for protocol or provenance
violations without invoking a model or reading engine internals.

Completion evidence:

- `textskill_optimizer.paper` exposes the validated profile, run assessment,
  claim/lineage contracts, consumed registry, Algorithm 1 events, independent
  edit/state types, and injected backend seam.
- `tests/conformance` and `tests/provenance` cover forbidden extension fields,
  scalar-only selection, strict ties, test access, consumed splits, receipt
  immutability, and static imports.
- No paper optimizer engine or paid/model-backed experiment was introduced.

#### WP1B — Wire the runtime data firewall

Status: completed on 2026-07-15. Dependencies: WP1.

1. Freeze a content-addressed controller registry: the actual argv prefix must
   exactly match an ordered, hash-verified executable/runner chain, and that
   chain plus its Ed25519 response key may own exactly one train, selection, or
   test role and one hash-verified split manifest. Split IDs and manifest
   hashes are globally single-owner; consumed splits are rejected. Reverify
   signed train responses and manifest identity at the optimizer seam.
2. Run the selection data owner in its registered process; accept exactly one
   finite `score` field and retain only normalized `SelectionScore` values.
3. Put final-test evaluation in a disconnected module and registered process;
   rebuild the exact `FinalEvaluationPlan`, then bind registry, split, runner,
   scorer, and harness hashes to verified bytes immediately before test access.
4. Make `paper.__init__` lazy so cold final imports do not execute optimization
   modules. Add process-owned selection sentinels, role-reuse/split-spoof,
   signed-evidence/final-plan attacks, rich-response rejection, and both static
   transitive and cold-process runtime import audits.

Exit gate: after selection has run, a captured optimizer request contains only
train-origin fields signed by the registered train owner; a runner/key cannot
cross roles, and a split ID/manifest cannot cross owners or be relabeled;
selection/test task objects never enter the
optimizer process; the test process cannot run through the paper controller
without a revalidated artifact-bound frozen plan; cold final imports load no
optimization module.

#### WP2 — Implement the paper fast loop

Status: completed. Dependencies: WP1B.

1. Implement `append`, `insert_after`, `replace`, and `delete` with protected
   slow-field enforcement and one apply report per edit.
2. Implement separate failure/success reflection and real semantic teacher
   refinement capped at three rounds.
3. Implement failure merge, success merge, failure-prioritized final merge, and
   optimizer-model ranking clipped to top `L`; local string ranking is not an
   allowed success fallback. Same-source proposals merge hierarchically in
   frozen profile-sized batches; semantic merge/rank failures follow the
   recorded retry policy and then skip the step unchanged.
4. Implement strict scalar selection gating, current/best separation, skill-hash
   score cache initialized by the selection owner, and replayable step
   artifacts. External cache restoration is forbidden until WP3 supplies an
   authenticated checkpoint path.

Exit gate: a deterministic fake backend produces a golden event trace matching
Algorithm 1, and every candidate can be reconstructed from its input skill,
ranked patch, and apply report.

#### WP3 — Implement epoch state and remaining mechanisms

Status: completed on 2026-07-16. Dependencies: WP2.

1. Implement epoch-local rejected-step buffers and deterministic data plans.
2. Implement checkpoints whose resumed event/artifact sequence matches an
   uninterrupted run.
3. Implement the epoch/step edit-budget scheduler, including persisted scheduler
   state needed for exact resume.
4. Skip real slow/meta updates in epoch 1. From epoch 2, compare adjacent epoch
   skills on the same training samples, generate four-way longitudinal state,
   gate the protected slow candidate, and expose meta guidance only to future
   optimizer stages.
5. Add `A>1` accumulation, stable analyst concurrency, autonomous LR, and
   `rewrite_from_suggestions` after the default patch/cosine path passes.

Exit gate: Phase 4–6 mechanism tests pass independently, including buffer reset,
resume equivalence, slow gating, and meta visibility.

#### WP4 — Zero-cost conformance

Status: completed on 2026-07-16. Dependencies: WP3.

Run golden traces, static import and prompt scans, selection/test data-access
audits, patch replay, upstream-deviation checks, lineage validation, and claim
eligibility tests using scripted target and optimizer backends.

Exit gate: Phase 7 passes as one automated gate. Failure keeps all paid work
blocked.

#### WP5 — Development evidence

Dependencies: WP4.

1. Run one cheap full-call-graph smoke on open development data.
2. Preregister one full-profile pilot, including model identities, split, scorer,
   retries, seed, budgets, stop conditions, and artifact hashes.
3. Use SearchQA as the first paper benchmark anchor. Run a reduced development
   slice only for smoke, then the official split/configuration for a reproduction
   claim.
4. If a separate coding claim is still useful, seal `coding-hidden-v3` before
   optimization with structurally distinct tasks and preserved family/contract
   metadata.
5. After a successful pilot, run three confirmatory seeds with same-run
   `no_skill`, human-skill, and one-shot baselines, followed by registered
   single-factor ablations.

Exit gate: the preregistered development gate passes, selection is not
saturated, call/time limits and audit-only model-token scope are complete, and
no test data was accessed.

#### WP6 — Atomic held-out evidence and breadth

Dependencies: WP5.

Freeze the default method, baselines, and published ablations before one atomic
evaluation on an untouched test split. Expand to additional paper benchmarks,
models, harnesses, baselines, and transfer only after this first held-out gate
passes.

Exit gate: claim class is generated from measured scope. The first eligible
result may be called `paper_faithful_heldout`; broader replication language
requires the corresponding measured breadth.

## 14. Non-Goals and Stop Rules

- Do not rerun or reopen the old `coding-hidden-v2` locked test.
- Do not relabel Stage 5/7 as paper-faithful.
- Do not continue adding paper behavior inside the existing executive class.
- Do not copy the official repository wholesale.
- Do not feed contract/family diagnostics to the paper optimizer; they are
  report-only and may support an external go/no-go decision.
- Do not run expensive full-selection experiments before zero-cost conformance.
- Do not expand toward 52 cells before one official benchmark passes.
- Do not claim paper-equivalent cost until target-agent usage is captured.

## 15. Execution Milestones

These names follow the project M0–M8 sequence; the WP sections remain the
executable plan.

- **M0 — Baseline closed:** WP0 passes and provenance plus engineering gates are
  reproducible.
- **M1 — Paper contract enforced:** WP1 passes; extension controls cannot enter
  the profile, and selection/test interfaces are fail-closed before engine
  wiring.
- **M2 — Data firewall enforced:** WP1B passes; selection exposes only a scalar,
  final test is isolated, and neither source can enter optimizer payloads.
- **M3 — Fast loop conformant:** WP2 passes with the Algorithm 1 fake-backend
  event sequence, real reflection/merge/rank stages, replayable patches, and a
  strict cached gate.
- **M4 — Epoch loop conformant:** WP3 state, scheduler, epoch-local buffer,
  resume, slow, and meta lifecycle tests pass.
- **M5 — Mechanisms complete:** accumulation, autonomous LR, rewrite mode,
  concurrency, and full artifact lineage pass independent tests.
- **M6 — Zero-cost acceptance:** WP4 passes and emits a clean-commit-bound
  receipt; this authorizes M7 development work but is not empirical evidence.
- **M7 — Development evidence:** WP5's smoke, pilot, three-seed baselines, and
  registered ablations pass without protocol drift.
- **M8 — Final evidence:** WP6 evaluates all frozen candidates once on a new
  untouched test and may then emit `paper_faithful_heldout`.

Do not make a calendar commitment from the old executive code size. Estimate
remaining implementation time only after WP1 fixes the reuse boundary and WP2
completes one end-to-end fake-backend vertical slice. External experiment
runtime and model cost remain controlled by the evidence ladder above.
