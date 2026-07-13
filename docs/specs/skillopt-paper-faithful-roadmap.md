---
id: SPEC-skillopt-paper-faithful-roadmap
status: active
date: 2026-07-13
normative_source:
  - ../papers/paper-notes.md
reference_implementation:
  - https://github.com/microsoft/SkillOpt
supersedes_as_active_plan:
  - skillopt-paper-gap-plan.md
  - skillopt-gap-epics.md
---

# SkillOpt Paper-Faithful Roadmap

## 1. Objective

Build and evaluate an implementation that is faithful to the SkillOpt paper.
The paper is the specification; the Microsoft implementation is a source of
reusable code and a reference for ambiguities, not the acceptance standard.

The work has four claim levels:

1. `paper-mechanism-conformant`: the implementation follows Algorithm 1 and
   the Appendix contracts.
2. `fresh-local-efficacy`: the conformant implementation beats eligible
   baselines on a new untouched local held-out set.
3. `partial-paper-reproduction`: at least one paper benchmark is run with the
   published split, scorer, configuration, target, and applicable baselines.
4. `paper-scope-replication`: the benchmark, model, harness, baseline,
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
  confirmation_rounds: 0
  contract_guard: false

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
selection_feedback_to_optimizer: false
benchmark_specific_prompt_rules: false
```

Benchmark-specific overrides may change only fields explicitly justified by
the paper and must be recorded in the run manifest.

## 9. Execution Roadmap

| Phase | Deliverables | Exit gate |
| --- | --- | --- |
| 0. Freeze and provenance | Archive extension v1 claim; correct Stage 7 status; create repository baseline commit/tag; pin paper and official references | Existing 258 tests pass and every artifact is tied to a code/config/data lineage |
| 1. Spec and profile | Machine-readable paper profile, forbidden-override validation, Algorithm 1 event schema, claim taxonomy | A run can be classified without reading implementation code |
| 2. Data firewall | Scalar-only selection interface, isolated test controller, consumed-split registry, claim eligibility checker | Selection sentinels never reach optimizer payloads; old Stage 7 cannot produce a paper claim |
| 3. Patch fast core | Injected backend, analyst prompts, true refinement, hierarchical merge, ranking, four patch ops, apply reports, strict gate/cache | Scripted event trace matches Algorithm 1 fast loop |
| 4. Epoch state | Epoch-local buffer, current/best state, score cache, scheduler, deterministic data plan, resumable checkpoints | Buffer resets and crash/resume matches an uninterrupted fake run |
| 5. Slow/meta | Adjacent-epoch comparison, four-way longitudinal classification, protected slow field, gated slow candidate, future-only meta | Algorithm 1 epoch-boundary trace matches the paper |
| 6. Mechanism completeness | `A>1` accumulation, stable analyst concurrency, autonomous LR, `rewrite_from_suggestions`, complete resume | Paper-supported modes have independent mechanism tests; default remains patch/cosine |
| 7. Zero-cost conformance | Golden trace, static import/prompt scans, data-access audit, upstream-deviation checks, provenance linter | All checks pass before any paid run |
| 8. Cheap development smoke | One seed, open development data, reduced sizes but complete call graph and strict gate | Merge/rank/slow/meta are actually exercised; artifacts complete; budget respected |
| 9. Fresh pilot | New or official benchmark, full default profile, one non-headline pilot seed, complete cost capture | Accepted update exists; selection is not saturated; no leakage/drift; cost is affordable |
| 10. Confirmatory development | Three preregistered seeds and same-run eligible baselines | Mean beats strongest simple baseline by the preregistered margin and at least 2/3 seeds improve |
| 11. Ablations | One-factor LR, buffer, slow/meta, batch/minibatch, accumulation, patch/rewrite comparisons | Each run differs from the frozen default in exactly one registered factor |
| 12. Fresh held-out matrix | Freeze all SkillOpt/baseline/ablation artifacts, then evaluate them atomically on one untouched test split | Eligible for `paper_faithful_heldout`; test cannot be reused after code/prompt changes |
| 13. Paper breadth | Paper benchmarks, multiple targets/harnesses, transfers, remaining baselines | Claims increase only to the breadth actually measured |

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

Set each paid stage's call/token/wall-time cap from dry-run or pilot estimates
with a 1.25–1.5 safety factor. A budget breach invalidates or stops the stage;
do not silently shrink the protocol after execution starts.

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

- claim class and `protocol_id`
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

A profile other than `paper-faithful-v1` cannot create a
`paper_faithful_*` claim. A split already consumed by another protocol cannot
create a new paper-faithful held-out claim.

## 13. Immediate Work Queue

Execute in this order:

1. Correct current-state and README references; freeze Stage 5/7 as extension
   v1; never modify their raw artifacts.
2. Establish the repository's first reproducible baseline commit and tag. The
   current project files are untracked, so implementation must not proceed
   without a recoverable starting point.
3. Add the paper profile, claim taxonomy, lineage schema, and consumed-split
   registry.
4. Write the selection/test firewall and extension-isolation tests first.
5. Scaffold `textskill_optimizer.paper` with injected optimizer backend and
   independent paper edit types.
6. Implement patch application, strict gate/cache, and scheduler.
7. Implement reflection, true refinement, hierarchical merge, and ranking.
8. Implement epoch buffer, slow/meta, and paper CLI/artifact gates.
9. Run only zero-cost conformance tests, then one cheap development smoke.
10. Design and preregister the official/fresh benchmark campaign before the
    first full-profile run.

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

## 15. Milestones and Definition of Done

- **M0 — Paper patch core:** default patch-mode Algorithm 1 trace passes.
- **M1 — Paper mechanisms:** accumulation, concurrency, autonomous LR, rewrite,
  and resume have independent mechanism evidence.
- **M2 — Development evidence:** three-seed paper-faithful development matrix
  passes its preregistered gate.
- **M3 — Fresh held-out evidence:** a new atomic test matrix supports a
  `paper_faithful_heldout` claim.
- **M4 — Paper-scale replication:** measured breadth supports the claimed
  benchmark/model/harness scope.

The default patch core is roughly 18–26 engineering person-days. The complete
mechanism set, including rewrite, autonomous LR, and fully equivalent resume,
is roughly 29–43 person-days. External experiment runtime and model cost are
additional and are controlled by the evidence ladder above.
