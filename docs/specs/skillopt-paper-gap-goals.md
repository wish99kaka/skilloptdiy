---
id: SPEC-skillopt-paper-gap-goals
companions:
  - skillopt-experiment-runbook.md
  - skillopt-current-state.md
  - skillopt-gap-epics.md
  - ../skillopt-executive-protocol.md
sources:
  - ../papers/paper-notes.md
---

# SkillOpt Paper Gap Goals

> Historical goals for the completed contract-aware Stage 4-7 campaign. The
> active implementation goal is defined in
> `docs/specs/skillopt-paper-faithful-roadmap.md`. The `coding-hidden-v2` locked
> test was consumed once on 2026-07-13 and must not be rerun.

## Why

The repository is a SkillOpt-style engineering prototype and experiment gate system, not yet a paper-grade optimization result. The next work must close the result gap, not merely add more mechanism: stable accepted edits, held-out improvement over strong baselines, contract-safe behavior, operationally complete runs, and cost-accounted scale-up.

## Role Definitions

- Target agent: Coco local default. Run names or gateway labels such as `deepseek` do not make DeepSeek the target agent.
- Optimizer: external OpenAI-compatible editor model injected through environment variables.
- Benchmark of record: `coding-hidden-v2`.
- Smoke runs: mechanism evidence only. They can validate proposal/rejection/control-loop behavior, but cannot prove full benchmark effectiveness.
- Paper-grade result: full-selection development evidence, same-run baseline comparison, cost-accounted reporting, and locked-test eligibility through preflight.

## Capabilities

- id: CAP-1
  intent: The project can demonstrate that `executive` beats the best eligible baseline on `coding-hidden-v2` before any locked test.
  success: A full-selection development run passes the hardened development gate: task-accuracy margin, required seed wins, contract-macro non-regression, and configured critical-contract regression checks.

- id: CAP-2
  intent: The optimizer can convert contract evidence into behavior-changing edits, not just well-formed proposal metadata.
  success: Contract effect audit passes: evidence-guided candidates improve at least one targeted priority contract and do not regress protected or anti-regression contracts. Proposal targeting metadata alone is not a pass.

- id: CAP-3
  intent: Accepted skill edits remain compact, procedural, and attributable to validation-gated evidence.
  success: Each accepted edit has an auditable gate record, a proposal record, targeted-contract evidence when available, and no hidden-answer or task-specific rule leakage.

- id: CAP-4
  intent: Experiment stages produce only the decision they are strong enough to support.
  success: Targeted smoke is labeled mechanism evidence only; full-selection executive-only uses `validation_confirmation_rounds >= 1`; same-run baselines are run only after full-selection executive evidence passes.

- id: CAP-5
  intent: Final comparison and cost claims use same-run evidence instead of cached smoke baselines.
  success: Any benchmark-level claim cites same-run baseline rows, actual optimizer usage, and an explicit target-agent accounting scope; cached baselines are labeled score comparators only and excluded from final cost conclusions.

- id: CAP-6
  intent: Locked test remains one-attempt final evaluation, not a development tool.
  success: `work/skillopt_locked_preflight.py` reports `allowed` only after full-selection evidence, smoke gate, contract effect audit, no persistent anomalies, actual optimizer usage, and no prior locked receipt; Stage 7 then pins the selected skill and execution inputs, writes an attempt marker before launch, consumes the locked test once, and joins receipt plus held-out scores into the final report.

- id: CAP-7
  intent: Paper-scale expansion happens only after the local same-target loop works.
  success: Cross-agent, cross-harness, cross-benchmark, and larger benchmark matrix work is labeled transfer or expansion evidence and starts only after the Coco `coding-hidden-v2` development gate passes.

- id: CAP-8
  intent: Guarded smoke runs complete far enough to produce decision artifacts before their algorithmic result is interpreted.
  success: v5 guarded smoke or its successor writes `summary.json`, `decision.json`, `smoke_gate_report.json`, and `contract_effect_audit.json`; endpoint/DNS failures are classified as operational failures, not candidate-guard evidence.

## Constraints

- Target agent stays Coco local default; do not override or modify the Coco model.
- The optimizer is an external OpenAI-compatible editor model injected through environment variables, not written into manifests.
- `coding-hidden-v2` is the current benchmark of record.
- The locked test is blocked until locked preflight reports `allowed`.
- Targeted smoke cannot justify scale-up unless the smoke gate and contract effect audit both pass.
- Cached baselines can compare smoke scores but cannot support final cost claims.
- Proposal targeting metadata is necessary but not sufficient; behavior-level contract effect decides whether the proposal mechanism is working.
- Current primary token accounting covers project executor usage and optimizer API usage; Coco/CCR/Kilo usage is out of scope until explicitly added.
- Implementation reviews should cite stable artifacts, fields, and function names where possible; source line numbers are acceptable for local inspection but not stable contract references.

## Non-goals

- Do not chase paper-scale benchmark breadth before the single-target development gate passes.
- Do not treat v5 guarded smoke connection failure as algorithm evidence.
- Do not add more metadata-only proposal policy unless failure artifacts show metadata compliance is the blocker.
- Do not use cross-agent transfer results to replace the same-target Coco development comparison.
- Do not rerun locked test or inspect encrypted test content during development.
- Do not make final cost-per-point claims from optimizer-only accounting or cached baselines.

## Success signal

The local same-target contract-aware loop met this signal: full-selection and
contract-effect gates passed, the same-run matrix confirmed improvement, locked
preflight was allowed, and Stage 7 completed once with `20/20` task accuracy.
This closes the historical extension goal only. Paper-faithful claims require a
separate conformant protocol and a new untouched split or official benchmark.

## Current Gaps

- Paper-faithful held-out gap: the consumed Stage 7 split is historical contract-aware evidence and cannot be reused; a new untouched split or official benchmark is required.
- Scale gap: current primary evidence is one benchmark and one same-target Coco comparison; the paper reports six benchmarks, seven target models, and three harnesses.
- Baseline gap: the same-run matrix covers no-skill, one-shot, human-skill, and executive, but not Trace2Skill, TextGrad, GEPA, or EvoSkill.
- Transfer gap: cross-agent work can support transfer and robustness claims later, but it cannot replace the same-target Coco development gate.
- Cost gap: same-run optimizer API usage is tracked, but target-agent token usage remains out of scope, so paper-equivalent total cost-per-point cannot be claimed.

Closed local-development gaps:

- Stage 5 executive mean is `1.0` versus same-run human-skill mean `0.8667`, with `+0.1333` delta and `3/3` seed wins.
- Contract effect audit passes with zero accepted protected regressions; rejected regressions remain diagnostic evidence that the guard worked.
- Locked preflight is `allowed`, and the prepared Stage 7 manifest selects the compact `seed-c` skill before any held-out access.
- Stage 7 consumed the locked test exactly once and completed with task accuracy `20/20 = 1.0` and family macro `1.0`; all task executions returned `0` without timeout.
- Locked task records contain no contract tags, so `unknown_contract=1.0` is not evidence of contract-level generalization.

## Assumptions

- This spec treats `docs/papers/paper-notes.md` as the local paper source and `docs/specs/skillopt-current-state.md` as the latest repository-state source.
- This spec preserves the current project decision that Coco/CCR/Kilo token usage is out of scope for primary accounting until a later accounting change explicitly adds it.
- The external review supplied in chat is treated as corroborating analysis, not as a separate source artifact.
