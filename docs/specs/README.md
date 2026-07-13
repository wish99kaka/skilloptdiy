# Specs

Project facts and decision records that should be read before re-reviewing raw experiment artifacts.

## Active Specs

- `docs/specs/skillopt-paper-faithful-roadmap.md`: canonical roadmap for the new paper-faithful engine, conformance gates, evidence program, and claim provenance.
- `docs/specs/skillopt-experiment-runbook.md`: operational order for preflight, smoke, scale-up, same-run baselines, locked test, and cross-agent evaluation.
- `docs/specs/skillopt-operator-handoff.md`: low-Codex-token execution handoff contract and compact result-packet format.
- `docs/specs/skillopt-current-state.md`: current objective, frozen decisions, implemented mechanism, latest results, paper gap, and next plan.

## Historical Contract-Aware Specs

- `docs/specs/skillopt-paper-gap-goals.md`: historical goal contract for the contract-aware Stage 4–7 campaign.
- `docs/specs/skillopt-paper-gap-plan.md`: historical execution plan superseded by the paper-faithful roadmap.
- `docs/specs/skillopt-gap-epics.md`: historical Epic breakdown for the contract-aware optimizer.

## Operator READMEs

- `docs/ops/skillopt-stage4-operator-readme.md`: concrete two-phase handoff for Stage 4 full-selection executive-only.
- `docs/ops/skillopt-stage5-operator-readme.md`: concrete two-phase handoff for Stage 5 same-run baseline matrix.
- `docs/ops/skillopt-stage7-operator-readme.md`: irreversible two-phase handoff for the one-attempt locked test.
- `docs/ops/templates/`: fixed Stage 4/5 and Stage 7 manifest-check and result-packet templates.

## Reading Order

1. Read `docs/specs/skillopt-paper-faithful-roadmap.md` for the active goal, implementation order, and evidence gates.
2. Read `docs/specs/skillopt-current-state.md` for current status and latest decisions.
3. Read `docs/papers/paper-notes.md` when checking a normative algorithm or experiment requirement.
4. Read `docs/specs/skillopt-experiment-runbook.md` only before operating existing historical experiment tooling; it does not define the new paper engine.
5. Read `docs/specs/skillopt-operator-handoff.md` when handing long-running execution to another agent or human operator.
6. Read `docs/skillopt-executive-protocol.md` for the historical contract-aware executive protocol.
7. Inspect Stage 5/7 artifacts only when exact historical extension metrics are needed.
8. Inspect raw per-seed artifacts only when debugging a specific historical failure.
