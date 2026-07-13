# Specs

Project facts and decision records that should be read before re-reviewing raw experiment artifacts.

## Active Specs

- `docs/specs/skillopt-paper-faithful-roadmap.md`: canonical roadmap for the new paper-faithful engine, conformance gates, evidence program, and claim provenance.
- `docs/specs/skillopt-current-state.md`: current objective, frozen decisions, implemented mechanism, latest results, paper gap, and next plan.

## Historical Contract-Aware Specs

- `docs/specs/skillopt-experiment-runbook.md`: completed Stage 4–7 operational order; historical tooling only and not authority for the new paper engine.
- `docs/specs/skillopt-operator-handoff.md`: completed low-token execution handoff contract for the historical campaign.
- `docs/specs/skillopt-paper-gap-goals.md`: historical goal contract for the contract-aware Stage 4–7 campaign.
- `docs/specs/skillopt-paper-gap-plan.md`: historical execution plan superseded by the paper-faithful roadmap.
- `docs/specs/skillopt-gap-epics.md`: historical Epic breakdown for the contract-aware optimizer.

## Operator READMEs

- `docs/ops/skillopt-stage4-operator-readme.md`: completed Stage 4 handoff, retained for audit only.
- `docs/ops/skillopt-stage5-operator-readme.md`: completed Stage 5 handoff, retained for audit only.
- `docs/ops/skillopt-stage7-operator-readme.md`: completed one-attempt Stage 7 handoff, retained for audit only.
- `docs/ops/templates/`: fixed Stage 4/5 and Stage 7 manifest-check and result-packet templates.

## Reading Order

1. Read `docs/specs/skillopt-paper-faithful-roadmap.md` for the active goal, implementation order, and evidence gates.
2. Read `docs/specs/skillopt-current-state.md` for current status and latest decisions.
3. Read `docs/papers/paper-notes.md` when checking a normative algorithm or experiment requirement.
4. Read `docs/specs/skillopt-experiment-runbook.md` only when auditing the completed historical campaign; do not execute it.
5. Read `docs/specs/skillopt-operator-handoff.md` only when auditing the completed handoff protocol.
6. Read `docs/skillopt-executive-protocol.md` for the historical contract-aware executive protocol.
7. Inspect Stage 5/7 artifacts only when exact historical extension metrics are needed.
8. Inspect raw per-seed artifacts only when debugging a specific historical failure.
