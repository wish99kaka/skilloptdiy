# Traceability Matrix

| PRD ID | Requirement | Design Section | Technical Tradeoff | Spec ID | Scenario / Boundary / Non-goal | Code Location | Test Coverage | Evidence | Status | Risk |
|---|---|---|---|---|---|---|---|---|---|---|
| DEC-001 | Optimize for Codex development-task tokens, not experiment-model tokens. | Roadmap 11.1 | Experiment receipts cannot measure the Codex task itself. | Current State / Frozen Decisions | Model tokens remain observable but cannot drive go/no-go. | `preregistration._validate_budgets` | `test_requires_model_tokens_to_be_audit_only` | Full suite pass | Pass | Low |
| DEC-002 | Target and optimizer token usage is audit-only. | Roadmap 11.1 | Preserve actual usage without a hard token stop. | `budgets.token_policy=audit_only` | Token projections and actual totals remain in artifacts. | `TargetBudgetGuard`, `PaidBudgetGuard`, `_usage_summary` | `test_model_tokens_are_audit_only_but_call_caps_still_stop` | Targeted and full suites pass | Pass | Low |
| DEC-003 | Call counts and wall time remain hard safety limits. | Roadmap 11.1 | Safety is bounded without treating model tokens as cost. | Preregistered call/time budgets | Calls beyond the cap and expired deadlines stop execution. | `TargetBudgetGuard`, `PaidBudgetGuard`, `_require_within_budgets` | `test_model_tokens_are_audit_only_but_call_caps_still_stop` | Unit suite pass | Pass | Low |
| DEC-004 | Selection saturation must persist an auditable stop outcome. | WP5 selection gate | Persist only the scalar; do not expose selection items to optimization. | Stop receipt v1 | Receipt precedes the failing exit and permanently blocks rerun. | `run_searchqa_experiment` | `test_selection_saturation_writes_a_single_use_stop_receipt` | Integration suite pass | Pass | Low |
| DEC-005 | Development runs cannot access test data. | N1 data boundary | Stop receipts must preserve the same access evidence. | `test_access={allowed:false,attempt:0}` | Test payload stays unmaterialized. | `run_searchqa_experiment`, preregistration validator | Stop-receipt and zero-call integration tests | Conformance/provenance pass | Pass | Low |

## Status Legend

- `Pass`: traced and verified.
- `Partial`: partially traced or partially verified.
- `Missing`: expected artifact or implementation is absent.
- `Conflict`: artifacts contradict each other.
- `Unverified`: evidence is insufficient.
- `Out of Scope`: explicitly excluded from this delivery.
