# Evidence Log

## Audit Baseline

- Repository: `/Users/bytedance/CodeA/skilloptdiy`
- Commit: `c8087f9997d7c326ee8d6c9861b759d4ba698698` plus reviewed local diff
- Branch: `main`, tracking `origin/main`
- Requirement source: user decisions in the current task
- Design/spec sources: `README.md`, `docs/specs/skillopt-current-state.md`, `docs/specs/skillopt-paper-faithful-roadmap.md`
- Test files: SearchQA preregistration, controller, and experiment unit/integration suites
- Coverage files: none configured
- CI: pending commit/push; local evidence used for this audit

## Commands Run

| Command | Result | Notes |
|---|---|---|
| `git status -sb`, `git rev-parse HEAD`, `git branch -vv`, `git remote -v` | Pass | Baseline and tracking branch recorded. |
| `.venv/bin/python -m pytest -q tests/conformance/unit/test_paper_searchqa.py tests/conformance/integration/test_paper_searchqa_experiment.py tests/conformance/unit/test_paper_preregistration.py` | Pass | 20 passed before the final policy-validation test was added. |
| `.venv/bin/python -m pytest -q tests/conformance tests/provenance` | Pass | 114 passed, 1 skipped, 19 subtests passed. |
| `.venv/bin/python -m compileall -q textskill_optimizer scripts` | Pass | No Python compilation errors. |
| `.venv/bin/python -m pytest -q` | Pass | 376 passed, 1 skipped, 21 subtests passed. |
| Targeted suite after schema-v2 review fix | Pass | 21 passed. |
| Full suite after schema-v2 review fix | Pass | 376 passed, 1 skipped, 21 subtests passed. |
| `git diff --check` | Pass | No whitespace errors. |

## Tool-Assisted Reviews

| Tool | Available | Scope | Result | Notes |
|---|---|---|---|---|
| Cool Review | Yes | Specs -> code -> tests | Pass | This audit package. |
| Spec Kit | No project integration | Not used | N/A | Manual traceability was sufficient for the bounded change. |
| OpenSpec | No project integration | Not used | N/A | No change manifest was present. |
| BMad | Skills available | Not used | N/A | Active repository specs were directly reviewed. |

## Coverage Summary

No coverage tool is configured. Requirement-level evidence is provided by the
targeted unit/integration tests and the full repository suite.

## Manual Review Notes

- `token_policy` is validated as exactly `audit_only` and is mechanically added
  to new schema-v2 dry-run and paid preregistrations.
- Token totals remain in usage records and receipts.
- Target and optimizer reservation APIs return zero token reservations, so
  actual/estimated token totals cannot trigger a stop.
- Saturation receipts are deliberately a separate schema because they are not
  successful mechanism evidence and carry `claim_class=null`.
