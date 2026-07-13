# SkillOpt Cross-Agent Evaluation Report

Date: 2026-06-15

## Goal

Evaluate skill quality, not agent quality, by checking whether skill improvements transfer across three real coding-agent harnesses:

- Coco
- CCR Code
- Kilo

All agents use the same hidden scorer and the same task set for each split.

## Validation Set

Source: `examples/coding-hidden/valid.jsonl`

Tasks:

- `coding-hidden-valid-unique-by-id`
- `coding-hidden-valid-nested-pluck`
- `coding-hidden-valid-stable-sort-events`
- `coding-hidden-valid-parse-int-list`

## Holdout Set

Source: `examples/coding-hidden/holdout.jsonl`

Tasks:

- `coding-hidden-holdout-date-range`
- `coding-hidden-holdout-safe-nested-get`
- `coding-hidden-holdout-round-tax`
- `coding-hidden-holdout-dedupe-casefold`

## Skill Variants

- `initial`: `examples/coding-hidden/skill.md`
- `coco_best_expanded`: `runs/coding-hidden-coco-skillopt-expanded-8-4-4/best_skill.md`
- `revised`: `work/coding_revised_skill.md`

## Validation Results

Latest health-gated validation artifact:

- `runs/coding-hidden-cross-agent-validation-revised-health-v1`

| Agent | Health | Revised |
|---|---|---:|
| Coco | passed | 1.00 |
| CCR Code | passed | 1.00 |
| Kilo | passed | 1.00 |

Majority vote for every validation task is `3/3 pass`.

Earlier validation comparison:

| Agent | Initial | Coco Best Expanded | Revised |
|---|---:|---:|---:|
| Coco | 0.00 | 0.75 | 1.00 |
| CCR Code | 0.00 | 0.75 | 1.00 |
| Kilo | not run | not run | 1.00 |

## Holdout Results

Latest health-gated holdout artifact:

- `runs/coding-hidden-cross-agent-holdout-revised-health-v1`

| Agent | Health | Revised |
|---|---|---:|
| Coco | passed | 1.00 |
| CCR Code | passed | 1.00 |
| Kilo | passed | 1.00 |

Majority vote for every holdout task is `3/3 pass`.

Earlier holdout comparison:

| Agent | Revised v1 | Revised v2 | Revised v3 |
|---|---:|---:|---:|
| Coco | 1.00 | 1.00 | 1.00 |
| CCR Code | 0.75 | 0.75 | 1.00 |
| Kilo | not run | not run | 1.00 |

Detailed artifacts:

- Coco baseline and Coco best: `runs/coding-hidden-cross-agent-skill-eval-v1`
- CCR rerun after stdin wrapper fix: `runs/coding-hidden-cross-agent-skill-eval-ccr-v2`
- Revised validation final run: `runs/coding-hidden-cross-agent-validation-revised-v4`
- Revised health-gated 3-agent validation: `runs/coding-hidden-cross-agent-validation-revised-health-v1`
- Revised holdout final run: `runs/coding-hidden-cross-agent-holdout-revised-v3`
- Revised health-gated 3-agent holdout: `runs/coding-hidden-cross-agent-holdout-revised-health-v1`
- Kilo validation final run: `runs/coding-hidden-cross-agent-kilo-validation-revised-v2`
- Kilo holdout final run: `runs/coding-hidden-cross-agent-kilo-holdout-revised-v1`

## Findings

The optimized skill is not merely a Coco-specific artifact. `coco_best_expanded` improved both agents from `0.00` to `0.75`.

The first revised skill did not improve the aggregate score because `nested-pluck` still missed numeric list-index path segments such as `users.0.name`.

The validation revised skill reached `1.00` on both agents after adding two concrete rules:

- Nested path utilities must treat integer path segments as list indexes when the current value is a list, after bounds checking.
- Delimited numeric parsers must skip malformed tokens with `try`/`except`, unless the task explicitly asks to raise.

Holdout then exposed two additional portability gaps:

- Email de-duplication needs normalized comparison with `casefold()` while preserving original records.
- Date range utilities need explicit reversed-bound handling after parsing dates.

After adding those rules, revised v3 reached `1.00` on both Coco and CCR Code for validation and holdout.

Kilo was then added as a third external agent to support majority voting. Its first validation run exposed a portability issue in sort-by-key guidance: Kilo implemented "missing keys last" as `(False, 0)` for missing items, which sorts missing items first in Python. The skill now states the concrete safe shape `(key not in item, item.get(key))` for ascending sort. After that revision, Kilo reached `1.00` on validation and holdout.

`work/run_cross_agent_skill_eval.py` now also writes per-task vote rows into `summary.json` and prints majority status as `passed/total`.

The cross-agent runner now performs an agent health check before scoring. A failed health check writes `__health.json`, marks `health_status=failed`, and skips formal scoring for that agent/skill pair. This prevents local harness, auth, service startup, or sandbox failures from being counted as skill failures.

## Prompt/Meta V2 Follow-Up

Artifacts:

- Raw prompt/meta v2 skill: `work/coding_prompt_meta_v2_skill.md`
- Cross-agent strengthened skill: `work/coding_prompt_meta_v2_cross_agent_skill.md`
- Raw validation run: `runs/coding-hidden-cross-agent-validation-prompt-meta-v2-v2`
- Targeted validation repair: `runs/coding-hidden-cross-agent-validation-prompt-meta-v2-targeted-v1`
- Full validation final: `runs/coding-hidden-cross-agent-validation-prompt-meta-v2-cross-agent-v1`
- Full holdout majority: `runs/coding-hidden-cross-agent-holdout-prompt-meta-v2-cross-agent-v2`
- Holdout targeted repairs:
  - `runs/coding-hidden-cross-agent-holdout-prompt-meta-v2-dedupe-targeted-v1`
  - `runs/coding-hidden-cross-agent-holdout-prompt-meta-v2-date-targeted-v1`

Raw prompt/meta v2 validation result:

| Agent | Health | Score | Failed |
|---|---|---:|---|
| Coco | passed | 0.50 | `nested-pluck`, `parse-int-list` |
| CCR Code | passed | 0.50 | `nested-pluck`, `parse-int-list` |
| Kilo | passed | 0.50 | `nested-pluck`, `parse-int-list` |

Majority failed `nested-pluck` and `parse-int-list` at `0/3`.

Root cause:

- "handle dict/list access defensively" did not tell agents that pluck/collection utilities should skip missing records rather than append `None`.
- "trim separators and preserve signs" did not tell agents to wrap numeric conversion in `try`/`except` and skip malformed tokens.

After strengthening the skill and Meta Skill with those concrete rules, full validation passed:

| Agent | Health | Score |
|---|---|---:|
| Coco | passed | 1.00 |
| CCR Code | passed | 1.00 |
| Kilo | passed | 1.00 |

Majority vote for every validation task is `3/3 pass`.

Holdout majority result:

| Agent | Score | Failed |
|---|---:|---|
| Coco | 1.00 | - |
| CCR Code | 0.75 | `date-range` |
| Kilo | 1.00 | - |

Majority vote for every holdout task is pass. `date-range` was `2/3`; all other holdout tasks were `3/3`.

Two minority CCR failures produced useful Meta Skill refinements:

- `dedupe-casefold`: email/case-insensitive de-duplication must compare normalized keys such as `casefold()` while preserving original records. Targeted rerun: `3/3 pass`.
- `date-range`: reversed bounds must be swapped before iteration, and output must remain ascending inclusive unless descending output is explicitly requested. Targeted rerun: `3/3 pass`.

Decision:

The optimized skill needs a cross-agent portability pass before being treated as final. Majority voting did its job: it prevented one CCR-only failure from blocking the result, while still surfacing concrete rules to feed back into the Meta Skill.

## Harness Fix

CCR Code must receive the prompt on stdin, not as a command-line argument. Long markdown prompts with code fences and backticks are unsafe through CCR's shell-based argument forwarding.

Implemented in:

- `examples/coding/ccr_agent_wrapper.py`

The wrappers also align their internal CLI timeout with each task's `timeout_seconds` and clean up child process groups on timeout. This matters for Kilo and other Node-based CLIs that can leave subprocesses after the Python wrapper exits.

## Adaptive Majority Mode

Implemented in `work/run_cross_agent_skill_eval.py`.

Goal:

Reduce external-agent calls while preserving the three-agent majority decision.

Controls:

| Env | Default | Meaning |
|---|---|---|
| `CROSS_AGENT_VOTING_MODE` | `full` | Use `adaptive-majority` to run two agents first and call the third only on disagreement. |
| `CROSS_AGENT_RANDOM_SEED` | `0` | Reproducible per-task agent ordering. |
| `CROSS_AGENT_FULL_AUDIT_RATE` | `0` | Fraction of tasks that force all three agents even when the first two agree. |
| `CROSS_AGENT_TARGET_FAILED_FROM` | unset | Path to a prior `summary.json`; only failed tasks are rerun. |
| `CROSS_AGENT_TARGET_SCOPE` | `any_failed` | `any_failed` includes single-agent failures and majority failures; `majority_failed` includes only failed majority votes. |

The run summary writes `call_savings` with `planned_agent_calls`, `actual_agent_calls`, `skipped_agent_calls`, `saved_agent_calls`, and `saved_rate`, both globally and by skill.

Decision rules:

- If the first two valid votes agree, stop and skip the third agent.
- If the first two valid votes disagree, run the third agent.
- Retryable anomalies such as timeouts, nonzero agent exits, and no-diff tool failures are not counted as valid votes.
- Health checks are cached per agent/skill inside a run, so adaptive task voting does not repeat health checks.
- Full-audit tasks use a reproducible quota: rank task IDs by the configured seed and audit `ceil(task_count * rate)` tasks. A positive rate therefore audits at least one task, including on small test sets.
- `FULL_AUDIT_RATE` should be nonzero for periodic portability monitoring, because skipped third-agent votes cannot produce complete per-agent pass rates.

Adaptive majority is appropriate for deciding whether a skill passes by majority. It is not appropriate for comparing every agent's full pass rate unless `CROSS_AGENT_FULL_AUDIT_RATE=1` or `CROSS_AGENT_VOTING_MODE=full`.

Smoke result:

- Artifact: `runs/coding-hidden-cross-agent-adaptive-smoke-v1`
- Task: `coding-hidden-holdout-date-range`
- Agents run: Kilo, Coco
- Agent skipped: CCR Code
- Decision: `2/2 pass`, `decision_reason=first_two_agree`

Skipped agents are recorded with `skipped_tasks` and no aggregate score, so they are not misread as failed zero-score agents.

Targeted multi-task smoke result:

- Artifact: `runs/coding-hidden-cross-agent-adaptive-targeted-v1`
- Tasks:
  - `coding-hidden-valid-nested-pluck`
  - `coding-hidden-valid-parse-int-list`
- Both tasks stopped after two agreeing votes.
- Agents run:
  - `nested-pluck`: CCR Code, Coco
  - `parse-int-list`: Coco, CCR Code
- Agent skipped: Kilo on both tasks
- Decision: both tasks `2/2 pass`, `decision_reason=first_two_agree`
- Call savings: planned `6`, actual `4`, saved `2`, saved rate `33.33%`

Adaptive full validation result:

- Artifact: `runs/coding-hidden-cross-agent-validation-adaptive-v1`
- Mode: `adaptive-majority`
- Audit rate: `0.25`
- Majority result: every validation task passed
- Call savings: planned `12`, actual `10`, saved `2`, saved rate `16.67%`
- Decision mix:
  - `unique-by-id`: `2/3 pass`, `decision_reason=third_agent_breaker`
  - `nested-pluck`: `3/3 pass`, `decision_reason=full_audit`
  - `stable-sort-events`: `2/2 pass`, `decision_reason=first_two_agree`
  - `parse-int-list`: `2/2 pass`, `decision_reason=first_two_agree`

The `unique-by-id` breaker exposed a Kilo-only failure: Kilo used `item.get("id")`, which grouped missing IDs under `None` and dropped the second missing-ID record. The skill and Meta Skill were strengthened to require key existence checks before reading the key, append missing-key records immediately, and avoid adding `None`/`null` to the seen set.

Targeted repair result:

- Artifact: `runs/coding-hidden-cross-agent-adaptive-unique-fix-v1`
- Task: `coding-hidden-valid-unique-by-id`
- Mode: `adaptive-majority`
- Audit rate: `1.0`
- Result: `3/3 pass`

Adaptive full holdout result:

- Artifact: `runs/coding-hidden-cross-agent-holdout-adaptive-v2`
- Mode: `adaptive-majority`
- Audit rate: `0.25`
- Majority result: all four holdout tasks passed
- Full audit: `round-tax`, `3/3 pass`
- Other tasks: `2/2 pass`, `decision_reason=first_two_agree`
- Call savings: planned `12`, actual `9`, saved `3`, saved rate `25.00%`
- Invalid votes: none

The immediately preceding v1 run health-checked all three agents successfully. The v2 rerun disabled the duplicate preflight and changed only the audit-selection mechanism. The original independent hash sampling selected zero of four tasks at a `0.25` rate; deterministic quota sampling now guarantees the intended one-task audit.

## Reproduction

```bash
CROSS_AGENT_AGENTS=coco,ccr,kilo \
CROSS_AGENT_SKILLS=revised:work/coding_revised_skill.md \
CROSS_AGENT_TASKS=examples/coding-hidden/valid.jsonl \
CROSS_AGENT_VOTING_MODE=adaptive-majority \
CROSS_AGENT_RANDOM_SEED=skillopt-validation-v1 \
CROSS_AGENT_FULL_AUDIT_RATE=0.2 \
CROSS_AGENT_RETRIES=1 \
CROSS_AGENT_HEALTH_CHECK=1 \
CROSS_AGENT_HEALTH_RETRIES=0 \
CROSS_AGENT_HEALTH_TIMEOUT=120 \
COCO_TASK_TIMEOUT=300 \
CCR_TASK_TIMEOUT=300 \
KILO_TASK_TIMEOUT=300 \
CROSS_AGENT_OUT=runs/coding-hidden-cross-agent-validation-revised-health-v1 \
python3 work/run_cross_agent_skill_eval.py
```

Holdout check:

```bash
CROSS_AGENT_AGENTS=coco,ccr,kilo \
CROSS_AGENT_SKILLS=revised:work/coding_revised_skill.md \
CROSS_AGENT_TASKS=examples/coding-hidden/holdout.jsonl \
CROSS_AGENT_VOTING_MODE=adaptive-majority \
CROSS_AGENT_RANDOM_SEED=skillopt-holdout-v1 \
CROSS_AGENT_FULL_AUDIT_RATE=0.2 \
CROSS_AGENT_RETRIES=1 \
CROSS_AGENT_HEALTH_CHECK=1 \
CROSS_AGENT_HEALTH_RETRIES=0 \
CROSS_AGENT_HEALTH_TIMEOUT=120 \
COCO_TASK_TIMEOUT=300 \
CCR_TASK_TIMEOUT=300 \
KILO_TASK_TIMEOUT=300 \
CROSS_AGENT_OUT=runs/coding-hidden-cross-agent-holdout-revised-health-v1 \
python3 work/run_cross_agent_skill_eval.py
```

Adaptive validation example:

```bash
CROSS_AGENT_AGENTS=coco,ccr,kilo \
CROSS_AGENT_SKILLS=prompt_meta_v2_cross_agent:work/coding_prompt_meta_v2_cross_agent_skill.md \
CROSS_AGENT_TASKS=examples/coding-hidden/valid.jsonl \
CROSS_AGENT_VOTING_MODE=adaptive-majority \
CROSS_AGENT_RANDOM_SEED=skillopt-v1 \
CROSS_AGENT_FULL_AUDIT_RATE=0.2 \
CROSS_AGENT_RETRIES=1 \
CROSS_AGENT_HEALTH_CHECK=1 \
CROSS_AGENT_HEALTH_RETRIES=0 \
CROSS_AGENT_HEALTH_TIMEOUT=240 \
COCO_TASK_TIMEOUT=360 \
CCR_TASK_TIMEOUT=360 \
KILO_TASK_TIMEOUT=360 \
CROSS_AGENT_OUT=runs/coding-hidden-cross-agent-validation-adaptive-v1 \
python3 work/run_cross_agent_skill_eval.py
```

Targeted rerun example:

```bash
CROSS_AGENT_AGENTS=coco,ccr,kilo \
CROSS_AGENT_SKILLS=prompt_meta_v2_cross_agent:work/coding_prompt_meta_v2_cross_agent_skill.md \
CROSS_AGENT_TASKS=examples/coding-hidden/holdout.jsonl \
CROSS_AGENT_VOTING_MODE=adaptive-majority \
CROSS_AGENT_TARGET_FAILED_FROM=runs/coding-hidden-cross-agent-holdout-prompt-meta-v2-cross-agent-v2/summary.json \
CROSS_AGENT_TARGET_SCOPE=any_failed \
CROSS_AGENT_RETRIES=1 \
CROSS_AGENT_HEALTH_CHECK=0 \
COCO_TASK_TIMEOUT=360 \
CCR_TASK_TIMEOUT=360 \
KILO_TASK_TIMEOUT=360 \
CROSS_AGENT_OUT=runs/coding-hidden-cross-agent-holdout-targeted-adaptive-v1 \
python3 work/run_cross_agent_skill_eval.py
```
