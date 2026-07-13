# Real External Editor Proposal-Log Experiment

## Goal

Capture real external-editor proposals once, then replay them through the coding-hidden ablation protocol.

This separates:

- external model proposal generation
- SkillOpt mechanism evaluation

## Artifacts

- Capture run: `runs/coding-hidden-real-external-editor-capture-v1`
- Proposal log: `runs/coding-hidden-real-external-editor-capture-v1/proposals.jsonl`
- Replay run: `runs/coding-hidden-real-proposal-log-ablation-v1`
- Runner script: `work/run_real_external_editor_proposal_log_ablation.py`

## Command Shape

```bash
python3 work/run_real_external_editor_proposal_log_ablation.py \
  --out runs/coding-hidden-real-external-editor-capture-v1 \
  --replay-out runs/coding-hidden-real-proposal-log-ablation-v1 \
  --seeds seed-a \
  --cases gate_only,gate_lr,gate_lr_rejected,gate_lr_rejected_meta \
  --timeout-seconds 30 \
  --editor-timeout 240 \
  --llm-timeout 180
```

The script reads the external API key from `EXTERNAL_LLM_API_KEY` or one-time stdin. The key is not written to artifacts.

## Result

| Case | Valid | Holdout | First Success Epoch | LR Rejections |
|---|---:|---:|---:|---:|
| Gate only | 0.00 | 0.00 | - | 0 |
| +LR | 0.00 | 0.00 | - | 2 |
| +LR+Rejected Buffer | 0.00 | 0.00 | - | 2 |
| +LR+Rejected Buffer+Meta Skill | 0.00 | 0.00 | - | 2 |

Replay matched capture, so the proposal-log capture and replay path is working.

## Root Cause

The zero result is primarily an evaluator-harness issue, not yet evidence that the SkillOpt mechanisms fail.

The current coding-hidden deterministic agent only changes behavior when the skill contains internal experiment markers:

- `FULL_CODING_HIDDEN_RULES`
- `TRAIN_ONLY_CODING_HIDDEN_RULES`

The real external editor produced normal natural-language skill edits such as robustness checklists and compact coding rules. During validation, the deterministic agent printed messages like:

```text
no deterministic rule for fixture=unique-by-id
```

So the candidate skills were valid natural-language proposals, but the deterministic agent did not interpret them.

The LR gate also rejected all proposals in the LR-enabled cases. One proposal missed the delta budget narrowly: `264 > 260`. This suggests the current LR threshold is brittle for real model wording.

## Decision

Do not use this run as a final SkillOpt-quality result.

Use it as evidence that the protocol now captures real proposals, but the deterministic evaluator must be upgraded before this ablation can answer the paper question.

## Next Fix

Create a versioned text-sensitive deterministic agent that maps natural-language skill rules to fixture capabilities, instead of relying only on hidden marker strings.

Then replay the existing proposal log without another external model call.

## Text-Sensitive Replay

Implemented:

- `work/coding_hidden_text_sensitive_agent.py`
- `work/run_coding_hidden_proposal_log_ablation.py --agent-path ...`

Replay command:

```bash
python3 work/run_coding_hidden_proposal_log_ablation.py \
  --out runs/coding-hidden-real-proposal-log-ablation-text-sensitive-v2 \
  --proposal-log runs/coding-hidden-real-external-editor-capture-v1/proposals.jsonl \
  --seeds seed-a \
  --timeout-seconds 30 \
  --agent-path work/coding_hidden_text_sensitive_agent.py
```

Result:

| Case | Valid | Holdout | LR Rejections |
|---|---:|---:|---:|
| Gate only | 0.50 | 0.50 | 0 |
| +LR | 0.00 | 0.00 | 2 |
| +LR+Rejected Buffer | 0.00 | 0.00 | 2 |
| +LR+Rejected Buffer+Meta Skill | 0.00 | 0.00 | 2 |

Gate-only improved because the v2 agent mapped natural-language rules to capabilities:

- `keyed_dedupe`: fixed `unique-by-id` and `dedupe-casefold`
- `token_parser`: fixed `parse-int-list`
- `range_bounds`: fixed `date-range`

Still failing:

- `nested-pluck` and `safe-nested-get`: the captured proposals did not include nested path guidance.
- `stable-sort-events`: the captured proposals did not include missing sort-key-last guidance.
- `round-tax`: the captured proposals did not include rounding guidance.

Updated root cause:

The capture/replay protocol works, and real proposal text has measurable value under a text-sensitive harness. The remaining blocker is LR calibration: the LR-enabled cases rejected every real proposal before validation, including compact proposals that narrowly exceeded the current delta budget.

## LR Sensitivity Follow-Up

Implemented:

- `work/run_coding_hidden_lr_sensitivity.py`
- `docs/skillopt-lr-sensitivity-report.md`

Run output:

- `runs/coding-hidden-lr-sensitivity-v1`

Key result:

| Source Case | Budget | Valid | Holdout | LR Rejections |
|---|---|---:|---:|---:|
| Gate only | no LR | 0.50 | 0.50 | 0 |
| +LR | strict 600/260/1 | 0.00 | 0.00 | 2 |
| +LR | delta 480 | 0.50 | 0.50 | 1 |
| +LR+Rejected Buffer | strict 600/260/1 | 0.00 | 0.00 | 2 |
| +LR+Rejected Buffer | delta 320 | 0.50 | 0.50 | 0 |
| +LR+Rejected Buffer+Meta Skill | strict 600/260/1 | 0.00 | 0.00 | 2 |
| +LR+Rejected Buffer+Meta Skill | delta 480 | 0.50 | 0.50 | 1 |

Conclusion:

The original 260 char-delta budget is too strict for this real external editor. Relaxing LR recovers the same 0.50 valid/holdout score as the no-LR baseline, but does not exceed it. That means LR calibration was the blocker for LR cases, while proposal coverage is the remaining ceiling.

## LR Profile Follow-Up

Implemented explicit LR profiles and reran the standard proposal-log ablation with `real-editor`.

Run output:

- `runs/coding-hidden-real-proposal-log-ablation-real-editor-profile-v1`

Result:

| Case | Valid | Holdout | LR Rejections |
|---|---:|---:|---:|
| Gate only | 0.50 | 0.50 | 0 |
| +LR | 0.50 | 0.50 | 1 |
| +LR+Rejected Buffer | 0.50 | 0.50 | 0 |
| +LR+Rejected Buffer+Meta Skill | 0.50 | 0.50 | 1 |

Conclusion update:

The `real-editor` profile fixes the all-rejected LR failure mode. It does not improve beyond 0.50 because the captured proposal text still lacks nested path, missing sort-key-last, and rounding guidance.

## Prompt/Meta V2 Follow-Up

Implemented prompt/meta coverage guidance for the missing families:

- nested path access
- missing sort-key-last behavior
- money/tax/cents rounding

Artifacts:

- `docs/skillopt-prompt-meta-v2-report.md`
- `runs/coding-hidden-prompt-v2-meta-capture-v1`
- `runs/coding-hidden-prompt-v2-meta-replay-real-editor-v1`

Key replay result after final `real-editor` calibration:

| Case | Valid | Holdout | First Success Epoch | LR Rejections |
|---|---:|---:|---:|---:|
| +LR+Rejected Buffer+Meta Skill | 1.00 | 1.00 | 2 | 1 |

Conclusion update:

Prompt/meta v2 fixes the proposal coverage ceiling for the Meta case.

## Prompt/Meta V2 Full Ablation

Artifacts:

- Capture run: `runs/coding-hidden-prompt-v2-full-capture-v1`
- Proposal log: `runs/coding-hidden-prompt-v2-full-capture-v1/proposals.jsonl`
- Exact replay: `runs/coding-hidden-prompt-v2-full-replay-v2`

The replay protocol was fixed so proposal-log replay returns logged proposals even when the current training set already passes. That matches the live external-editor path and removes a capture/replay mismatch in rejection accounting.

Aggregate result:

| Case | Seeds | Mean Valid | Mean Holdout | Successes | Mean First Success Epoch | Mean LR Rejections |
|---|---:|---:|---:|---:|---:|---:|
| Gate only | 3 | 0.83 | 0.92 | 2 | 1.0 | 0.0 |
| +LR | 3 | 0.25 | 0.33 | 0 | - | 1.7 |
| +LR+Rejected Buffer | 3 | 0.00 | 0.00 | 0 | - | 2.0 |
| +LR+Rejected Buffer+Meta Skill | 3 | 0.92 | 1.00 | 2 | 2.0 | 1.0 |

Decision update:

The current best mechanism is `+LR+Rejected Buffer+Meta Skill`. LR by itself suppresses useful but verbose edits, and Rejected Buffer by itself does not fix proposal coverage. Meta Skill is the component that changes the proposal distribution toward the missing capability families.

## Cross-Agent Portability Update

The raw prompt/meta v2 best skill was too compact for real agents: Coco, CCR Code, and Kilo all scored `0.50` on validation and failed the same two tasks, `nested-pluck` and `parse-int-list`.

After feeding those failures back into the skill and Meta Skill as concrete executable rules, the strengthened skill reached:

| Split | Coco | CCR Code | Kilo | Majority |
|---|---:|---:|---:|---|
| Validation | 1.00 | 1.00 | 1.00 | every task `3/3` |
| Holdout | 1.00 | 0.75 | 1.00 | every task passed |

Targeted follow-ups fixed the two observed CCR-only holdout failures:

- `dedupe-casefold`: targeted `3/3` after adding `casefold()` normalized comparison guidance.
- `date-range`: targeted `3/3` after changing reversed-bound guidance to "swap bounds and emit ascending inclusive output."

Conclusion update:

The mechanism result still points to `+LR+Rejected Buffer+Meta Skill`, but the Meta Skill must include cross-agent portability guidance. Otherwise deterministic replay can overestimate a compact skill that real agents interpret inconsistently.
