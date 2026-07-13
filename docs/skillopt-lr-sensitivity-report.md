# Coding-Hidden LR Sensitivity

## Goal

Use the same real external-editor proposal log and the same text-sensitive deterministic agent, then vary only the textual learning-rate budget.

This answers one question:

> Did the LR-enabled cases fail because proposals were bad, or because LR rejected useful proposals too aggressively?

## Artifacts

- Runner: `work/run_coding_hidden_lr_sensitivity.py`
- Proposal log: `runs/coding-hidden-real-external-editor-capture-v1/proposals.jsonl`
- Run output: `runs/coding-hidden-lr-sensitivity-v1`
- Summary: `runs/coding-hidden-lr-sensitivity-v1/summary.json`
- Report: `runs/coding-hidden-lr-sensitivity-v1/report.md`

## Command

```bash
python3 work/run_coding_hidden_lr_sensitivity.py \
  --out runs/coding-hidden-lr-sensitivity-v1 \
  --proposal-log runs/coding-hidden-real-external-editor-capture-v1/proposals.jsonl \
  --seeds seed-a \
  --timeout-seconds 30 \
  --agent-path work/coding_hidden_text_sensitive_agent.py
```

## Result

| Source Case | Budget | Valid | Holdout | LR Rejections |
|---|---|---:|---:|---:|
| Gate only | no LR | 0.50 | 0.50 | 0 |
| +LR | strict 600/260/1 | 0.00 | 0.00 | 2 |
| +LR | delta 320 | 0.00 | 0.00 | 2 |
| +LR | delta 480 | 0.50 | 0.50 | 1 |
| +LR | loose 750/700/3 | 0.50 | 0.50 | 0 |
| +LR+Rejected Buffer | strict 600/260/1 | 0.00 | 0.00 | 2 |
| +LR+Rejected Buffer | delta 320 | 0.50 | 0.50 | 0 |
| +LR+Rejected Buffer | delta 480 | 0.50 | 0.50 | 0 |
| +LR+Rejected Buffer | loose 750/700/3 | 0.50 | 0.50 | 0 |
| +LR+Rejected Buffer+Meta Skill | strict 600/260/1 | 0.00 | 0.00 | 2 |
| +LR+Rejected Buffer+Meta Skill | delta 320 | 0.00 | 0.00 | 2 |
| +LR+Rejected Buffer+Meta Skill | delta 480 | 0.50 | 0.50 | 1 |
| +LR+Rejected Buffer+Meta Skill | loose 750/700/3 | 0.50 | 0.50 | 0 |

Budget names mean:

- `strict 600/260/1`: max skill chars 600, max delta chars 260, max added bullet lines 1
- `delta 320`: max skill chars 600, max delta chars 320, max added bullet lines 1
- `delta 480`: max skill chars 600, max delta chars 480, max added bullet lines 1
- `loose 750/700/3`: max skill chars 750, max delta chars 700, max added bullet lines 3

## Interpretation

The original strict LR budget caused false negatives. It rejected every real proposal in the LR-enabled cases.

Useful proposal text starts passing once the delta budget is relaxed:

- `+LR+Rejected Buffer` recovers at delta 320.
- `+LR` and `+LR+Rejected Buffer+Meta Skill` recover at delta 480.
- Loosening further removes LR rejections but does not improve beyond 0.50.

So there are two distinct ceilings:

1. LR calibration ceiling: strict 260 delta is too low for real external-editor wording.
2. Proposal quality ceiling: even when accepted, these proposals only cover keyed dedupe, token parsing, and range bounds. They do not cover nested path handling, missing sort-key-last behavior, or rounding.

## Decision

Do not claim that LR hurts SkillOpt in general.

The correct conclusion is narrower:

> A fixed char-delta LR budget of 260 is too strict for this real external editor. The LR mechanism needs calibration or an adaptive policy before it can be judged fairly.

## Next Fix

Replace the brittle fixed LR default with a profile-based budget:

- `strict`: current synthetic-ablation budget
- `real-editor`: around 600 chars / 520 delta / 1 bullet
- `loose-diagnostic`: 750 chars / 700 delta / 3 bullets

Then rerun the real proposal-log ablation with `real-editor` as the default LR profile.

## Profile Follow-Up

Implemented:

- `textskill_optimizer/lr_profiles.py`
- `textskill_optimizer.cli --lr-profile`
- `work/run_coding_hidden_ablation.py --lr-profile`
- `work/run_coding_hidden_proposal_log_ablation.py --lr-profile`
- `work/run_real_external_editor_proposal_log_ablation.py --lr-profile`

Standard proposal-log replay with `real-editor`:

```bash
python3 work/run_coding_hidden_proposal_log_ablation.py \
  --out runs/coding-hidden-real-proposal-log-ablation-real-editor-profile-v1 \
  --proposal-log runs/coding-hidden-real-external-editor-capture-v1/proposals.jsonl \
  --seeds seed-a \
  --timeout-seconds 30 \
  --agent-path work/coding_hidden_text_sensitive_agent.py \
  --lr-profile real-editor
```

Result:

| Case | Valid | Holdout | LR Rejections |
|---|---:|---:|---:|
| Gate only | 0.50 | 0.50 | 0 |
| +LR | 0.50 | 0.50 | 1 |
| +LR+Rejected Buffer | 0.50 | 0.50 | 0 |
| +LR+Rejected Buffer+Meta Skill | 0.50 | 0.50 | 1 |

Decision update:

`real-editor` is a better default profile for real external-editor proposal replay. It prevents the all-rejected failure mode while still rejecting the largest over-budget proposal in `+LR` and `+LR+Rejected Buffer+Meta Skill`.

After prompt/meta v2, `real-editor` was calibrated from 480 to 520 delta. This still rejects a 703-character broad proposal, but admits the 575-character compact all-capability Meta proposal that reaches 1.00 validation and holdout in `runs/coding-hidden-prompt-v2-meta-replay-real-editor-v1`.
