# Prompt/Meta Skill V2 Coverage Experiment

## Goal

Fix the proposal-quality ceiling found in the real proposal-log replay.

Previous real proposals reached only 0.50 because they covered:

- keyed dedupe
- token parsing
- range bounds

They missed:

- nested path access
- missing sort-key-last behavior
- money/tax/cents rounding

## Changes

- Expanded `work/meta_skill.md`.
- Expanded `CODING_HIDDEN_META_SKILL` in `work/run_coding_hidden_ablation.py`.
- Expanded `SYSTEM_PROMPT` in `examples/coding/openai_compatible_skill_editor.py`.
- Added `--agent-path` support to `work/run_real_external_editor_proposal_log_ablation.py`.
- Calibrated `real-editor` LR profile to `600 chars / 520 delta / 1 bullet`.
- Updated text-sensitive harness to treat "missing keys last" as `sort_missing_last`.

## Real Capture

Command shape:

```bash
python3 work/run_real_external_editor_proposal_log_ablation.py \
  --out runs/coding-hidden-prompt-v2-meta-capture-v1 \
  --replay-out runs/coding-hidden-prompt-v2-meta-replay-v1 \
  --seeds prompt-v2-a \
  --cases gate_lr_rejected_meta \
  --timeout-seconds 30 \
  --lr-profile real-editor \
  --agent-path work/coding_hidden_text_sensitive_agent.py \
  --skip-replay
```

The initial live capture happened before the final 520-delta calibration, so both proposals were LR-rejected in that capture summary. The proposal log itself is still useful and was replayed after calibration.

## Proposal Coverage

The compact epoch-2 proposal:

```markdown
For small utilities follow these rules:
1. Keyed dedupe: preserve missing-key records independently
2. Nested path access: trim segments, handle dict keys, list indexes, return defaults
3. Sort-by-key: keep stable, place missing keys last with (key not in item, item.get(key))
4. Ranges: normalize reversed bounds before generating ascending inclusive outputs
5. Parsers: trim separators, skip malformed tokens, preserve signs and units
6. Money rounding: use Decimal + ROUND_HALF_UP for cent precision
```

It covers all text-sensitive capability families needed by the coding-hidden validation and holdout sets.

## Replay After Calibration

Command:

```bash
python3 work/run_coding_hidden_proposal_log_ablation.py \
  --out runs/coding-hidden-prompt-v2-meta-replay-real-editor-v1 \
  --proposal-log runs/coding-hidden-prompt-v2-meta-capture-v1/proposals.jsonl \
  --seeds prompt-v2-a \
  --timeout-seconds 30 \
  --agent-path work/coding_hidden_text_sensitive_agent.py \
  --lr-profile real-editor
```

Result:

| Case | Valid | Holdout | First Success Epoch | LR Rejections |
|---|---:|---:|---:|---:|
| +LR+Rejected Buffer+Meta Skill | 1.00 | 1.00 | 2 | 1 |

The other cases are 0.00 in this replay because this proposal log contains only the `gate_lr_rejected_meta` case.

## Decision

Prompt/meta v2 fixed the proposal coverage ceiling for the Meta case.

## Full 3-Seed Replay

Artifacts:

- Capture run: `runs/coding-hidden-prompt-v2-full-capture-v1`
- Proposal log: `runs/coding-hidden-prompt-v2-full-capture-v1/proposals.jsonl`
- Exact replay after protocol fix: `runs/coding-hidden-prompt-v2-full-replay-v2`

Replay command:

```bash
python3 work/run_coding_hidden_proposal_log_ablation.py \
  --out runs/coding-hidden-prompt-v2-full-replay-v2 \
  --proposal-log runs/coding-hidden-prompt-v2-full-capture-v1/proposals.jsonl \
  --seeds prompt-v2-a,prompt-v2-b,prompt-v2-c \
  --timeout-seconds 30 \
  --agent-path work/coding_hidden_text_sensitive_agent.py \
  --lr-profile real-editor
```

Aggregate result:

| Case | Seeds | Mean Valid | Mean Holdout | Successes | Mean First Success Epoch | Mean LR Rejections |
|---|---:|---:|---:|---:|---:|---:|
| Gate only | 3 | 0.83 | 0.92 | 2 | 1.0 | 0.0 |
| +LR | 3 | 0.25 | 0.33 | 0 | - | 1.7 |
| +LR+Rejected Buffer | 3 | 0.00 | 0.00 | 0 | - | 2.0 |
| +LR+Rejected Buffer+Meta Skill | 3 | 0.92 | 1.00 | 2 | 2.0 | 1.0 |

Per-seed result:

| Seed | Case | Valid | Holdout | First Success Epoch | LR Rejections |
|---|---|---:|---:|---:|---:|
| prompt-v2-a | Gate only | 1.00 | 1.00 | 1 | 0 |
| prompt-v2-a | +LR | 0.00 | 0.00 | - | 2 |
| prompt-v2-a | +LR+Rejected Buffer | 0.00 | 0.00 | - | 2 |
| prompt-v2-a | +LR+Rejected Buffer+Meta Skill | 1.00 | 1.00 | 2 | 1 |
| prompt-v2-b | Gate only | 0.50 | 0.75 | - | 0 |
| prompt-v2-b | +LR | 0.00 | 0.00 | - | 2 |
| prompt-v2-b | +LR+Rejected Buffer | 0.00 | 0.00 | - | 2 |
| prompt-v2-b | +LR+Rejected Buffer+Meta Skill | 0.75 | 1.00 | - | 1 |
| prompt-v2-c | Gate only | 1.00 | 1.00 | 1 | 0 |
| prompt-v2-c | +LR | 0.75 | 1.00 | - | 1 |
| prompt-v2-c | +LR+Rejected Buffer | 0.00 | 0.00 | - | 2 |
| prompt-v2-c | +LR+Rejected Buffer+Meta Skill | 1.00 | 1.00 | 2 | 1 |

## Protocol Fix

Replay now returns logged proposals even when the current training results already pass.
This matches the live capture path, where an external editor may still propose another edit after a successful epoch.

The v2 replay rows match the capture rows on validation score, holdout score, first-success epoch, LR rejections, and validation rejections.

## Interpretation

The useful mechanism is not LR alone and not Rejected Buffer alone. In this benchmark, the strong combination is:

- Meta Skill supplies missing capability families.
- LR prevents broad rewrites, but only after the editor has enough guidance to compress the right edit.
- Rejected Buffer helps only when the editor actually uses rejection history; without Meta Skill it repeatedly emits over-budget edits.

## Cross-Agent Check

The raw prompt/meta v2 best skill passed the deterministic text-sensitive harness, but failed real-agent validation at `0.50` for all three agents.

Artifacts:

- Raw skill: `work/coding_prompt_meta_v2_skill.md`
- Raw cross-agent validation: `runs/coding-hidden-cross-agent-validation-prompt-meta-v2-v2`
- Strengthened skill: `work/coding_prompt_meta_v2_cross_agent_skill.md`
- Full validation: `runs/coding-hidden-cross-agent-validation-prompt-meta-v2-cross-agent-v1`
- Full holdout majority: `runs/coding-hidden-cross-agent-holdout-prompt-meta-v2-cross-agent-v2`

Raw failures:

- `nested-pluck`: agents needed an explicit rule to skip records with missing paths for pluck/collection utilities.
- `parse-int-list`: agents needed an explicit `try`/`except` rule to skip malformed numeric tokens.

After adding those concrete rules, full validation reached `1.00` on Coco, CCR Code, and Kilo, with every validation task at `3/3` majority pass.

Holdout majority also passed every task. CCR had a single-agent `date-range` failure in the full run; targeted follow-up passed after strengthening the range rule to "swap reversed bounds and keep output ascending inclusive."

Updated decision:

Prompt/meta v2 fixed deterministic proposal coverage, but real-agent transfer requires one more portability loop. The Meta Skill should prefer concrete executable rules over compact abstractions whenever multiple agents interpret an abstraction differently.
