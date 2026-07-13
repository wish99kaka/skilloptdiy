# SkillOpt Minimal Stability Ablation

Date: 2026-06-15

## Goal

Measure the optimizer mechanisms themselves without mixing in external-agent variance.

This controlled ablation uses a scripted editor and synthetic scorer:

- a train-only oversized edit can satisfy visible training behavior but fails validation
- a compact general edit transfers to validation
- the editor can only switch strategy when it receives rejected-buffer feedback or meta-skill guidance

Artifact:

- `runs/optimizer-ablation-minimal-v1`

## Results

| Case | Final Score | First Success Epoch | LR Rejections | Validated Rejections | Rejected Total |
|---|---:|---:|---:|---:|---:|
| Gate only | 0.00 | - | 0 | 2 | 2 |
| +LR | 0.00 | - | 2 | 0 | 2 |
| +LR+Rejected Buffer | 1.00 | 2 | 1 | 0 | 1 |
| +LR+Rejected Buffer+Meta Skill | 1.00 | 1 | 0 | 0 | 0 |

## Interpretation

- Validation Gate blocks the bad train-only edit, but does not teach the editor to stop repeating it.
- Textual LR blocks oversized edits before validation, changing the failure mode from validation failure to budget rejection.
- Rejected Buffer gives the editor a memory of that failure, so it emits the compact transferable rule in epoch 2.
- Meta Skill provides optimizer-side prior guidance, so the editor emits the compact transferable rule in epoch 1.

## Limit

This is a mechanism smoke test, not a paper-grade ablation. It isolates causal behavior in a controlled setup. The next stronger version should run the same ablation over real coding-hidden optimization with fixed external-editor outputs or repeated seeds.
