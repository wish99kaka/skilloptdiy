# Coding-Hidden Mechanism Ablation

Date: 2026-06-15

## Goal

Measure whether the minimal SkillOpt stability mechanisms matter on the real `coding-hidden` task suite, while keeping external-model randomness out of the optimization loop.

This experiment uses:

- real `examples/coding-hidden/train.jsonl`
- real `examples/coding-hidden/valid.jsonl`
- real `examples/coding-hidden/holdout.jsonl`
- real `CodingRunner` and hidden-test scorer
- deterministic local coding agent: `work/coding_hidden_deterministic_agent.py`
- fixed scripted skill editor: `work/run_coding_hidden_ablation.py`

Artifact:

- `runs/coding-hidden-ablation-v1`

## Results

| Case | Valid | Holdout | First Success Epoch | LR Rejections | Validated Rejections | Rejected Total |
|---|---:|---:|---:|---:|---:|---:|
| Gate only | 0.00 | 0.00 | - | 0 | 2 | 2 |
| +LR | 0.00 | 0.00 | - | 2 | 0 | 2 |
| +LR+Rejected Buffer | 1.00 | 1.00 | 2 | 1 | 0 | 1 |
| +LR+Rejected Buffer+Meta Skill | 1.00 | 1.00 | 1 | 0 | 0 | 0 |

## Interpretation

- Validation Gate blocks train-only overfitting, but by itself does not make the editor change strategy.
- Textual LR blocks the oversized train-only skill before validation, reducing wasted validation attempts.
- Rejected Buffer changes the editor's second epoch behavior after the LR rejection, producing the compact transferable skill.
- Meta Skill makes the editor produce the compact transferable skill in epoch 1.

## Limit

This is stronger than the synthetic controlled ablation because it uses real coding-hidden repos and hidden tests. It is still not a paper-grade ablation because the editor is scripted. The next stronger step is repeated runs with a real external editor under fixed seeds or fixed proposal logs.
