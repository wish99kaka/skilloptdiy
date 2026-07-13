# Coding-Hidden Proposal-Log Ablation

Date: 2026-06-15

## Goal

Move one step closer to paper-style multi-seed ablation by separating proposal generation from proposal evaluation.

Instead of calling an external editor during every ablation run, this experiment replays fixed proposal logs. A future real external editor run can write the same JSONL shape, then the optimizer can replay the exact same proposals under different mechanism settings.

Artifact:

- `runs/coding-hidden-proposal-log-ablation-v1`

Proposal log:

- `runs/coding-hidden-proposal-log-ablation-v1/sample_proposals.jsonl`

## Results

| Case | Seeds | Mean Valid | Mean Holdout | Successes | Mean First Success Epoch |
|---|---:|---:|---:|---:|---:|
| Gate only | 2 | 0.00 | 0.00 | 0 | - |
| +LR | 2 | 0.00 | 0.00 | 0 | - |
| +LR+Rejected Buffer | 2 | 1.00 | 1.00 | 2 | 2.0 |
| +LR+Rejected Buffer+Meta Skill | 2 | 1.00 | 1.00 | 2 | 1.0 |

## Proposal Log Shape

Each JSONL line records proposals for one seed, case, and epoch:

```json
{
  "seed": "sample-a",
  "case": "gate_lr_rejected",
  "epoch": 2,
  "proposals": [
    {
      "name": "proposal-log-rejected-buffer-guided-compact-edit-epoch-2",
      "skill_text": "...",
      "rationale": "..."
    }
  ]
}
```

Real command-backed editors can now write this shape directly:

```bash
python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding-hidden/skill.md \
  --train examples/coding-hidden/train.jsonl \
  --valid examples/coding-hidden/valid.jsonl \
  --epochs 2 \
  --editor-command "python3 examples/coding/openai_compatible_skill_editor.py" \
  --proposal-log-out runs/proposal-logs/external.jsonl \
  --proposal-log-seed seed-a \
  --proposal-log-case gate_lr_rejected \
  --out runs/coding-hidden-external-editor-seed-a-gate-lr-rejected
```

## Reproduction

```bash
python3 work/run_coding_hidden_proposal_log_ablation.py \
  --out runs/coding-hidden-proposal-log-ablation-v1 \
  --seeds sample-a,sample-b \
  --timeout-seconds 30
```

To generate a log without running:

```bash
python3 work/run_coding_hidden_proposal_log_ablation.py \
  --write-sample-log /tmp/sample_proposals.jsonl \
  --seeds sample-a,sample-b
```

## Limit

The sample proposal log is still scripted. The new value is the replay protocol: once a real external editor writes proposal logs in this shape, the same runner can produce deterministic, auditable ablations without re-querying the model.
