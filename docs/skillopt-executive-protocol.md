# SkillOpt Executive Protocol

Date: 2026-06-23

> Status: supported historical contract-aware extension. This protocol is not
> paper-faithful and cannot produce a `paper_faithful_*` claim. Never use its
> commands to access the consumed `coding-hidden-v2` locked split.

## Purpose

`--protocol executive` is the contract-aware atomic-edit optimizer path. The
legacy optimizer remains available for historical replay; the independent
paper-faithful path is defined by
`docs/specs/skillopt-paper-faithful-roadmap.md`.

## Implemented Controls

- Deterministically shuffled rollout batches for each epoch.
- Separate failure and success reflection minibatches.
- Structured `add`, `delete`, and `replace` edits with exact targets.
- Batch-level proposal de-duplication, conflict resolution, evidence-support ranking, and learning-rate clipping.
- Constant, linear, and cosine integer edit-budget schedules.
- Adaptive paired validation gate: reject immediately when the first run does not improve; otherwise run two order-balanced current/candidate confirmations.
- Configurable majority and aggregate-effect thresholds. The coding-hidden-v2 protocol requires at least `2/3` wins and mean improvement of `0.05`.
- Final development score reuses accepted gate evidence instead of adding an unregistered stochastic rerun.
- Epoch-local rejected-edit feedback.
- Candidate skill hashes to avoid repeated evaluation.
- Epoch-wise comparison of the same sampled tasks under previous and current skills.
- Optimizer-only dynamic Meta Skill updates.
- Validation-gated slow updates in a protected skill field.

Fast edits cannot modify the protected slow-update field. Meta Skill text is saved for audit but is not deployed with the target agent.

## External Editor Contract

Before target evaluation, the command receives a local capability probe:

```json
{"operation": "capabilities"}
```

An executive-compatible command must answer without calling a model:

```json
{"capabilities": ["atomic_edits"]}
```

Reflection returns atomic edits:

```json
{
  "proposals": [
    {
      "name": "evidence-based-update",
      "rationale": "Repeated trajectory evidence",
      "edits": [
        {
          "operation": "add",
          "target": "__end__",
          "content": "One concise procedural rule",
          "rationale": "Why this edit follows from evidence",
          "priority": 0.8
        }
      ]
    }
  ]
}
```

Epoch state updates return optimizer guidance and an optional slow candidate:

```json
{
  "meta_skill": "Optimizer-only longitudinal guidance",
  "slow_update": "Durable target-side guidance, or empty",
  "rationale": "Improvements, regressions, and persistent failures"
}
```

## Development Command

This command uses only train and selection. Do not add a holdout path.

```bash
TEXTSKILL_EDITOR_CMD="python3 examples/coding/openai_compatible_skill_editor.py" \
python3 -m textskill_optimizer.cli optimize \
  --protocol executive \
  --plugin coding \
  --skill examples/coding-hidden-v2/skill.md \
  --train examples/coding-hidden-v2/train.jsonl \
  --valid examples/coding-hidden-v2/selection.jsonl \
  --epochs 4 \
  --rollout-batch-size 5 \
  --reflection-minibatch-size 2 \
  --text-learning-rate 4 \
  --text-learning-rate-floor 2 \
  --text-learning-rate-schedule cosine \
  --slow-update-sample-size 10 \
  --validation-confirmation-rounds 2 \
  --validation-required-wins 2 \
  --validation-mean-delta 0.05 \
  --meta-skill work/meta_skill.md \
  --out runs/coding-hidden-v2-executive
```

Each candidate writes its initial report, paired confirmation reports, and a `selection_*_gate.json` decision record. Confirmation runs happen only after an initial improvement, limiting target-agent cost while preventing a one-run fluctuation from updating the skill.

Recovery reports must record both `source_target_model` and `recovery_target_model`. A model mismatch is labeled `cross_model_transfer_only` and cannot be treated as a same-model optimizer comparison.

The final test is encrypted at `examples/coding-hidden-v2/test.enc`. It can only be supplied to the final experiment through `textskill_optimizer.locked_eval`, which records and consumes one attempt.
