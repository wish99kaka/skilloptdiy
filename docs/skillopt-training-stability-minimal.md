# SkillOpt Training Stability Minimal Mechanisms

Date: 2026-06-15

This project now has a minimal implementation of three SkillOpt stability mechanisms:

- Rejected Buffer
- Textual Learning Rate
- Meta Skill

## Rejected Buffer

Rejected proposals are retained as optimizer feedback instead of being discarded.

Artifacts:

- `rejected_buffer.jsonl`
- `result.json.rejected_buffer`
- `history.json.rejected_buffer`

Rejected records include:

- `epoch`
- `candidate`
- `reason`
- `rationale`
- `validation_score`
- `failed_task_ids`
- `metadata.learning_rate`

Current rejection reasons:

- `learning_rate_exceeded`
- `validation_not_improved`
- `lower_than_best_candidate`

The next editor call receives the latest rejected records through the command-editor JSON payload as `rejected_buffer`.

## Textual Learning Rate

The optimizer now applies a small edit budget before validation:

- `max_skill_chars`
- `max_skill_delta_chars`
- `max_added_bullet_lines`

Candidates that exceed the budget are rejected before validation and added to the rejected buffer.

CLI controls:

```bash
--max-skill-chars 6000
--max-skill-delta-chars 1800
--max-added-bullet-lines 8
--rejected-buffer-limit 20
```

Environment equivalents:

```bash
TEXTSKILL_MAX_SKILL_CHARS
TEXTSKILL_MAX_SKILL_DELTA_CHARS
TEXTSKILL_MAX_ADDED_BULLET_LINES
TEXTSKILL_REJECTED_BUFFER_LIMIT
```

This is a replacement-skill budget, not yet a full patch-edit protocol.

## Meta Skill

The optimizer can load an optimizer-side meta skill and pass it to external editors.

Default project artifact:

- `work/meta_skill.md`

CLI:

```bash
--meta-skill work/meta_skill.md
```

Environment:

```bash
TEXTSKILL_META_SKILL=work/meta_skill.md
```

The meta skill is guidance for the skill editor, not instructions for the coding agent under evaluation.

## Command Editor Payload

External editors now receive:

```json
{
  "epoch": 1,
  "skill_text": "...",
  "train_results": [],
  "rejected_buffer": [],
  "meta_skill": "...",
  "optimizer_controls": {
    "max_skill_chars": 6000,
    "max_skill_delta_chars": 1800,
    "max_added_bullet_lines": 8
  }
}
```

## Remaining Gaps

This is intentionally not the full paper mechanism yet.

Still missing:

- structured add/delete/replace patch edits
- semantic edit merging/ranking
- slow update memory across optimization runs
- richer retry/health anomaly taxonomy and aggregate stability metrics
