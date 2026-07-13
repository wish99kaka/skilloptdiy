# coding-hidden-v2

This benchmark has ten capability families. Development exposes one train and one selection task per family. Two additional variants per family are stored only in `test.enc`.

Rules:

- Optimize only with `train.jsonl`.
- Accept or reject edits only with `selection.jsonl`.
- Do not decrypt or evaluate `test.enc` during development.
- Run harness health checks on development tasks before final evaluation.
- Final evaluation must use `python3 -m textskill_optimizer.locked_eval run` and writes a one-attempt receipt even when the child command fails.
- Report task accuracy, family macro accuracy, and contract macro accuracy.

The key file is intentionally outside the repository. The lock prevents accidental evaluation and creates an auditable commitment; it does not defend against a malicious workspace owner.
