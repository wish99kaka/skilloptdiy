# Paper Archive

Local copies of papers used as design references.

## SkillOpt

- Local PDF: `docs/papers/skillopt-2605.23904.pdf`
- Clean Markdown: `docs/papers/paper-notes.md`
- Source lock: `docs/papers/source-lock.json`
- Normative paper: arXiv `2605.23904v2`
- Official code reference: Microsoft SkillOpt `v0.2.0`, commit
  `e4ea6a6771e797ef820cdd8bfea64c57e0481065`
- Title: SkillOpt: Executive Strategy for Self-Evolving Agent Skills
- Use: primary reference for the project protocol, optimizer mechanism, and reporting gap analysis.

When reviewing the project against the paper, prefer `paper-notes.md` for fast text search and agent reading. Use the local PDF as the canonical source for pagination, figures, tables, and citation checks. Use the network only to verify a newer revision or citation metadata.

## Reproduce the locked references

Verify the tracked paper:

```bash
shasum -a 256 docs/papers/skillopt-2605.23904.pdf
```

Expected SHA256:

```text
87f7f0f323b1671e9202b3ebb1596e909e507c71ecd1b360b0075a5ee1727fe3
```

Reproduce the official reference in a separate directory:

```bash
git clone --depth 1 --branch v0.2.0 \
  https://github.com/microsoft/SkillOpt.git /tmp/skillopt-v0.2.0
git -C /tmp/skillopt-v0.2.0 rev-parse HEAD 'HEAD^{tree}' refs/tags/v0.2.0
```

Expected identities, in order:

```text
e4ea6a6771e797ef820cdd8bfea64c57e0481065
5a603e937a20f1078059f94039a50028c022487a
51d0a4d96e88558c84dee637f98e24e3fb2d1547
```

`source-lock.json` records the sixteen byte-identical v0.2.0 prompts reused by
the paper fast and epoch loops. Add a path and SHA256 there before porting any
additional official source or prompt. A newer release requires an explicit
re-pin; never use upstream `main` as an implicit source.

`prompt-snapshot-v1.json` also records the two explicit local refinement
resolutions. Run the complete zero-external-call acceptance gate with:

```bash
python3 scripts/run_paper_zero_cost_gate.py
```

The command scrubs API credentials, runs only `tests/conformance` and
`tests/provenance`, and emits a machine-readable receipt only after the locked
sources, golden trace, static firewalls, replay, lineage, and claim checks pass.
Authorization also requires a clean Git commit so the receipt identifies the
exact code under test. `--audit-only` validates provenance but never authorizes
paid development.
