# SkillOpt Meta Skill

Use this guidance when editing a skill document.

- Treat the skill as a compact policy, not a benchmark answer sheet.
- Introduce a rule only when supplied trajectories, verifier feedback, or longitudinal state support it. Do not preload likely benchmark answers.
- Compare failures with successes and identify the smallest recurring procedural cause that distinguishes them.
- Preserve instructions associated with stable successes unless direct evidence shows a regression.
- Treat single-example explanations as hypotheses. Prefer patterns repeated across tasks or epochs.
- Use rejected edits as negative evidence: change the direction or directly resolve the recorded reason before trying it again.
- When rejected edits include contract deltas, target the regressed or unimproved contracts first; do not repeat broad "audit the full contract" advice unless the deltas fail to isolate a smaller blocker.
- Keep updates small and auditable. Add, replace, or remove only what the evidence requires.
- Generalize across task families. Do not mention task IDs, fixture names, hidden tests, or exact expected outputs.
- Express retained lessons as executable process guidance: what evidence to inspect, what invariant to preserve, and how the target agent should verify its work.
- Distinguish agent execution anomalies from skill gaps. Empty diffs, failed public tests, permission errors, timeouts, or malformed agent output should trigger retry/evaluation handling, not skill changes.
- When agents disagree, retain only conclusions supported by scorer evidence; majority agreement is not a substitute for root-cause analysis.
