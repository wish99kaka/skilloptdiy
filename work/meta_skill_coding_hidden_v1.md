# SkillOpt Meta Skill

Use this guidance when editing a skill document.

- Treat the skill as a compact policy, not a benchmark answer sheet.
- Prefer executable rules over abstract advice. Replace phrases like "handle malformed input" with concrete checks such as `try`/`except`, bounds checks, or normalized comparison keys.
- Preserve rules that explain successful validation behavior unless there is direct evidence they cause regressions.
- Do not repeat rejected edit directions. If an edit failed validation, make the next proposal address the failure reason explicitly or choose a different direction.
- Keep edits small. Add, replace, or remove the fewest rules needed to explain repeated failure patterns.
- Generalize across task families. Do not mention task IDs, fixture names, hidden tests, or exact expected outputs.
- For coding utility skills, prefer concrete capability rules over generic "edge cases" wording. Cover keyed de-duplication, nested path traversal, missing sort keys, and numeric rounding when failure evidence suggests identity, data access, ordering, or money/tax/cents behavior.
- For keyed de-duplication, specify that the implementation must check key existence before reading the key. Missing-key records should be appended immediately and must not add `None`/`null` to the seen set. When the domain is email or case-insensitive matching, compare a normalized key such as `value.casefold()` while preserving the original record.
- For nested path utilities, specify how to split path segments, ignore empty separators, handle dict keys, and handle list indexes with bounds checks. Be explicit that getter utilities return defaults for missing paths, while pluck/collection utilities skip records with missing paths rather than appending `None`.
- For delimited numeric parsers, specify trimming tokens, skipping empty tokens, wrapping conversion in `try`/`except`, and skipping malformed tokens unless the task explicitly asks to raise.
- For sort-by-key utilities, specify stable sorting and make missing keys sort last with a safe key shape such as `(key not in item, item.get(key))`.
- For range/date utilities, specify that reversed bounds are swapped before iteration and the output remains ascending and inclusive unless the task explicitly asks for descending output.
- For rounding utilities, specify decimal-safe rounding such as `Decimal` plus `ROUND_HALF_UP` when dealing with money, tax, or cents.
- Distinguish agent execution anomalies from skill gaps. Empty diffs, failed public tests, permission errors, timeouts, or malformed agent output should trigger retry/evaluation handling, not skill changes.
- When multiple agents disagree, prefer rules that explain majority success and the concrete minority failure without overfitting to one agent's wording.
