# Coding Agent Skill

Fix failing implementations without editing tests.

- Run public tests first, then use the failure to find the root cause in source files.
- Make the smallest complete source change that fixes the underlying behavior, not only the single public assertion.
- After public tests pass, audit the changed code against every concrete rule in this skill.
- For keyed de-duplication or unique-by-key utilities, including names like `unique_by_id`, `dedupe_by_email`, or "dedupe by <key>", check that the key exists before reading it.
- For keyed de-duplication, preserve the first record for each present key. If a record is missing the key, append that record as an independent unique record and do not group missing keys under `None` or `null`.
- For email de-duplication or tasks that mention casefold/case-insensitive matching, compare string keys using a normalized comparison key such as `value.casefold()`, while preserving the original record and original string in the output.
- For nested get/pluck/path utilities, normalize path segments by trimming whitespace and ignoring empty segments from repeated separators.
- For nested get/pluck/path utilities, resolve each path segment defensively. If the current value is a dictionary, require the segment key to exist before reading it. If the current value is a list and the segment is an integer string, treat it as a list index only after checking bounds. Getter utilities should return the requested default for a missing path. Pluck/collection utilities should skip records with a missing path unless the task explicitly asks for placeholder values.
- For sort-by-key utilities, keep sorting stable. Items missing the sort key should sort last unless the task explicitly says otherwise; do not replace a missing numeric key with `0`. In Python, a safe key shape is `(key not in item, item.get(key))` for ascending sort, because `False` sorts before `True`.
- For inclusive range utilities, normalize lower and upper bounds before generating an ascending inclusive range.
- For date range utilities, parse both start and end dates, swap them when start is after end, and then generate an ascending inclusive list of ISO date strings.
- For delimited numeric parsers such as comma-separated integer lists, trim each token, skip empty tokens from repeated separators, wrap numeric conversion in `try`/`except`, and skip malformed tokens unless the task explicitly asks to raise errors.
- For parsing, normalization, and rounding utilities, handle malformed input fallbacks, negative sign preservation, singleton inputs, and correct rounding.
