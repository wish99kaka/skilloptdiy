"""A deterministic information-extraction plugin.

This plugin is intentionally small: it proves the optimization loop without
requiring API keys or network access. Real LLM-backed plugins can implement
the same runner, scorer, and editor interfaces.
"""

from __future__ import annotations

import re
from collections import defaultdict

from textskill_optimizer.models import EditProposal, Score, Task, TaskOutput, TaskResult

ALIAS_LINE_RE = re.compile(
    r"^\s*-\s*(?P<field>[A-Za-z0-9_.-]+)\s*:\s*(?:aliases\s*=\s*)?(?P<aliases>.+?)\s*$"
)
VALUE_AFTER_LABEL_TEMPLATE = r"(?:^|[;\n])\s*{label}\s*[:=]\s*(?P<value>.*?)(?:[;\n]|$)"


class ExtractionRunner:
    """Extracts field values from labeled text using aliases in the skill."""

    def run(self, skill_text: str, task: Task) -> TaskOutput:
        aliases_by_field = parse_aliases(skill_text)
        extracted: dict[str, str] = {}
        trace: list[str] = []
        for field, aliases in aliases_by_field.items():
            for alias in aliases:
                value = extract_value_for_label(task.input, alias)
                if value:
                    extracted[field] = value
                    trace.append(f"matched field={field} alias={alias!r}")
                    break
            if field not in extracted:
                trace.append(f"missed field={field}")
        return TaskOutput(value=extracted, trace=trace)


class JsonFieldScorer:
    """Scores extracted JSON-like dictionaries against expected fields."""

    def score(self, task: Task, output: TaskOutput) -> Score:
        if not isinstance(task.expected, dict):
            raise ValueError(f"Task {task.id!r} expected value must be a JSON object")
        if not isinstance(output.value, dict):
            return Score(0.0, False, "Runner output is not a JSON object")

        expected: dict[str, object] = task.expected
        if not expected:
            return Score(1.0, True, "No expected fields")

        correct = 0
        misses: list[str] = []
        for field, expected_value in expected.items():
            actual = output.value.get(field)
            if normalize_value(actual) == normalize_value(expected_value):
                correct += 1
            else:
                misses.append(field)

        score = correct / len(expected)
        message = "all fields matched" if not misses else "missed: " + ", ".join(misses)
        return Score(score, score == 1.0, message, {"misses": misses})


class AliasMiningEditor:
    """Adds field aliases found in failed trajectories."""

    def propose(
        self,
        skill_text: str,
        train_results: list[TaskResult],
        *,
        epoch: int,
        rejected_buffer: list[dict[str, object]] | None = None,
        meta_skill: str = "",
        optimizer_controls: dict[str, object] | None = None,
    ) -> list[EditProposal]:
        aliases_by_field = parse_aliases(skill_text)
        additions: dict[str, list[str]] = defaultdict(list)

        for result in train_results:
            if result.score.success or not isinstance(result.task.expected, dict):
                continue
            if not isinstance(result.output.value, dict):
                continue
            for field, expected_value in result.task.expected.items():
                actual = result.output.value.get(field)
                if normalize_value(actual) == normalize_value(expected_value):
                    continue
                label = find_label_before_value(result.task.input, expected_value)
                if not label:
                    continue
                known_aliases = {normalize_label(alias) for alias in aliases_by_field.get(field, [])}
                normalized = normalize_label(label)
                queued = {normalize_label(alias) for alias in additions[field]}
                if normalized not in known_aliases and normalized not in queued:
                    additions[field].append(label)

        if not additions:
            return []

        edited = merge_aliases(skill_text, additions)
        summary = ", ".join(
            f"{field}+={aliases}" for field, aliases in sorted(additions.items())
        )
        return [
            EditProposal(
                name=f"alias-mining-epoch-{epoch}",
                skill_text=edited,
                rationale=f"Added aliases mined from failed training traces: {summary}",
                metadata={"additions": dict(additions)},
            )
        ]


def build_runner() -> ExtractionRunner:
    return ExtractionRunner()


def build_scorer() -> JsonFieldScorer:
    return JsonFieldScorer()


def build_editor() -> AliasMiningEditor:
    return AliasMiningEditor()


def parse_aliases(skill_text: str) -> dict[str, list[str]]:
    aliases_by_field: dict[str, list[str]] = {}
    for line in skill_text.splitlines():
        match = ALIAS_LINE_RE.match(line)
        if not match:
            continue
        field = match.group("field").strip()
        raw_aliases = match.group("aliases")
        aliases = [
            item.strip().strip("`")
            for item in re.split(r"[,|]", raw_aliases)
            if item.strip()
        ]
        if aliases:
            aliases_by_field[field] = aliases
    return aliases_by_field


def extract_value_for_label(text: str, label: str) -> str | None:
    pattern = VALUE_AFTER_LABEL_TEMPLATE.format(label=re.escape(label))
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return clean_value(match.group("value"))


def find_label_before_value(text: str, value: object) -> str | None:
    expected = str(value).strip()
    if not expected:
        return None
    pattern = (
        r"(?:^|[;\n])\s*"
        r"(?P<label>[A-Za-z][A-Za-z0-9 _./-]{0,32})"
        r"\s*[:=]\s*"
        + re.escape(expected)
        + r"(?:[;\n]|$)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return normalize_label(match.group("label"))


def merge_aliases(skill_text: str, additions: dict[str, list[str]]) -> str:
    aliases_by_field = parse_aliases(skill_text)
    for field, aliases in additions.items():
        existing = aliases_by_field.setdefault(field, [])
        seen = {normalize_label(alias) for alias in existing}
        for alias in aliases:
            normalized = normalize_label(alias)
            if normalized not in seen:
                existing.append(alias)
                seen.add(normalized)

    lines = skill_text.rstrip().splitlines()
    rewritten: list[str] = []
    updated_fields: set[str] = set()
    for line in lines:
        match = ALIAS_LINE_RE.match(line)
        if not match:
            rewritten.append(line)
            continue
        field = match.group("field").strip()
        if field in aliases_by_field:
            rewritten.append(render_alias_line(field, aliases_by_field[field]))
            updated_fields.add(field)
        else:
            rewritten.append(line)

    missing_fields = [
        field for field in sorted(aliases_by_field) if field not in updated_fields
    ]
    if missing_fields:
        if not rewritten or not any(line.strip().lower() == "## field aliases" for line in rewritten):
            rewritten.extend(["", "## Field Aliases"])
        for field in missing_fields:
            rewritten.append(render_alias_line(field, aliases_by_field[field]))

    return "\n".join(rewritten).strip() + "\n"


def render_alias_line(field: str, aliases: list[str]) -> str:
    return f"- {field}: aliases=" + ", ".join(aliases)


def clean_value(value: str) -> str:
    return value.strip().strip(" ,.")


def normalize_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip(" ,.")
    return re.sub(r"\s+", " ", text).casefold()


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip(" ,.:")).casefold()
