"""Deterministic coding agent for coding-hidden ablation experiments."""

from __future__ import annotations

import os
import json
from pathlib import Path


FULL_MARKER = "FULL_CODING_HIDDEN_RULES"
TRAIN_MARKER = "TRAIN_ONLY_CODING_HIDDEN_RULES"


TRAIN_FIXTURES = {
    "slug-normalizer",
    "money-parser",
    "dedupe-by-email",
    "number-range",
    "nested-default",
    "stable-sort",
    "parse-duration",
    "round-cents",
}


SOLUTIONS = {
    "slug-normalizer": (
        "app/slug.py",
        '''import re


def normalize_slug(title: str) -> str:
    text = title.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
''',
    ),
    "money-parser": (
        "app/money.py",
        '''from decimal import Decimal, ROUND_HALF_UP


def parse_money(value: str) -> int:
    text = str(value).strip().replace(",", "")
    sign = -1 if text.startswith("-") else 1
    text = text.replace("USD", "").replace("$", "").strip()
    if text.startswith("-"):
        text = text[1:].strip()
    cents = (Decimal(text) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return sign * int(cents)
''',
    ),
    "dedupe-by-email": (
        "app/users.py",
        '''def dedupe_by_email(users: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for user in users:
        if "email" not in user:
            result.append(user)
            continue
        email = user["email"]
        if email not in seen:
            seen.add(email)
            result.append(user)
    return result
''',
    ),
    "number-range": (
        "app/ranges.py",
        '''def number_range(start: int, end: int) -> list[int]:
    lower, upper = sorted((start, end))
    return list(range(lower, upper + 1))
''',
    ),
    "nested-default": (
        "app/data.py",
        '''def get_path(data: dict, path: str, default=None):
    current = data
    segments = [segment.strip() for segment in path.split(".") if segment.strip()]
    for segment in segments:
        if isinstance(current, dict):
            if segment not in current:
                return default
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
            except ValueError:
                return default
            if index < 0 or index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current
''',
    ),
    "stable-sort": (
        "app/ranking.py",
        '''def sort_by_score(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: row.get("score", 0), reverse=True)
''',
    ),
    "parse-duration": (
        "app/duration.py",
        '''def parse_duration(text: str) -> int:
    value = str(text).strip()
    if not value:
        return 0
    unit = value[-1]
    number_text = value[:-1] if unit.isalpha() else value
    try:
        number = int(number_text.strip())
    except ValueError:
        return 0
    if unit == "h":
        return number * 3600
    if unit == "m":
        return number * 60
    if unit == "s":
        return number
    return number
''',
    ),
    "round-cents": (
        "app/money.py",
        '''from decimal import Decimal, ROUND_HALF_UP


def to_cents(amount) -> int:
    cents = (Decimal(str(amount)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(cents)
''',
    ),
    "unique-by-id": (
        "app/items.py",
        '''def unique_by_id(items: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for item in items:
        if "id" not in item:
            result.append(item)
            continue
        item_id = item["id"]
        if item_id not in seen:
            seen.add(item_id)
            result.append(item)
    return result
''',
    ),
    "nested-pluck": (
        "app/pluck.py",
        '''def pluck(records: list[dict], path: str) -> list:
    segments = [segment.strip() for segment in path.split(".") if segment.strip()]
    result = []
    for record in records:
        current = record
        found = True
        for segment in segments:
            if isinstance(current, dict):
                if segment not in current:
                    found = False
                    break
                current = current[segment]
            elif isinstance(current, list):
                try:
                    index = int(segment)
                except ValueError:
                    found = False
                    break
                if index < 0 or index >= len(current):
                    found = False
                    break
                current = current[index]
            else:
                found = False
                break
        if found:
            result.append(current)
    return result
''',
    ),
    "stable-sort-events": (
        "app/events.py",
        '''def sort_events(events: list[dict]) -> list[dict]:
    return sorted(events, key=lambda event: ("ts" not in event, event.get("ts")))
''',
    ),
    "parse-int-list": (
        "app/parse.py",
        '''def parse_int_list(text: str) -> list[int]:
    result = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(token))
        except ValueError:
            continue
    return result
''',
    ),
    "date-range": (
        "app/dates.py",
        '''from datetime import date, timedelta


def date_range(start: str, end: str) -> list[str]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    values = []
    current = start_date
    while current <= end_date:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return values
''',
    ),
    "safe-nested-get": (
        "app/access.py",
        '''def safe_get(data: dict, path: str, default=None):
    current = data
    segments = [segment.strip() for segment in path.split(".") if segment.strip()]
    for segment in segments:
        if isinstance(current, dict):
            if segment not in current:
                return default
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
            except ValueError:
                return default
            if index < 0 or index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current
''',
    ),
    "round-tax": (
        "app/tax.py",
        '''from decimal import Decimal, ROUND_HALF_UP


def add_tax(cents: int, rate: float) -> int:
    total = Decimal(cents) * (Decimal("1") + Decimal(str(rate)))
    return int(total.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
''',
    ),
    "dedupe-casefold": (
        "app/users.py",
        '''def dedupe_emails(users: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for user in users:
        if "email" not in user:
            result.append(user)
            continue
        key = str(user["email"]).casefold()
        if key not in seen:
            seen.add(key)
            result.append(user)
    return result
''',
    ),
}


def main() -> int:
    repo = Path(os.environ["TEXTSKILL_REPO_DIR"])
    skill = Path(os.environ["TEXTSKILL_SKILL_PATH"]).read_text(encoding="utf-8")
    fixture = fixture_name_from_task(Path(os.environ["TEXTSKILL_TASK_PATH"]))
    if FULL_MARKER in skill:
        apply_solution(repo, fixture)
        return 0
    if TRAIN_MARKER in skill and fixture in TRAIN_FIXTURES:
        apply_solution(repo, fixture)
        return 0
    print(f"no deterministic rule for fixture={fixture}")
    return 0


def fixture_name_from_task(task_path: Path) -> str:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    repo = str(metadata.get("repo") or "")
    if repo:
        return Path(repo).name
    return "unknown"


def apply_solution(repo: Path, fixture: str) -> None:
    if fixture not in SOLUTIONS:
        print(f"unknown fixture={fixture}")
        return
    relative, source = SOLUTIONS[fixture]
    target = repo / relative
    target.write_text(source, encoding="utf-8")
    print(f"rewrote {relative} for {fixture}")


if __name__ == "__main__":
    raise SystemExit(main())
