"""Small validator for the JSON-Schema subset used by paper contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SchemaViolation:
    path: str
    message: str


def validate_schema(instance: Any, schema: Mapping[str, Any]) -> tuple[SchemaViolation, ...]:
    violations: list[SchemaViolation] = []
    _validate(instance, schema, "$", violations)
    return tuple(violations)


def _validate(
    instance: Any,
    schema: Mapping[str, Any],
    path: str,
    violations: list[SchemaViolation],
) -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(instance, expected_type):
        violations.append(
            SchemaViolation(path=path, message=f"expected {expected_type}, got {type(instance).__name__}")
        )
        return

    if "enum" in schema and instance not in schema["enum"]:
        violations.append(
            SchemaViolation(path=path, message=f"must be one of {schema['enum']!r}")
        )

    if isinstance(instance, Mapping):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in instance:
                violations.append(
                    SchemaViolation(path=f"{path}.{required}", message="required field is missing")
                )
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    violations.append(
                        SchemaViolation(path=f"{path}.{key}", message="unknown field")
                    )
        for key, child_schema in properties.items():
            if key in instance:
                _validate(instance[key], child_schema, f"{path}.{key}", violations)

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            violations.append(
                SchemaViolation(path=path, message=f"requires at least {schema['minItems']} items")
            )
        maximum = schema.get("maxItems")
        if maximum is not None and len(instance) > maximum:
            violations.append(
                SchemaViolation(path=path, message=f"allows at most {maximum} items")
            )
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(instance):
                _validate(item, item_schema, f"{path}[{index}]", violations)

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            violations.append(
                SchemaViolation(path=path, message=f"minimum length is {schema['minLength']}")
            )
        pattern = schema.get("pattern")
        if pattern and re.fullmatch(pattern, instance) is None:
            violations.append(
                SchemaViolation(path=path, message=f"does not match {pattern!r}")
            )

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        if minimum is not None and instance < minimum:
            violations.append(
                SchemaViolation(path=path, message=f"must be >= {minimum}")
            )
        maximum = schema.get("maximum")
        if maximum is not None and instance > maximum:
            violations.append(
                SchemaViolation(path=path, message=f"must be <= {maximum}")
            )


def _matches_type(instance: Any, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(instance, item) for item in expected)
    checks = {
        "object": lambda value: isinstance(value, Mapping),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "null": lambda value: value is None,
    }
    return checks[expected](instance)
