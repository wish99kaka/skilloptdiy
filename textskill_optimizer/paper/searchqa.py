"""Pinned SearchQA data and scoring contract for paper-faithful runs."""

from __future__ import annotations

import json
import hashlib
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen


SEARCHQA_DATASET_REPO = "lucadiliello/searchqa"
SEARCHQA_DATASET_REVISION = "c1a979068ba118d85467179b704031d113d689cc"
SEARCHQA_SOURCE_ID_FIELD = "key"
SEARCHQA_DATASET_SERVER_ENDPOINT = "https://datasets-server.huggingface.co/filter"
SEARCHQA_HUB_REVISION_ENDPOINT = (
    "https://huggingface.co/api/datasets/lucadiliello/searchqa/revision/main"
)
SEARCHQA_SOURCE_SPLITS = ("train", "validation")
OFFICIAL_SEARCHQA_SPLIT_COUNTS = {"train": 400, "selection": 200, "test": 1400}
OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256 = {
    "train": "3ae7532c2d11d9ec63ed20e3a4e425db5f6fa51a618622be848d88aa63187f0d",
    "selection": "2fccbfc16f858a1f2c9b16551c5047b2f4468dff2537f782059ac7355db37e8e",
    "test": "b6056005ed8ffa03239a0295644d83968513bed80235927716fca14926ff11bc",
}
OFFICIAL_SEARCHQA_DEVELOPMENT_OUTPUT_SHA256 = {
    "train": "807e3cee5e40652839df666b1e146420342036bcb36074f790487502480ccf67",
    "selection": "542f70f5e890a9da3c18a7622062e33d4e51e070885fca90f16e7210e902f8ac",
}
OFFICIAL_SEARCHQA_DEVELOPMENT_OUTPUT_SHA256_BY_SCHEMA = {
    "searchqa-development-materialization-v2": (
        OFFICIAL_SEARCHQA_DEVELOPMENT_OUTPUT_SHA256
    ),
    "searchqa-development-materialization-v3": {
        "train": "807e3cee5e40652839df666b1e146420342036bcb36074f790487502480ccf67",
        "selection": "1282918538c2d23cc77ffe6764bcfad965c0ca34a8e9a5b18fa45ac905ebc927",
    },
}


class SearchQAContractViolation(ValueError):
    """Raised when SearchQA data could alter the frozen benchmark contract."""


@dataclass(frozen=True)
class SearchQAMaterialization:
    receipt_path: Path
    train_manifest_path: Path
    selection_manifest_path: Path


@dataclass(frozen=True)
class SearchQADevelopmentMaterializationPolicy:
    schema_version: str
    seed: int
    train_limit: int
    selection_limit: int

    @property
    def requested_id_count(self) -> int:
        return self.train_limit + self.selection_limit


_SEARCHQA_DEVELOPMENT_MATERIALIZATION_POLICIES = (
    SearchQADevelopmentMaterializationPolicy(
        schema_version="searchqa-development-materialization-v2",
        seed=42,
        train_limit=40,
        selection_limit=5,
    ),
    SearchQADevelopmentMaterializationPolicy(
        schema_version="searchqa-development-materialization-v3",
        seed=42,
        train_limit=40,
        selection_limit=20,
    ),
)


@dataclass(frozen=True)
class SearchQASourceFetch:
    rows: tuple[Mapping[str, Any], ...]
    source_main_revision: str
    queried_splits: tuple[str, ...]


@dataclass(frozen=True)
class SearchQAItem:
    item_id: str
    question: str
    context: str
    answers: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SearchQAItem":
        expected = {"id", "question", "context", "answers"}
        if type(payload) is not dict or set(payload) != expected:
            raise SearchQAContractViolation(
                "SearchQA item must contain exactly id, question, context, answers"
            )
        item_id = payload["id"]
        question = payload["question"]
        context = payload["context"]
        answers = payload["answers"]
        if type(item_id) is not str or not item_id.strip():
            raise SearchQAContractViolation("SearchQA item id must be non-empty")
        if type(question) is not str or not question.strip():
            raise SearchQAContractViolation("SearchQA question must be non-empty")
        if type(context) is not str or not context.strip():
            raise SearchQAContractViolation("SearchQA context must be non-empty")
        if (
            type(answers) is not list
            or not answers
            or any(type(answer) is not str or not answer.strip() for answer in answers)
        ):
            raise SearchQAContractViolation(
                "SearchQA answers must be a non-empty string list"
            )
        return cls(
            item_id=item_id,
            question=question,
            context=context,
            answers=tuple(answers),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "question": self.question,
            "context": self.context,
            "answers": list(self.answers),
        }


@dataclass(frozen=True)
class SearchQAScore:
    exact_match: float
    predicted_answer: str


def load_searchqa_items(path: str | Path) -> tuple[SearchQAItem, ...]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SearchQAContractViolation(
            f"SearchQA split must be readable strict JSON: {source}"
        ) from error
    if type(payload) is not list or not payload:
        raise SearchQAContractViolation("SearchQA split must be a non-empty JSON array")
    items = tuple(SearchQAItem.from_mapping(item) for item in payload)
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise SearchQAContractViolation("SearchQA split contains duplicate item ids")
    return items


def get_searchqa_development_materialization_policy(
    *,
    train_limit: int,
    selection_limit: int,
    seed: int,
) -> SearchQADevelopmentMaterializationPolicy:
    for policy in _SEARCHQA_DEVELOPMENT_MATERIALIZATION_POLICIES:
        if (
            policy.train_limit == train_limit
            and policy.selection_limit == selection_limit
            and policy.seed == seed
        ):
            return policy
    raise SearchQAContractViolation(
        "unsupported SearchQA development materialization sample"
    )


def _searchqa_development_materialization_policy_for_schema(
    schema_version: object,
) -> SearchQADevelopmentMaterializationPolicy:
    for policy in _SEARCHQA_DEVELOPMENT_MATERIALIZATION_POLICIES:
        if policy.schema_version == schema_version:
            return policy
    raise SearchQAContractViolation("SearchQA materialization identity drift")


def normalize_searchqa_answer(value: str) -> str:
    """Apply the official v0.2.0 SearchQA SQuAD-style normalization."""

    if type(value) is not str:
        raise SearchQAContractViolation("SearchQA answer must be a string")
    normalized = value.lower()
    normalized = "".join(
        character for character in normalized if character not in string.punctuation
    )
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split()).strip()


def extract_searchqa_answer(response: str) -> str:
    if type(response) is not str:
        raise SearchQAContractViolation("SearchQA response must be a string")
    matches = re.findall(
        r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE
    )
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return lines[-1] if lines else response.strip()


def score_searchqa_response(
    response: str,
    gold_answers: Sequence[str],
) -> SearchQAScore:
    if (
        not isinstance(gold_answers, Sequence)
        or isinstance(gold_answers, (str, bytes))
        or not gold_answers
        or any(type(answer) is not str or not answer.strip() for answer in gold_answers)
    ):
        raise SearchQAContractViolation("SearchQA gold answers must be non-empty strings")
    predicted = extract_searchqa_answer(response)
    normalized = normalize_searchqa_answer(predicted)
    exact_match = float(
        any(normalize_searchqa_answer(answer) == normalized for answer in gold_answers)
    )
    return SearchQAScore(exact_match=exact_match, predicted_answer=predicted)


def select_searchqa_development_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    train_ids: Sequence[str],
    selection_ids: Sequence[str],
    train_limit: int,
    selection_limit: int,
    seed: int,
) -> dict[str, tuple[SearchQAItem, ...]]:
    """Materialize only preregistered development IDs; test IDs are not accepted."""

    if type(seed) is not int:
        raise SearchQAContractViolation("SearchQA development seed must be an integer")
    sampled = sample_searchqa_development_ids(
        train_ids=train_ids,
        selection_ids=selection_ids,
        train_limit=train_limit,
        selection_limit=selection_limit,
        seed=seed,
    )
    train_wanted = sampled["train"]
    selection_wanted = sampled["selection"]
    if set(train_wanted) & set(selection_wanted):
        raise SearchQAContractViolation(
            "SearchQA development train and selection ids must be disjoint"
        )
    wanted = set(train_wanted) | set(selection_wanted)
    selected: dict[str, SearchQAItem] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SearchQAContractViolation("SearchQA source rows must be mappings")
        source_id = row.get(SEARCHQA_SOURCE_ID_FIELD)
        if str(source_id) not in wanted:
            continue
        item_id = str(source_id)
        if item_id in selected:
            raise SearchQAContractViolation(
                f"SearchQA source contains duplicate selected id: {item_id}"
            )
        missing = [
            field for field in ("question", "context", "answers") if field not in row
        ]
        if missing:
            raise SearchQAContractViolation(
                f"SearchQA selected source row {item_id!r} is missing: {', '.join(missing)}"
            )
        selected[item_id] = SearchQAItem.from_mapping(
            {
                "id": item_id,
                "question": row["question"],
                "context": row["context"],
                "answers": row["answers"],
            }
        )
    missing = wanted - set(selected)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise SearchQAContractViolation(
            f"SearchQA source is missing {len(missing)} selected ids: {preview}"
        )
    return {
        "train": tuple(selected[item_id] for item_id in train_wanted),
        "selection": tuple(selected[item_id] for item_id in selection_wanted),
    }


def sample_searchqa_development_ids(
    *,
    train_ids: Sequence[str],
    selection_ids: Sequence[str],
    train_limit: int,
    selection_limit: int,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    if type(seed) is not int:
        raise SearchQAContractViolation("SearchQA development seed must be an integer")
    train_wanted = _sample_ids(train_ids, train_limit, seed=seed)
    selection_wanted = _sample_ids(selection_ids, selection_limit, seed=seed + 1)
    if set(train_wanted) & set(selection_wanted):
        raise SearchQAContractViolation(
            "SearchQA development train and selection ids must be disjoint"
        )
    return {"train": train_wanted, "selection": selection_wanted}


def fetch_searchqa_rows_by_id(
    item_ids: Sequence[str],
    *,
    timeout_seconds: float = 60.0,
) -> SearchQASourceFetch:
    """Fetch only requested payload rows through the server-side key filter."""

    if (
        isinstance(item_ids, (str, bytes))
        or not isinstance(item_ids, Sequence)
        or not item_ids
        or len(item_ids) > 100
        or len(item_ids) != len(set(item_ids))
        or any(re.fullmatch(r"[0-9a-f]{32}", item_id) is None for item_id in item_ids)
    ):
        raise SearchQAContractViolation(
            "SearchQA filtered fetch requires 1 to 100 unique lowercase IDs"
        )
    revision_before = _fetch_json(
        SEARCHQA_HUB_REVISION_ENDPOINT, timeout_seconds=timeout_seconds
    )
    if revision_before.get("sha") != SEARCHQA_DATASET_REVISION:
        raise SearchQAContractViolation(
            "SearchQA dataset main revision drifted from the pinned source"
        )
    where = " OR ".join(
        f'"{SEARCHQA_SOURCE_ID_FIELD}"=\'{item_id}\'' for item_id in item_ids
    )
    selected: dict[str, Mapping[str, Any]] = {}
    for source_split in SEARCHQA_SOURCE_SPLITS:
        query = urlencode(
            {
                "dataset": SEARCHQA_DATASET_REPO,
                "config": "default",
                "split": source_split,
                "where": where,
                "length": len(item_ids),
            }
        )
        payload = _fetch_json(
            f"{SEARCHQA_DATASET_SERVER_ENDPOINT}?{query}",
            timeout_seconds=timeout_seconds,
        )
        if (
            type(payload.get("rows")) is not list
            or payload.get("partial") is not False
            or payload.get("num_rows_total") != len(payload["rows"])
        ):
            raise SearchQAContractViolation(
                "SearchQA dataset server returned a partial filtered response"
            )
        for wrapper in payload["rows"]:
            if (
                type(wrapper) is not dict
                or type(wrapper.get("row")) is not dict
                or wrapper.get("truncated_cells") != []
            ):
                raise SearchQAContractViolation(
                    "SearchQA dataset server returned a truncated or invalid row"
                )
            row = wrapper["row"]
            item_id = row.get(SEARCHQA_SOURCE_ID_FIELD)
            if item_id not in item_ids or item_id in selected:
                raise SearchQAContractViolation(
                    "SearchQA dataset server returned an unexpected or duplicate ID"
                )
            selected[item_id] = row
    revision_after = _fetch_json(
        SEARCHQA_HUB_REVISION_ENDPOINT, timeout_seconds=timeout_seconds
    )
    if revision_after.get("sha") != SEARCHQA_DATASET_REVISION:
        raise SearchQAContractViolation(
            "SearchQA dataset revision changed during filtered materialization"
        )
    missing = set(item_ids) - set(selected)
    if missing:
        raise SearchQAContractViolation(
            f"SearchQA filtered source is missing {len(missing)} requested IDs"
        )
    return SearchQASourceFetch(
        rows=tuple(selected[item_id] for item_id in item_ids),
        source_main_revision=SEARCHQA_DATASET_REVISION,
        queried_splits=SEARCHQA_SOURCE_SPLITS,
    )


def verify_searchqa_materialization_receipt(
    receipt_path: str | Path,
    *,
    train_path: str | Path,
    selection_path: str | Path,
) -> SearchQAMaterialization:
    source = Path(receipt_path).resolve()
    try:
        receipt = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SearchQAContractViolation(
            "SearchQA materialization receipt must be readable JSON"
        ) from error
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "source_repo",
            "source_revision",
            "source_access",
            "official_manifest_sha256",
            "manifest_files",
            "sample",
            "counts",
            "outputs",
            "test_payload_status",
        },
        context="SearchQA materialization receipt",
    )
    policy = _searchqa_development_materialization_policy_for_schema(
        receipt["schema_version"]
    )
    if (
        receipt["source_repo"] != SEARCHQA_DATASET_REPO
        or receipt["source_revision"] != SEARCHQA_DATASET_REVISION
        or receipt["test_payload_status"] != "not_materialized"
    ):
        raise SearchQAContractViolation("SearchQA materialization identity drift")
    _require_exact_keys(
        receipt["source_access"],
        {
            "method",
            "endpoint",
            "source_main_revision",
            "queried_splits",
            "requested_id_count",
            "received_id_count",
        },
        context="SearchQA source access",
    )
    access = receipt["source_access"]
    if (
        access["method"] != "hf_dataset_server_filter_v1"
        or access["endpoint"] != SEARCHQA_DATASET_SERVER_ENDPOINT
        or access["source_main_revision"] != SEARCHQA_DATASET_REVISION
        or access["queried_splits"] != list(SEARCHQA_SOURCE_SPLITS)
        or access["requested_id_count"] != policy.requested_id_count
        or access["received_id_count"] != policy.requested_id_count
    ):
        raise SearchQAContractViolation("SearchQA source access was not payload-isolated")
    if receipt["official_manifest_sha256"] != {
        "train": OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["train"],
        "selection": OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["selection"],
        "test_commitment": OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["test"],
    }:
        raise SearchQAContractViolation("official SearchQA manifest identity drift")
    if receipt["sample"] != {
        "seed": policy.seed,
        "train_limit": policy.train_limit,
        "selection_limit": policy.selection_limit,
    } or receipt["counts"] != {
        "train": policy.train_limit,
        "selection": policy.selection_limit,
    }:
        raise SearchQAContractViolation("SearchQA development sample drift")
    manifest_paths = _verify_manifest_files(source.parent, receipt["manifest_files"])
    actual_train = Path(train_path).resolve()
    actual_selection = Path(selection_path).resolve()
    _verify_materialized_output(
        source.parent,
        receipt["outputs"],
        "train",
        actual_train,
        expected_count=policy.train_limit,
        expected_sha256=OFFICIAL_SEARCHQA_DEVELOPMENT_OUTPUT_SHA256_BY_SCHEMA[
            policy.schema_version
        ]["train"],
    )
    _verify_materialized_output(
        source.parent,
        receipt["outputs"],
        "selection",
        actual_selection,
        expected_count=policy.selection_limit,
        expected_sha256=OFFICIAL_SEARCHQA_DEVELOPMENT_OUTPUT_SHA256_BY_SCHEMA[
            policy.schema_version
        ]["selection"],
    )
    train_ids = _load_manifest_ids(manifest_paths["train"])
    selection_ids = _load_manifest_ids(manifest_paths["selection"])
    sampled = sample_searchqa_development_ids(
        train_ids=train_ids,
        selection_ids=selection_ids,
        train_limit=policy.train_limit,
        selection_limit=policy.selection_limit,
        seed=policy.seed,
    )
    if tuple(item.item_id for item in load_searchqa_items(actual_train)) != sampled[
        "train"
    ] or tuple(
        item.item_id for item in load_searchqa_items(actual_selection)
    ) != sampled["selection"]:
        raise SearchQAContractViolation(
            "SearchQA materialized rows do not match the official sampled IDs"
        )
    return SearchQAMaterialization(
        receipt_path=source,
        train_manifest_path=manifest_paths["train"],
        selection_manifest_path=manifest_paths["selection"],
    )


def _sample_ids(ids: Sequence[str], limit: int, *, seed: int) -> tuple[str, ...]:
    if (
        isinstance(ids, (str, bytes))
        or not isinstance(ids, Sequence)
        or not ids
        or any(type(item_id) is not str or not item_id.strip() for item_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise SearchQAContractViolation("SearchQA manifest ids must be unique strings")
    if type(limit) is not int or not 1 <= limit <= len(ids):
        raise SearchQAContractViolation(
            "SearchQA development limit must fit the manifest"
        )
    if limit == len(ids):
        return tuple(ids)
    return tuple(random.Random(seed).sample(list(ids), limit))


def _fetch_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise SearchQAContractViolation(
            "SearchQA filtered source request failed"
        ) from error
    if type(payload) is not dict:
        raise SearchQAContractViolation("SearchQA source response must be an object")
    return payload


def _require_exact_keys(payload: object, expected: set[str], *, context: str) -> None:
    if type(payload) is not dict or set(payload) != expected:
        raise SearchQAContractViolation(f"{context} fields do not match the contract")


def _resolve_receipt_path(root: Path, raw: object) -> Path:
    if type(raw) is not str or not raw.strip():
        raise SearchQAContractViolation("SearchQA receipt path must be non-empty")
    path = Path(raw)
    return (path if path.is_absolute() else root / path).resolve()


def _load_manifest_ids(path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SearchQAContractViolation("official SearchQA manifest is invalid") from error
    if (
        type(payload) is not list
        or not payload
        or any(type(item) is not dict or set(item) != {"id"} for item in payload)
    ):
        raise SearchQAContractViolation("official SearchQA manifest has invalid fields")
    ids = tuple(item["id"] for item in payload)
    if any(type(item_id) is not str or not item_id for item_id in ids) or len(
        ids
    ) != len(set(ids)):
        raise SearchQAContractViolation("official SearchQA manifest IDs are invalid")
    return ids


def _verify_manifest_files(root: Path, payload: object) -> dict[str, Path]:
    _require_exact_keys(payload, {"train", "selection"}, context="manifest files")
    result: dict[str, Path] = {}
    for role in ("train", "selection"):
        item = payload[role]
        _require_exact_keys(item, {"path", "sha256"}, context=f"{role} manifest")
        expected = OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256[role]
        path = _resolve_receipt_path(root, item["path"])
        if item["sha256"] != expected or _sha256(path) != expected:
            raise SearchQAContractViolation(f"official SearchQA {role} manifest drift")
        result[role] = path
    return result


def _verify_materialized_output(
    root: Path,
    payload: object,
    role: str,
    actual_path: Path,
    *,
    expected_count: int,
    expected_sha256: str,
) -> None:
    _require_exact_keys(payload, {"train", "selection"}, context="outputs")
    item = payload[role]
    _require_exact_keys(item, {"path", "sha256"}, context=f"{role} output")
    receipt_path = _resolve_receipt_path(root, item["path"])
    if (
        receipt_path != actual_path
        or item["sha256"] != expected_sha256
        or item["sha256"] != _sha256(actual_path)
    ):
        raise SearchQAContractViolation(f"SearchQA {role} output hash drift")
    if len(load_searchqa_items(actual_path)) != expected_count:
        raise SearchQAContractViolation(f"SearchQA {role} output count drift")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SearchQAContractViolation(f"SearchQA artifact is missing: {path}") from error
