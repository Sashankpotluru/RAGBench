"""Construction and validation of versioned evaluation artifacts."""

import hashlib
import json
import math
import re
from typing import Any

from .io import finite_number, identifier, validate_json
from .metrics import ANSWER_METRICS, RETRIEVAL_METRICS

SCORING_VERSION = "ragbench-v1"
_COLUMNS = {"answers": ANSWER_METRICS, "retrieval": RETRIEVAL_METRICS}
_FIELDS = {
    "kind",
    "dataset_id",
    "split_id",
    "reference_sha256",
    "query_ids",
    "means",
    "schema_version",
    "scoring_version",
    "per_query",
    "groups",
    "config",
    "provenance",
}


def reference_digest(value: Any) -> str:
    validate_json(value)
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unit_score(value: Any, label: str) -> float:
    number = finite_number(value, label)
    if not 0 <= number <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _columns(row: Any, columns: tuple[str, ...], label: str) -> dict[str, float]:
    if not isinstance(row, dict) or set(row) != set(columns):
        raise ValueError(f"incorrect metric columns in {label}")
    return {metric: _unit_score(row[metric], metric) for metric in columns}


def make_evaluation(
    *,
    kind: str,
    dataset_id: str,
    split_id: str,
    reference_sha256: str,
    per_query: dict[str, dict[str, float]],
    groups: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(kind, str) or kind not in _COLUMNS:
        raise ValueError("kind must be retrieval or answers")
    identifier(dataset_id, "dataset_id")
    identifier(split_id, "split_id")
    if not isinstance(reference_sha256, str) or not re.fullmatch("[0-9a-f]{64}", reference_sha256):
        raise ValueError("reference_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(per_query, dict) or not per_query:
        raise ValueError("per_query must be a nonempty query object")
    qids = sorted(identifier(qid, "query ID") for qid in per_query)
    columns = _COLUMNS[kind]
    rows = {qid: _columns(per_query[qid], columns, qid) for qid in qids}
    if groups is not None:
        if not isinstance(groups, dict) or set(groups) != set(qids):
            raise ValueError("groups must cover exactly the evaluated queries")
        groups = {qid: identifier(groups[qid], "group ID") for qid in qids}
    for value in (config, provenance):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("config and provenance must be JSON objects")
            validate_json(value)
    return {
        "schema_version": 1,
        "scoring_version": SCORING_VERSION,
        "kind": kind,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "reference_sha256": reference_sha256,
        "query_ids": qids,
        "per_query": rows,
        "means": {m: math.fsum(rows[q][m] for q in qids) / len(qids) for m in columns},
        "groups": groups,
        "config": config or {},
        "provenance": provenance or {},
    }


def validate_evaluation(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise ValueError("incorrect evaluation artifact fields")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if data["scoring_version"] != SCORING_VERSION:
        raise ValueError("unsupported scoring_version")
    if data["config"] is None or data["provenance"] is None:
        raise ValueError("config and provenance must be JSON objects")
    canonical = make_evaluation(
        kind=data["kind"],
        dataset_id=data["dataset_id"],
        split_id=data["split_id"],
        reference_sha256=data["reference_sha256"],
        per_query=data["per_query"],
        groups=data["groups"],
        config=data["config"],
        provenance=data["provenance"],
    )
    if canonical["query_ids"] != data["query_ids"]:
        raise ValueError("query_ids must equal the sorted IDs in per_query")
    means = _columns(data["means"], _COLUMNS[data["kind"]], "means")
    if means != canonical["means"]:
        raise ValueError("means do not match the per-query scores")
    return canonical
