"""Strict readers for benchmark inputs and atomic JSON output."""

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

MAX_INPUT_BYTES = 64 * 1024 * 1024


def _read_text(path: Path) -> str:
    with path.open("rb") as handle:
        data = handle.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError(f"{path} exceeds the {MAX_INPUT_BYTES} byte input limit")
    return data.decode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token!r}")


def _parse_json(text: str) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        validate_json(value)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the supported depth") from exc
    return value


def validate_json(value: Any) -> None:
    """Reject lossy coercions (e.g. integer keys) and non-finite numbers."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            validate_json(item)
    elif isinstance(value, list):
        for item in value:
            validate_json(item)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
    elif value is not None and not isinstance(value, (str, int)):
        raise ValueError("expected a JSON value")


def read_json(path: Path) -> Any:
    return _parse_json(_read_text(path))


def write_json_atomic(path: Path, data: Any, *, match_existing: bool = False) -> None:
    """Publish complete bytes, optionally requiring a pre-existing file to match exactly."""
    validate_json(data)
    encoded = (json.dumps(data, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        if match_existing:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                    raise FileExistsError(f"conflicting content at {path}") from None
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} is outside the finite score range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _grade(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("qrel grade must be a nonnegative integer")
    return value


def _score(value: Any) -> float:
    score = finite_number(value, "run score")
    if isinstance(value, int) and value != score:
        raise ValueError("integer run score must be exactly representable in binary64")
    return score


def _numeric_rows[T](
    value: Any, convert: Callable[[Any], T], *, qrels: bool
) -> dict[str, dict[str, T]]:
    if not isinstance(value, dict) or (qrels and not value):
        raise ValueError("expected a query object with at least one judged query for qrels")
    result = {}
    for qid, docs in value.items():
        identifier(qid, "query ID")
        if not isinstance(docs, dict) or (qrels and not docs):
            raise ValueError(f"invalid document map for {qid!r}")
        result[qid] = {
            identifier(doc, "document ID"): convert(score) for doc, score in docs.items()
        }
    return result


def validate_qrels(value: Any) -> dict[str, dict[str, int]]:
    return _numeric_rows(value, _grade, qrels=True)


def validate_run(value: Any) -> dict[str, dict[str, float]]:
    return _numeric_rows(value, _score, qrels=False)


def _records(path: Path, *, qrels: bool) -> Any:
    text = _read_text(path)
    if path.suffix.lower() == ".json" or text.lstrip().startswith(("{", "[")):
        return _parse_json(text)

    result: dict[str, dict[str, int | float]] = {}
    lines = ((n, line.split()) for n, line in enumerate(text.splitlines(), 1) if line.strip())
    beir = False
    for index, (number, fields) in enumerate(lines):
        if index == 0 and qrels and fields == ["query-id", "corpus-id", "score"]:
            beir = True
            continue
        try:
            if beir:
                qid, doc, value = fields
            elif qrels:
                qid, _, doc, value = fields
            else:
                qid, _, doc, rank, value, _ = fields
                if int(rank) < 1:
                    raise ValueError("TREC rank must be a positive integer")
            scores = result.setdefault(qid, {})
            if doc in scores:
                raise ValueError(f"duplicate record for {qid!r}, {doc!r}")
            scores[doc] = _grade(int(value)) if qrels else _score(float(value))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    return result


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    return validate_qrels(_records(path, qrels=True))


def read_run(path: Path) -> dict[str, dict[str, float]]:
    return validate_run(_records(path, qrels=False))
