"""Generate all three frozen SQuAD answer runs from retained evaluation rankings."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from ragbench.io import read_json, write_json_atomic

from .answer_inputs import persist, source_inputs
from .data import load_dataset
from .generation import CONFIGS, PROTOCOL, Generator
from .run_retrieval import runtime


def archive_failure(dest: Path, record: dict, legacy: Path | None = None) -> Path:
    dest = dest.resolve()
    tree = dest.parent if dest.name == "answers" and dest.parent != dest.parent.parent else dest
    sibling = tree.with_name(tree.name + ".answer-failures")
    fallback = Path.home() / ".ragbench" / "answer-failures"
    name = f"{dest.name}-{uuid4().hex}.json"
    errors = []
    for archive in (sibling, fallback):
        try:
            if archive.resolve().is_relative_to(tree):
                continue
            archive.mkdir(mode=0o700, exist_ok=True, parents=True)
            path = archive / name
            if legacy is None:
                write_json_atomic(path, record)
            else:
                shutil.move(legacy, path)
            return path
        except OSError as exc:
            errors.append(f"{archive}: {exc}")
    raise OSError(f"no writable failure archive outside {tree}: {', '.join(errors)}")


def archive_legacy_failure(dest: Path) -> None:
    sidecar = dest.with_name(dest.name + "-failure.json")
    if not sidecar.exists() and not sidecar.is_symlink():
        return
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"expected a previous answer failure record at {sidecar}")
    record = read_json(sidecar)
    if not (
        isinstance(record, dict)
        and record.keys() == {"system", "query_id", "error", "message", "complete"}
        and record["complete"] is False
        and (
            record["system"] is None
            or isinstance(record["system"], str)
            and record["system"] in CONFIGS
        )
        and (record["query_id"] is None or isinstance(record["query_id"], str))
        and isinstance(record["error"], str)
        and isinstance(record["message"], str)
    ):
        raise ValueError(f"unrecognized answer failure record at {sidecar}")
    archived = archive_failure(dest, record, legacy=sidecar)
    print(f"Archived previous answer failure at {archived}", file=sys.stderr)


def run_answers(dest: Path, retrieval: Path, cache: Path):
    if dest.exists():
        raise FileExistsError("choose a new output directory for this experiment")
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive_legacy_failure(dest)
    source = retrieval / "squad-v1.1"
    machine = runtime("cpu")
    data = load_dataset("squad-v1.1", cache)
    identity, ranks, bodies = source_inputs(source, data)
    label = qid = None
    try:
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            stage = Path(tmp) / "answers"
            stage.mkdir()
            (stage / "predictions").mkdir()
            write_json_atomic(stage / "runtime.json", machine)
            write_json_atomic(stage / "questions.json", data.eval_queries)
            write_json_atomic(stage / "protocol.json", PROTOCOL)
            persist(
                stage,
                identity,
                bodies,
                {
                    "stage": "before_inference",
                    "measurement_source_commit": machine["source_commit"],
                    "binding_source_commit": machine["source_commit"],
                },
            )
            gen = Generator(cache)
            # Warm CPU paths once with authored text; never score or report this row.
            gen.predict("What color is grass?", [("warmup", "Grass is green.")])
            for label, (system, _) in CONFIGS.items():
                rows = {}
                for qid, question in data.eval_queries.items():
                    passages = [(d, bodies[d]) for d in ranks[system][qid]] if system else []
                    rows[qid] = gen.predict(question, passages)
                    if len(rows) % 50 == 0:
                        print(f"{label}: {len(rows)}/{len(data.eval_queries)}", flush=True)
                write_json_atomic(stage / "predictions" / f"{label}.json", rows)
            write_json_atomic(stage / "failures.json", [])
            stage.rename(dest)
    except Exception as exc:
        record = {
            "system": label,
            "query_id": qid,
            "error": type(exc).__name__,
            "message": str(exc),
            "complete": False,
        }
        try:
            failure_path = archive_failure(dest, record)
            print(f"Saved answer failure at {failure_path}", file=sys.stderr)
        except Exception as archive_error:
            print(f"Could not save answer failure: {archive_error}", file=sys.stderr)
            print(json.dumps(record), file=sys.stderr)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/answers"))
    parser.add_argument("--retrieval", type=Path, default=Path("results/retrieval"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    run_answers(args.output, args.retrieval, args.cache)
