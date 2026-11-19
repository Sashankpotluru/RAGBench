"""Generate all three frozen SQuAD answer runs from retained evaluation rankings."""

import argparse
import tempfile
from pathlib import Path

from ragbench.io import read_json, read_run, write_json_atomic

from .data import load_dataset, sha256_file
from .generation import CONFIGS, PROTOCOL, Generator
from .retrieval import rank
from .run_retrieval import runtime


def run_answers(dest: Path, retrieval: Path, cache: Path):
    if dest.exists():
        raise FileExistsError("choose a new output directory for this experiment")
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = retrieval / "squad-v1.1"
    machine = runtime("cpu")
    data = load_dataset("squad-v1.1", cache)
    selection = read_json(source / "selection.json")
    if (
        list(data.eval_queries) != selection["eval_ids"]
        or data.eval_groups != selection["eval_groups"]
        or data.eval_qrels != read_json(source / "eval-qrels.json")
        or data.eval_answers != read_json(source / "eval-answers.json")
        or len(data.eval_queries) != 200
    ):
        raise ValueError("retained SQuAD evidence differs from verified dataset")
    ranks = {}
    for system, k in CONFIGS.values():
        if system:
            run_file = source / "runs" / f"{system}.trec"
            evaluation = read_json(source / "evaluations" / f"{system}.json")
            if (
                evaluation["provenance"]["run_sha256"] != sha256_file(run_file)
                or evaluation["provenance"]["selected_sha256"]
                != sha256_file(source / "selected.json")
                or evaluation["provenance"]["source_sha256"] != data.fingerprint
            ):
                raise ValueError("retrieval source or selection digest mismatch")
            scores = read_run(run_file)
            if scores.keys() != data.eval_queries.keys() or any(
                len(s) != 100 for s in scores.values()
            ):
                raise ValueError("incomplete retrieval rankings")
            ranks[system] = {q: list(rank(s, k)) for q, s in scores.items()}
    failure_path = dest.with_name(dest.name + "-failure.json")
    label = qid = None
    try:
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            stage = Path(tmp) / "answers"
            stage.mkdir()
            (stage / "predictions").mkdir()
            write_json_atomic(stage / "runtime.json", machine)
            write_json_atomic(stage / "questions.json", data.eval_queries)
            write_json_atomic(stage / "protocol.json", PROTOCOL)
            gen = Generator(cache)
            # Warm CPU paths once with authored text; never score or report this row.
            gen.predict("What color is grass?", [("warmup", "Grass is green.")])
            for label, (system, _) in CONFIGS.items():
                rows = {}
                for qid, question in data.eval_queries.items():
                    passages = (
                        [(d, data.documents[d]["text"]) for d in ranks[system][qid]]
                        if system
                        else []
                    )
                    rows[qid] = gen.predict(question, passages)
                    if len(rows) % 50 == 0:
                        print(f"{label}: {len(rows)}/{len(data.eval_queries)}", flush=True)
                write_json_atomic(stage / "predictions" / f"{label}.json", rows)
            write_json_atomic(stage / "failures.json", [])
            stage.rename(dest)
    except Exception as exc:
        write_json_atomic(
            failure_path,
            {
                "system": label,
                "query_id": qid,
                "error": type(exc).__name__,
                "message": str(exc),
                "complete": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/answers"))
    parser.add_argument("--retrieval", type=Path, default=Path("results/retrieval"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    run_answers(args.output, args.retrieval, args.cache)
