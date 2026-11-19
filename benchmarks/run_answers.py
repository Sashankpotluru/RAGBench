"""Generate all three frozen SQuAD answer runs from retained evaluation rankings."""

import argparse
import tempfile
from pathlib import Path

from ragbench.io import write_json_atomic

from .answer_inputs import persist, source_inputs
from .data import load_dataset
from .generation import CONFIGS, PROTOCOL, Generator
from .run_retrieval import runtime


def run_answers(dest: Path, retrieval: Path, cache: Path):
    if dest.exists():
        raise FileExistsError("choose a new output directory for this experiment")
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = retrieval / "squad-v1.1"
    machine = runtime("cpu")
    data = load_dataset("squad-v1.1", cache)
    identity, ranks, bodies = source_inputs(source, data)
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
