"""Recompute independent reference checks from retained IDs, scores, judgments and evaluations."""

import argparse
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ragbench.evaluation import validate_evaluation
from ragbench.io import read_json, read_qrels, read_run

from .data import FILES, Qrels
from .retrieval import rank
from .select import Run

COLUMNS = {
    "ndcg@10": "ndcg_cut_10",
    "recall@5": "recall_5",
    "recall@10": "recall_10",
    "recall@100": "recall_100",
    "precision@10": "P_10",
    "mrr@10": "recip_rank",
}


def reference_rows(qrels: Qrels, run: Run, *, dev: bool) -> dict[str, dict[str, float]]:
    import pytrec_eval

    # Only the independent scorer receives surrogate scores.
    ordered = {
        q: {d: float(100 - i) for i, d in enumerate(rank(run[q], 100))}
        for q in qrels
        if q in run and run[q]
    }
    measures = {"ndcg_cut.10"} if dev else {"ndcg_cut.10", "recall.5,10,100", "P.10"}
    rows = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(ordered)
    if not dev:
        truncated = {q: dict(list(docs.items())[:10]) for q, docs in ordered.items()}
        mrr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate(truncated)
        for qid in rows:
            rows[qid].update(mrr[qid])
    fields = ["ndcg@10"] if dev else list(COLUMNS)
    return {
        q: {col: rows[q][COLUMNS[col]] if q in ordered else 0.0 for col in fields} for q in qrels
    }


def compare(
    run: Run, qrels: Qrels, rows: dict[str, dict[str, float]], means: dict[str, float], *, dev: bool
) -> dict[str, float | int]:
    if rows.keys() != qrels.keys() or run.keys() != qrels.keys():
        raise ValueError("retained run, scored queries and references must match")
    from_ref = reference_rows(qrels, run, dev=dev)
    errors = [abs(rows[q][m] - value) for q, ref in from_ref.items() for m, value in ref.items()]
    aggregate = [
        abs(means[col] - math.fsum(r[col] for r in from_ref.values()) / len(rows)) for col in means
    ]
    maximum = max(errors + aggregate)
    if maximum > 1e-12:
        raise ValueError(f"trec_eval discrepancy: maximum absolute error {maximum}")
    return {"per_query_count": len(errors), "mean_count": len(aggregate), "max_abs_error": maximum}


def crosscheck(root: Path) -> dict[str, Any]:
    dev = read_qrels(root / "dev-qrels.json")
    test = read_qrels(root / "eval-qrels.json")
    table = read_json(root / "tuning.json")
    results: dict[str, dict[str, dict[str, float | int]]] = {"development": {}, "evaluation": {}}
    for row in table["rows"]:
        cid = row["config"]["id"]
        run = read_run(root / "dev-runs" / f"{cid}.trec")
        expected = {q: {"ndcg@10": val} for q, val in row["per_query"].items()}
        results["development"][cid] = compare(
            run, dev, expected, {"ndcg@10": row["mean_ndcg@10"]}, dev=True
        )
    for system in read_json(root / "selected.json")["systems"]:
        artifact = validate_evaluation(read_json(root / "evaluations" / f"{system}.json"))
        run = read_run(root / "runs" / f"{system}.trec")
        results["evaluation"][system] = compare(
            run, test, artifact["per_query"], artifact["means"], dev=False
        )
    summary = {
        phase: {
            "per_query_count": sum(r["per_query_count"] for r in configs.values()),
            "mean_count": sum(r["mean_count"] for r in configs.values()),
            "max_abs_error": max(r["max_abs_error"] for r in configs.values()),
        }
        for phase, configs in results.items()
    }
    return {
        "implementation": "pytrec-eval-terrier",
        "version": version("pytrec-eval-terrier"),
        "tolerance": 1e-12,
        "details": results,
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("results/retrieval"))
    args = parser.parse_args()
    for name in FILES:
        directory = args.root / name
        result = crosscheck(directory)
        if result != read_json(directory / "crosscheck.json"):
            raise SystemExit(f"recorded independent check differs for {name}")
        print(name, result["summary"])
