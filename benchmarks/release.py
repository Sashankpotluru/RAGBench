"""Build release summaries, paired results and the browser query index from retained evidence."""

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np

from ragbench.compare import compare_evaluations, holm_adjust
from ragbench.evaluation import make_evaluation, reference_digest
from ragbench.io import read_json, write_json_atomic
from ragbench.metrics import score_answers

from .data import FILES, SOURCES, sha256_file
from .generation import CONFIGS, MODEL, PROTOCOL, context_diagnostics
from .vendor.squad import compute_score as upstream

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("bm25-fixed", "bm25-tuned", "minilm", "hybrid-tuned")
SQUAD = "squad-v1.1"
GATE = {
    "dataset_id": SQUAD,
    "kind": "answers",
    "baseline": "closed-book",
    "candidate": "hybrid-top3",
    "metric": "token_f1",
    "margin": 0.02,
}
INPUT_FILES = {
    "benchmarks/protocol.json",
    "benchmarks/sources.json",
    "requirements-benchmark.txt",
    "benchmarks/vendor/squad/compute_score.py",
    "benchmarks/vendor/squad/LICENSE",
    "benchmarks/vendor/squad/source.json",
    "benchmarks/vendor/flan-tokenizer.json",
    "benchmarks/vendor/NOTICE.md",
}


def exact_ids(expected, actual, name):
    if set(expected) != set(actual) or len(expected) != len(actual):
        raise ValueError(f"{name}: incomplete or extra query IDs")


def answer_evaluation(refs, groups, rows, config, provenance, split):
    exact_ids(refs, rows, "predictions")
    return make_evaluation(
        kind="answers",
        dataset_id=SQUAD,
        split_id=split,
        reference_sha256=reference_digest(refs),
        groups=groups,
        config=config,
        provenance=provenance,
        per_query=score_answers(refs, {q: row["prediction"] for q, row in rows.items()}),
    )


def answer_check(refs, rows, evaluation):
    errors = []
    per_metric = {"exact_match": [], "token_f1": []}
    for q, aliases in refs.items():
        for metric, fn in (
            ("exact_match", upstream.exact_match_score),
            ("token_f1", upstream.f1_score),
        ):
            expected = upstream.metric_max_over_ground_truths(fn, rows[q]["prediction"], aliases)
            per_metric[metric].append(expected)
            errors.append(abs(evaluation["per_query"][q][metric] - expected))
    for metric, values in per_metric.items():
        errors.append(abs(evaluation["means"][metric] - math.fsum(values) / len(values)))
    maximum = max(errors)
    if maximum > 1e-12:
        raise ValueError("SQuAD v1.1 cross-check discrepancy")
    return {"per_query_count": len(errors) - 2, "mean_count": 2, "max_abs_error": maximum}


def comparisons(systems, *, primary):
    """One orientation per unordered pair. Adjust only distinct hypotheses."""
    anchor = next(iter(systems))
    families = {}
    for baseline, candidate in combinations(systems, 2):
        for metric in systems[baseline]["means"]:
            result = compare_evaluations(systems[baseline], systems[candidate], metric)
            primary_pair = baseline == anchor and metric == primary
            category = "primary" if primary_pair else "exploratory"
            family = f"{result['dataset_id']}/{result['kind']}/{metric}/{category}"
            row = {
                **result,
                "baseline": baseline,
                "candidate": candidate,
                "family": family,
                "analysis": category,
            }
            families.setdefault(family, {})[f"{baseline}/{candidate}"] = row
    for rows in families.values():
        adjusted = holm_adjust({key: row["p_value"] for key, row in rows.items()})
        for key, row in rows.items():
            row.update(p_holm=adjusted[key], family_size=len(rows))
    return [row for rows in families.values() for row in rows.values()]


def derive(root, results):
    from .replay_retrieval import audit_retrieval
    from .verify import audit_answers

    prefix = results.relative_to(root)
    release = {
        "schema_version": 1,
        "datasets": {},
        "comparisons": {},
        "explorer": str(prefix / "explorer.json"),
        "gates": [],
        "inputs": {
            "sources": "benchmarks/sources.json",
            "dependencies": "requirements-benchmark.txt",
            "retrieval_runtime": str(prefix / "retrieval/runtime.json"),
            "answer_runtime": str(prefix / "answers/runtime.json"),
        },
    }
    explorer = {
        "schema_version": 1,
        "preview_depth": 10,
        "rank_columns": ["docid", "rank", "score", "grade"],
        "datasets": {},
    }
    derived = {}
    inventory = set(INPUT_FILES)
    raw = {"retrieval/runtime.json"}
    for dataset in FILES:
        folder = results / "retrieval" / dataset
        rel = folder.relative_to(results)

        def path(name, base=prefix / rel):
            return str(base / name)

        data = audit_retrieval(folder, root)
        selected, qrels = data["selection"], data["qrels"]
        dataset_info = {
            "split": selected["eval_split"],
            "query_count": len(qrels),
            "document_count": data["provenance"]["document_count"],
            "development_query_count": len(selected["dev_ids"]),
            "selection": path("selection.json"),
            "selected": path("selected.json"),
            "tuning": path("tuning.json"),
            "provenance": path("provenance.json"),
            "retrieval_references": path("eval-qrels.json"),
            "retrieval_crosscheck": path("crosscheck.json"),
            "tokenization": path("tokenization.json"),
            "latency": path("latency.json"),
            "runs": {"retrieval": {}},
        }
        release["datasets"][dataset] = dataset_info
        queries = {q: {"retrieval": {}} for q in qrels}
        explorer["datasets"][dataset] = {"queries": queries}
        raw.update(str(rel / f) for f in data["files"])
        artifacts = data["evaluations"]
        for system, evaluation in artifacts.items():
            timing = data["latency"][system]
            dataset_info["runs"]["retrieval"][system] = {
                "config": evaluation["config"],
                "means": evaluation["means"],
                "evaluation": path(f"evaluations/{system}.json"),
                "ranking": path(f"runs/{system}.trec"),
                "diagnostics": path(f"diagnostics/{system}.json"),
                "p50_ms": timing["p50_ms"],
                "p95_ms": timing["p95_ms"],
                "latency_sample_count": timing["sample_count"],
            }
            for q, observation in queries.items():
                ranks = data["rankings"][system][q]
                observation["retrieval"][system] = {
                    "metrics": evaluation["per_query"][q],
                    **data["diagnostics"][system][q],
                    "top10": [
                        [d, i, s, qrels[q].get(d, 0)]
                        for i, (d, s) in enumerate(list(ranks.items())[:10], 1)
                    ],
                }
        comparisons_by_kind = {"retrieval": comparisons(artifacts, primary="ndcg@10")}
        if dataset == SQUAD:
            folder_a = results / "answers"
            ap = lambda f: str(prefix / "answers" / f)  # noqa: E731
            refs = read_json(folder / "eval-answers.json")
            questions = read_json(folder_a / "questions.json")
            runtime = read_json(folder_a / "runtime.json")
            audit_answers(folder_a, root, questions, qrels)
            raw.update(
                "answers/" + x
                for x in ["runtime.json", "protocol.json", "questions.json", "failures.json"]
            )
            dataset_info.update(
                answer_references=path("eval-answers.json"),
                questions=ap("questions.json"),
                answer_crosscheck=ap("crosscheck.json"),
                generation_protocol=ap("protocol.json"),
            )
            answer_runs = dataset_info["runs"]["answers"] = {}
            for q in qrels:
                queries[q].update(
                    question=questions[q],
                    references=refs[q],
                    article=selected["eval_groups"][q],
                    answers={},
                )
            evals = {}
            checks = {}
            from .verify import audit_prediction, replay_tokenizer

            tokenizer = replay_tokenizer(root)
            for system, (retriever, k) in CONFIGS.items():
                name = f"predictions/{system}.json"
                raw.add("answers/" + name)
                rows = read_json(folder_a / name)
                config = {"system": system, "retriever": retriever, "top_k": k, **PROTOCOL}
                provenance = {
                    "source_commit": runtime["source_commit"],
                    "source_sha256": SOURCES["sources"][FILES[SQUAD]]["sha256"],
                    "model_files": SOURCES["models"][MODEL]["files"],
                    "predictions_sha256": sha256_file(folder_a / name),
                    "questions_sha256": sha256_file(folder_a / "questions.json"),
                    "retrieval_run_sha256": sha256_file(folder / "runs" / f"{retriever}.trec")
                    if retriever
                    else None,
                }
                evaluation = answer_evaluation(
                    refs, selected["eval_groups"], rows, config, provenance, selected["eval_split"]
                )
                evals[system] = evaluation
                ename = f"evaluations/{system}.json"
                derived["answers/" + ename] = evaluation
                checks[system] = answer_check(refs, rows, evaluation)
                for q, row in rows.items():
                    ids = list(data["rankings"][retriever][q])[:k] if retriever else []
                    audit_prediction(row, questions[q], ids, tokenizer)
                    queries[q]["answers"][system] = {
                        **{
                            key: row[key]
                            for key in (
                                "prediction",
                                "spans",
                                "latency_ns",
                                "input_tokens",
                                "context_tokens",
                                "output_tokens",
                            )
                        },
                        "metrics": evaluation["per_query"][q],
                        **context_diagnostics(row["spans"], qrels[q], refs[q]),
                    }
                durations = [r["latency_ns"] / 1e6 for r in rows.values()]
                answer_runs[system] = {
                    "config": config,
                    "means": evaluation["means"],
                    "evaluation": ap(ename),
                    "predictions": ap(name),
                    "p50_ms": float(np.percentile(durations, 50)),
                    "p95_ms": float(np.percentile(durations, 95)),
                    "latency_sample_count": len(rows),
                    **{
                        f"mean_{col}": math.fsum(r[col] for r in rows.values()) / len(rows)
                        for col in ("input_tokens", "output_tokens", "context_tokens")
                    },
                    **{
                        f"{key}_count": sum(v["answers"][system][key] for v in queries.values())
                        for key in ("gold_passage_supplied", "answer_string_in_context")
                    },
                }
            derived["answers/crosscheck.json"] = {
                "upstream": read_json(root / "benchmarks/vendor/squad/source.json"),
                "tolerance": 1e-12,
                "systems": checks,
            }
            comparisons_by_kind["answers"] = comparisons(evals, primary="token_f1")
        release["comparisons"][dataset] = {}
        for kind, rows in comparisons_by_kind.items():
            name = f"comparisons/{dataset}-{kind}.json"
            derived[name] = rows
            release["comparisons"][dataset][kind] = str(prefix / name)
            for row in rows:
                if all(row[field] == value for field, value in GATE.items()):
                    release["gates"].append(
                        {**GATE, "gate": row["gate"], "ci95": row["ci95"], "family": row["family"]}
                    )
    if len(release["gates"]) != 1:
        raise ValueError("declared release gate unavailable")
    derived["explorer.json"] = explorer
    inventory.update(str(prefix / n) for n in raw | derived.keys())
    return release, derived, inventory


def build(root: Path, results: Path):
    from .verify import check_coverage, descriptor

    release, derived, inventory = derive(root, results)
    for name, value in derived.items():
        target = results / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, value)
    check_coverage(root, results, inventory)
    release["manifest"] = [descriptor(root, root / p) for p in sorted(inventory)]
    write_json_atomic(results / "release.json", release)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    build(ROOT, args.results.resolve())
