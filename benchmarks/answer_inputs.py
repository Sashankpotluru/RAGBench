"""Bind answer measurements to source-checked references and ranked passage bodies."""

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

from ragbench.evaluation import reference_digest
from ragbench.io import read_json, read_run, write_json_atomic

from .data import FILES, SOURCES, Dataset, load_dataset, sha256_file
from .generation import CONFIGS
from .retrieval import rank
from .run_retrieval import ROOT, git

DATASET = "squad-v1.1"
RETRIEVERS = {name: k for name, k in CONFIGS.values() if name is not None}


def input_identity(folder: Path, *, questions, refs, selection, qrels, passages):
    """Canonical semantic digests for objects; byte digests for the frozen TREC/config files."""
    return {
        "dataset_id": DATASET,
        "source_sha256": SOURCES["sources"][FILES[DATASET]]["sha256"],
        "split_id": selection["eval_split"],
        "query_order_sha256": reference_digest(selection["eval_ids"]),
        "questions_sha256": reference_digest(questions),
        "aliases_sha256": reference_digest(refs),
        "groups_sha256": reference_digest(selection["eval_groups"]),
        "qrels_sha256": reference_digest(qrels),
        "selected_sha256": sha256_file(folder / "selected.json"),
        "rankings_sha256": {s: sha256_file(folder / "runs" / f"{s}.trec") for s in RETRIEVERS},
        "passages_sha256": reference_digest(passages),
    }


def source_inputs(folder: Path, data: Dataset):
    """Read and compare every identity-bearing reference against the verified original source."""
    selection = read_json(folder / "selection.json")
    if (
        list(data.eval_queries) != selection["eval_ids"]
        or len(data.eval_queries) != 200
        or data.eval_split != selection["eval_split"]
        or data.eval_groups != selection["eval_groups"]
        or data.eval_qrels != read_json(folder / "eval-qrels.json")
        or data.eval_answers != read_json(folder / "eval-answers.json")
        or data.fingerprint != SOURCES["sources"][FILES[DATASET]]["sha256"]
    ):
        raise ValueError("retained SQuAD evidence differs from verified dataset")
    ranks = {}
    passages = {}
    for system, k in RETRIEVERS.items():
        run_file = folder / "runs" / f"{system}.trec"
        evaluation = read_json(folder / "evaluations" / f"{system}.json")
        provenance = evaluation["provenance"]
        if (
            provenance["run_sha256"] != sha256_file(run_file)
            or provenance["selected_sha256"] != sha256_file(folder / "selected.json")
            or provenance["source_sha256"] != data.fingerprint
        ):
            raise ValueError("retrieval source or selection digest mismatch")
        scores = read_run(run_file)
        if scores.keys() != data.eval_queries.keys() or any(len(s) != 100 for s in scores.values()):
            raise ValueError("incomplete retrieval rankings")
        ranks[system] = {q: list(rank(s, k)) for q, s in scores.items()}
        for ids in ranks[system].values():
            for doc in ids:
                passages[doc] = data.documents[doc]["text"]
    identity = input_identity(
        folder,
        questions=data.eval_queries,
        refs=data.eval_answers,
        selection=selection,
        qrels=data.eval_qrels,
        passages=passages,
    )
    return identity, ranks, passages


def validate_binding(binding, expected, runtime):
    if binding.get("schema_version") != 1 or reference_digest(
        binding.get("inputs")
    ) != reference_digest(expected):
        raise ValueError(
            "answer input binding differs from current references, passages or rankings"
        )
    origin = binding.get("origin", {})
    commit = runtime["source_commit"]
    if origin.get("measurement_source_commit") != commit:
        raise ValueError("answer input binding measurement commit mismatch")
    stage = origin.get("stage")
    if stage == "before_inference":
        if origin != {
            "stage": stage,
            "measurement_source_commit": commit,
            "binding_source_commit": commit,
        }:
            raise ValueError("invalid pre-inference binding provenance")
    elif (
        stage != "retrospective_source_verification"
        or set(origin)
        != {
            "stage",
            "measurement_source_commit",
            "binding_source_commit",
            "original_evidence_commit",
        }
        or any(
            not isinstance(value, str) or not re.fullmatch("[0-9a-f]{40}", value)
            for key, value in origin.items()
            if key.endswith("_commit")
        )
    ):
        raise ValueError("invalid retrospective binding provenance")


def persist(folder: Path, identity, passages, origin):
    if (folder / "input-binding.json").exists() or (folder / "retrieved-passages.json").exists():
        raise FileExistsError("answer input archive already exists")
    write_json_atomic(folder / "retrieved-passages.json", passages)
    write_json_atomic(
        folder / "input-binding.json", {"schema_version": 1, "origin": origin, "inputs": identity}
    )


def backfill(results: Path, cache: Path, evidence_commit: str):
    """One-time reconstruction for a committed pre-binding run. Never runs the model."""
    if not re.fullmatch("[0-9a-f]{40}", evidence_commit):
        raise ValueError("supply the full original evidence commit")
    if git("status", "--porcelain", "--", "benchmarks", "src"):
        raise ValueError("commit binding code before retrospective reconstruction")
    folder = results / "answers"
    source = results / "retrieval" / DATASET
    # Compare exact original-file bytes at the supplied immutable revision before backfilling.
    files = [
        folder / n for n in ("runtime.json", "questions.json", "protocol.json", "failures.json")
    ]
    files += [folder / "predictions" / f"{s}.json" for s in CONFIGS]
    files += [
        source / n
        for n in ("selection.json", "selected.json", "eval-answers.json", "eval-qrels.json")
    ]
    for system in RETRIEVERS:
        files += [source / "runs" / f"{system}.trec", source / "evaluations" / f"{system}.json"]
    for path in files:
        blob = subprocess.check_output(
            ["git", "show", f"{evidence_commit}:{path.relative_to(ROOT)}"], cwd=ROOT
        )
        if hashlib.sha256(blob).hexdigest() != sha256_file(path):
            raise ValueError(f"file differs from original evidence commit: {path.name}")
    data = load_dataset(DATASET, cache)
    if read_json(folder / "questions.json") != data.eval_queries:
        raise ValueError("retained questions differ from verified source")
    identity, _, passages = source_inputs(source, data)
    origin = {
        "stage": "retrospective_source_verification",
        "measurement_source_commit": read_json(folder / "runtime.json")["source_commit"],
        "binding_source_commit": git("rev-parse", "HEAD"),
        "original_evidence_commit": evidence_commit,
    }
    persist(folder, identity, passages, origin)
    print(
        f"Bound original predictions to {len(data.eval_queries)} questions and {len(passages)} candidate paragraphs."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrospective-evidence-commit", required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    backfill(args.results.resolve(), args.cache, args.retrospective_evidence_commit)
