"""Measure both pinned datasets, freeze development choices, and write replayable evidence."""

import argparse
import os
import platform
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ragbench.evaluation import make_evaluation, reference_digest, validate_evaluation
from ragbench.io import read_json, write_json_atomic
from ragbench.metrics import score_retrieval

from .crosscheck import crosscheck
from .data import FILES, SOURCES, Dataset, hash_order, load_dataset, sha256_file
from .retrieval import DENSE, PROTOCOL, Config, Encoder, Index, rank, retrieve
from .select import Run, tune

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def runtime(device: str) -> dict[str, Any]:
    import torch

    if device != "cpu":
        raise ValueError("protocol requires CPU float32")
    if git(
        "status", "--porcelain", "--", "benchmarks", "src", "scripts", "requirements-benchmark.txt"
    ):
        raise ValueError("commit benchmark sources before measurement")
    torch.set_num_threads(PROTOCOL["timing"]["threads"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    processor = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        processor = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "device": device,
        "dtype": "float32",
        "processor": processor,
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "tokenizers_parallelism": False,
        "synchronization": "CPU operations synchronous",
        "versions": {
            pkg: version(pkg)
            for pkg in (
                "numpy",
                "rank-bm25",
                "torch",
                "transformers",
                "huggingface-hub",
                "tokenizers",
                "safetensors",
                "pytrec-eval-terrier",
            )
        },
    }


def selection(data: Dataset, root: Path, machine: Config) -> None:
    root.mkdir()
    for name in ("dev-runs", "runs", "evaluations", "diagnostics"):
        (root / name).mkdir()
    write_json_atomic(
        root / "provenance.json",
        {
            "dataset_id": data.dataset_id,
            "source": SOURCES["sources"][FILES[data.dataset_id]],
            "filename": FILES[data.dataset_id],
            "source_commit": machine["source_commit"],
            "document_count": len(data.documents),
            "document_order_sha256": reference_digest(sorted(data.documents)),
            "protocol_sha256": sha256_file(Path(__file__).with_name("protocol.json")),
        },
    )
    write_json_atomic(
        root / "selection.json",
        {
            "dev_split": data.dev_split,
            "eval_split": data.eval_split,
            "dev_ids": list(data.dev_queries),
            "eval_ids": list(data.eval_queries),
            "dev_groups": data.dev_groups,
            "eval_groups": data.eval_groups,
            "article_pools": data.article_pools,
        },
    )
    write_json_atomic(root / "dev-qrels.json", data.dev_qrels)
    write_json_atomic(root / "eval-qrels.json", data.eval_qrels)
    if data.dev_answers is not None:
        write_json_atomic(root / "dev-answers.json", data.dev_answers)
        write_json_atomic(root / "eval-answers.json", data.eval_answers)


def write_run(path: Path, run: Run, tag: str, limit: int = 100) -> None:
    with path.open("x", encoding="utf8", newline="\n") as stream:
        for qid in sorted(run):
            for pos, (doc, score) in enumerate(rank(run[qid], limit).items(), 1):
                stream.write(f"{qid}\tQ0\t{doc}\t{pos}\t{score!r}\t{tag}\n")


def freeze(data: Dataset, root: Path, index: Index) -> None:
    table: list[Config] = []

    def record(config: Config, runs: Run, scores: dict[str, float], mean: float) -> None:
        write_run(root / "dev-runs" / f"{config['id']}.trec", runs, config["id"], limit=10)
        table.append({"config": config, "mean_ndcg@10": mean, "per_query": scores})

    systems = tune(data.dev_queries, data.dev_qrels, index, record)
    write_json_atomic(
        root / "tuning.json",
        {
            "split_id": data.dev_split,
            "reference_sha256": reference_digest(data.dev_qrels),
            "rows": table,
        },
    )
    write_json_atomic(
        root / "selected.json",
        {
            "systems": systems,
            "selection": PROTOCOL["selection"],
            "split_id": data.dev_split,
            "tuning_sha256": sha256_file(root / "tuning.json"),
        },
    )


def latency(queries: dict[str, str], config: Config, index: Index) -> Config:
    settings = PROTOCOL["timing"]
    ids = sorted(queries, key=lambda q: hash_order(settings["hash_prefix"], q))[
        : settings["query_count"]
    ]
    measurements = {}
    for qid in ids:
        query = queries[qid]
        retrieve(query, config, index)
        samples = []
        for _ in range(settings["repetitions"]):
            start = time.perf_counter_ns()
            retrieve(query, config, index)
            samples.append(time.perf_counter_ns() - start)
        measurements[qid] = samples
    raw = [x / 1e6 for samples in measurements.values() for x in samples]
    return {
        "query_order": ids,
        "samples_ns": measurements,
        "p50_ms": float(np.percentile(raw, 50)),
        "p95_ms": float(np.percentile(raw, 95)),
        "sample_count": len(raw),
    }


def evaluate(data: Dataset, root: Path, index: Index, machine: Config) -> None:
    frozen = read_json(root / "selected.json")
    dev_lengths = dict(
        zip(data.dev_queries, index.encoder.lengths(list(data.dev_queries.values())), strict=True)
    )
    lengths = dict(
        zip(data.eval_queries, index.encoder.lengths(list(data.eval_queries.values())), strict=True)
    )
    limit = DENSE["max_length"]
    write_json_atomic(
        root / "tokenization.json",
        {
            "max_wordpieces": limit,
            "includes_special_tokens": True,
            "document_truncated": sum(n > limit for n in index.lengths),
            "document_count": len(index.ids),
            "document_max": max(index.lengths),
            "dev_query_lengths": dev_lengths,
            "eval_query_lengths": lengths,
            "dev_query_truncated": sum(n > limit for n in dev_lengths.values()),
            "eval_query_truncated": sum(n > limit for n in lengths.values()),
            "dense_cache_key": index.key,
        },
    )
    timings = {}
    for label, config in frozen["systems"].items():
        print(f"{data.dataset_id} evaluating {label}", flush=True)
        run = {q: retrieve(text, config, index) for q, text in data.eval_queries.items()}
        run_path = root / "runs" / f"{label}.trec"
        write_run(run_path, run, label)
        artifact = make_evaluation(
            kind="retrieval",
            dataset_id=data.dataset_id,
            split_id=data.eval_split,
            reference_sha256=reference_digest(data.eval_qrels),
            per_query=score_retrieval(data.eval_qrels, run),
            groups=data.eval_groups,
            config=config,
            provenance={
                "system": label,
                "source_commit": machine["source_commit"],
                "source_sha256": data.fingerprint,
                "run_sha256": sha256_file(run_path),
                "selected_sha256": sha256_file(root / "selected.json"),
            },
        )
        write_json_atomic(root / "evaluations" / f"{label}.json", validate_evaluation(artifact))
        diagnostics = {}
        for qid, scores in run.items():
            docs = list(rank(scores, 100))
            judged = data.eval_qrels[qid]
            hits = [i for i, doc in enumerate(docs, 1) if judged.get(doc, 0) > 0]
            diagnostics[qid] = {
                "top_document": docs[0],
                "first_relevant_rank": hits[0] if hits else None,
                "relevant_in_top100": len(hits),
                "positive_judgments": sum(g > 0 for g in judged.values()),
            }
        write_json_atomic(root / "diagnostics" / f"{label}.json", diagnostics)
        timings[label] = latency(data.eval_queries, config, index)
    write_json_atomic(root / "latency.json", {"protocol": PROTOCOL["timing"], "systems": timings})
    write_json_atomic(root / "crosscheck.json", crosscheck(root))


def run_retrieval(output_dir: Path | str, cache_dir: Path | str, device: str = "cpu") -> None:
    dest, cache = Path(output_dir), Path(cache_dir)
    if dest.exists():
        raise FileExistsError("choose a new output directory for this experiment")
    dest.parent.mkdir(parents=True, exist_ok=True)
    machine = runtime(device)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temporary:
        stage = Path(temporary) / "retrieval"
        stage.mkdir()
        write_json_atomic(stage / "runtime.json", machine)
        datasets = [load_dataset(name, cache) for name in FILES]
        for data in datasets:
            selection(data, stage / data.dataset_id, machine)
        encoder = Encoder(cache)
        indexes = {}
        for data in datasets:
            print(f"{data.dataset_id} indexing and tuning", flush=True)
            index = Index(data, encoder, cache)
            indexes[data.dataset_id] = index
            freeze(data, stage / data.dataset_id, index)
        for data in datasets:
            evaluate(data, stage / data.dataset_id, indexes[data.dataset_id], machine)
        stage.rename(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/retrieval"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    run_retrieval(args.output, args.cache)
