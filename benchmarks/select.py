"""Tune one dataset using development queries and judgments only."""

import math
from collections.abc import Callable, Mapping
from itertools import product

from ragbench.metrics import score_retrieval

from .data import Qrels
from .retrieval import DENSE, PROTOCOL, Config, Index, Scores, fuse, retrieve

type Run = dict[str, Scores]


def bm25_grid() -> list[Config]:
    grid = PROTOCOL["bm25_grid"]
    return [
        {"id": f"bm25-k{k:g}-b{b:g}", "kind": "bm25", "k1": k, "b": b}
        for k, b in product(grid["k1"], grid["b"])
    ]


def rrf_grid(sparse: Config) -> list[Config]:
    grid = PROTOCOL["rrf_grid"]
    return [
        {
            "id": f"rrf-w{w:g}-c{c}",
            "kind": "rrf",
            "dense_weight": w,
            "constant": c,
            "candidates_per_side": 100,
            "sparse": sparse,
            "dense": DENSE,
        }
        for w, c in product(grid["dense_weight"], grid["constant"])
    ]


def select_winner(values: Mapping[str, float]) -> str:
    return min(values, key=lambda key: (-values[key], key))


def tune(
    dev_queries: dict[str, str],
    dev_qrels: Qrels,
    index: Index,
    record: Callable[[Config, Run, dict[str, float], float], None],
) -> dict[str, Config]:
    if dev_queries.keys() != dev_qrels.keys():
        raise ValueError("development queries must exactly match development labels")
    sparse_runs = {}
    grid = bm25_grid()
    index.configure([PROTOCOL["fixed_bm25"], *grid])

    def evaluate(config: Config, runs: Run) -> float:
        scores = {q: row["ndcg@10"] for q, row in score_retrieval(dev_qrels, runs).items()}
        mean = math.fsum(scores.values()) / len(scores)
        record(config, runs, scores, mean)
        return mean

    def rankings(config: Config) -> Run:
        return {q: retrieve(text, config, index) for q, text in dev_queries.items()}

    means = {}
    for config in grid:
        sparse_runs[config["id"]] = rankings(config)
        means[config["id"]] = evaluate(config, sparse_runs[config["id"]])
    sparse = next(c for c in grid if c["id"] == select_winner(means))
    sparse_run = sparse_runs[sparse["id"]]
    dense_run = rankings(DENSE)
    hybrids = rrf_grid(sparse)
    means = {}
    for config in hybrids:
        runs = {
            q: fuse(
                sparse_run[q],
                dense_run[q],
                dense_weight=config["dense_weight"],
                constant=config["constant"],
            )
            for q in dev_queries
        }
        means[config["id"]] = evaluate(config, runs)
    hybrid = next(c for c in hybrids if c["id"] == select_winner(means))
    return {
        "bm25-fixed": PROTOCOL["fixed_bm25"],
        "bm25-tuned": sparse,
        "minilm": DENSE,
        "hybrid-tuned": hybrid,
    }
