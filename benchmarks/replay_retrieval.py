"""Replay the retained development grid and held-out retrieval evidence."""

import math

import numpy as np

from ragbench.evaluation import make_evaluation, reference_digest
from ragbench.io import read_json, read_qrels
from ragbench.metrics import score_retrieval

from .crosscheck import crosscheck
from .data import FILES, SOURCES, hash_order, sha256_file
from .release import SQUAD, SYSTEMS, exact_ids
from .retrieval import DENSE, PROTOCOL, rank
from .select import bm25_grid, rrf_grid, select_winner
from .verify import confined, require, runtime_check, same


def canonical_run(path, ids, depth, tag):
    """Preserve and check every recorded rank, score, tag and tie, including unjudged rows."""
    run = {}
    with path.open(encoding="utf8") as stream:
        for line in stream:
            q, iteration, doc, pos, val, actual_tag = line.rstrip("\n").split("\t")
            docs = run.setdefault(q, {})
            score = float(val)
            require(
                iteration == "Q0"
                and tag == actual_tag
                and doc
                and doc not in docs
                and int(pos) == len(docs) + 1
                and math.isfinite(score),
                f"TREC row in {path.name}",
            )
            docs[doc] = score
    exact_ids(ids, run, f"{path.name} queries")
    for docs in run.values():
        require(
            len(docs) == depth and list(docs) == list(rank(docs, depth)),
            f"rank order or depth in {path.name}",
        )
    return run


def audit_selection(selected, tuning, default, rows, folder):
    same(
        selected,
        {
            "split_id": default,
            "selection": PROTOCOL["selection"],
            "tuning_sha256": sha256_file(folder / "tuning.json"),
            "systems": {"bm25-fixed": PROTOCOL["fixed_bm25"], "minilm": DENSE, **rows},
        },
        "development-selected config and evidence",
    )
    require(tuning["split_id"] == default, "development tuning split")


def audit_retrieval(folder, root):
    dataset = folder.name
    machine = read_json(folder.parent / "runtime.json")
    runtime_check(machine)
    files = set()

    def path(name):
        files.add(name)
        return confined(root, str((folder / name).relative_to(root)))

    provenance = read_json(path("provenance.json"))
    source = SOURCES["sources"][FILES[dataset]]
    require(
        provenance["source"] == source
        and provenance["filename"] == FILES[dataset]
        and provenance["dataset_id"] == dataset
        and provenance["protocol_sha256"] == sha256_file(root / "benchmarks/protocol.json")
        and provenance["source_commit"] == machine["source_commit"],
        "retrieval provenance",
    )
    cfg = read_json(path("selected.json"))
    sel = read_json(path("selection.json"))
    nf = dataset != SQUAD
    require(provenance["document_count"] == (3633 if nf else 2067), "document count")
    qrels = read_qrels(path("eval-qrels.json"))
    dev = read_qrels(path("dev-qrels.json"))
    for phase, refs, count, split in (
        ("dev", dev, 324 if nf else 50, "official-dev" if nf else "project-hash-v1-dev50"),
        ("eval", qrels, 323 if nf else 200, "official-test" if nf else "project-hash-v1-eval200"),
    ):
        ids, groups = sel[phase + "_ids"], sel[phase + "_groups"]
        exact_ids(refs, ids, phase + " IDs")
        require(len(ids) == count and sel[phase + "_split"] == split, phase + " split")
        if nf:
            require(groups is None and sel["article_pools"] is None, "NFCorpus groups")
        else:
            exact_ids(ids, groups, phase + " groups")
            exact_ids(ids, read_json(path(phase + "-answers.json")), phase + " answers")
            require(
                ids == sorted(ids, key=lambda q: hash_order("ragbench-v1-question:", q))
                and set(groups.values()) <= set(sel["article_pools"][phase]),
                "SQuAD selection",
            )
    require(not (qrels.keys() & dev.keys()), "split overlap")
    if not nf:
        pools = sel["article_pools"]
        all_articles = pools["dev"] + pools["eval"]
        require(
            len(pools["dev"]) == 10
            and len(set(all_articles)) == len(all_articles) == 48
            and all_articles
            == sorted(all_articles, key=lambda a: hash_order("ragbench-v1-article:", a)),
            "article-disjoint pools",
        )
    tuning = read_json(path("tuning.json"))
    require(tuning["reference_sha256"] == reference_digest(dev), "tuning qrels")
    table = tuning["rows"]
    choices = {}
    offset = 0
    for label in ("bm25-tuned", "hybrid-tuned"):
        configs = bm25_grid() if label == "bm25-tuned" else rrf_grid(choices["bm25-tuned"])
        means = {}
        for config in configs:
            cid = config["id"]
            run = canonical_run(path(f"dev-runs/{cid}.trec"), dev, 10, cid)
            scores = {q: entries["ndcg@10"] for q, entries in score_retrieval(dev, run).items()}
            means[cid] = math.fsum(scores.values()) / len(scores)
            same(
                table[offset],
                {"config": config, "mean_ndcg@10": means[cid], "per_query": scores},
                f"tuning row {cid}",
            )
            offset += 1
        choices[label] = next(c for c in configs if c["id"] == select_winner(means))
    require(len(table) == offset, "grid coverage")
    audit_selection(cfg, tuning, sel["dev_split"], choices, folder)

    lengths = read_json(path("tokenization.json"))
    key = lengths["dense_cache_key"]
    model = SOURCES["models"][DENSE["model"]]
    require(
        key["source_sha256"] == source["sha256"]
        and key["files"] == model["files"]
        and key["revision"] == model["revision"]
        and key["encoder"] == DENSE
        and key["document_order_sha256"] == provenance["document_order_sha256"]
        and key["representation"] == PROTOCOL["corpus_text"]
        and lengths["document_count"] == provenance["document_count"]
        and lengths["max_wordpieces"] == DENSE["max_length"]
        and lengths["includes_special_tokens"] is True
        and 0 <= lengths["document_truncated"] <= lengths["document_count"],
        "dense tokenization",
    )
    for phase, refs in [("dev", dev), ("eval", qrels)]:
        counts = lengths[phase + "_query_lengths"]
        exact_ids(refs, counts, phase + " query token counts")
        require(
            all(type(n) is int and n > 0 for n in counts.values())
            and lengths[phase + "_query_truncated"]
            == sum(n > lengths["max_wordpieces"] for n in counts.values()),
            phase + " query token counts",
        )
    timings = read_json(path("latency.json"))
    same(timings["protocol"], PROTOCOL["timing"], "latency protocol")
    exact_ids(SYSTEMS, timings["systems"], "latency systems")
    timed_ids = sorted(qrels, key=lambda q: hash_order(PROTOCOL["timing"]["hash_prefix"], q))[:32]
    evaluations, rankings, diagnostics = {}, {}, {}
    for system in SYSTEMS:
        runfile = path(f"runs/{system}.trec")
        run = canonical_run(runfile, qrels, 100, system)
        rankings[system] = run
        artifact = read_json(path(f"evaluations/{system}.json"))
        expected = make_evaluation(
            kind="retrieval",
            dataset_id=dataset,
            split_id=sel["eval_split"],
            reference_sha256=reference_digest(qrels),
            groups=sel["eval_groups"],
            per_query=score_retrieval(qrels, run),
            config=cfg["systems"][system],
            provenance={
                "system": system,
                "source_commit": machine["source_commit"],
                "source_sha256": source["sha256"],
                "run_sha256": sha256_file(runfile),
                "selected_sha256": sha256_file(folder / "selected.json"),
            },
        )
        same(artifact, expected, f"{system} retrieval evaluation")
        evaluations[system] = artifact
        diag = read_json(path(f"diagnostics/{system}.json"))
        expected_diag = {}
        for q, docs in run.items():
            hits = [i for i, d in enumerate(docs, 1) if qrels[q].get(d, 0) > 0]
            expected_diag[q] = {
                "top_document": next(iter(docs)),
                "first_relevant_rank": hits[0] if hits else None,
                "relevant_in_top100": len(hits),
                "positive_judgments": sum(g > 0 for g in qrels[q].values()),
            }
        same(diag, expected_diag, "retrieval diagnostics")
        diagnostics[system] = diag
        timing = timings["systems"][system]
        exact_ids(timed_ids, timing["samples_ns"], "timed query IDs")
        durations = []
        for samples in timing["samples_ns"].values():
            require(
                len(samples) == 5 and all(type(t) is int and t > 0 for t in samples),
                "timing sample counts",
            )
            durations.extend(t / 1e6 for t in samples)
        require(
            timing["query_order"] == timed_ids
            and timing["sample_count"] == len(durations)
            and [timing["p50_ms"], timing["p95_ms"]] == list(np.percentile(durations, [50, 95])),
            "retrieval timing percentiles",
        )
    same(read_json(path("crosscheck.json")), crosscheck(folder), "independent trec_eval replay")
    return {
        "files": files,
        "provenance": provenance,
        "selection": sel,
        "qrels": qrels,
        "evaluations": evaluations,
        "rankings": rankings,
        "diagnostics": diagnostics,
        "latency": timings["systems"],
    }
