"""Paired percentile bootstrap intervals, sign flips and noninferiority gates."""

import math
from typing import Any

import numpy as np

from .evaluation import reference_digest, validate_evaluation
from .io import finite_number, identifier


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    if not isinstance(pvalues, dict):
        raise ValueError("pvalues must be an object")
    valid: dict[str, float] = {}
    for name, value in pvalues.items():
        identifier(name, "comparison ID")
        p = finite_number(value, "p-value")
        if not 0 <= p <= 1:
            raise ValueError("p-values must be in [0, 1]")
        valid[name] = p
    adjusted = {}
    previous = 0.0
    for i, name in enumerate(sorted(valid, key=lambda key: (valid[key], key))):
        previous = max(previous, min(1.0, (len(valid) - i) * valid[name]))
        adjusted[name] = previous
    return adjusted


def _resample(
    units: dict[str, list[float]], resamples: int, seed: int
) -> tuple[list[float], float]:
    ordered = [units[key] for key in sorted(units)]
    sums = np.array([math.fsum(values) for values in ordered])
    if not sums.any():
        return [0.0, 0.0], 1.0
    sizes = np.array([len(values) for values in ordered])
    observed = abs(math.fsum(sums))
    n = len(sums)
    bootstrap_seed, flip_seed = np.random.SeedSequence(seed).spawn(2)
    bootstrap_rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    flip_rng = np.random.Generator(np.random.PCG64(flip_seed))
    samples = np.empty(resamples)
    extreme = 0
    # Bound temporary matrices independently of the number of queries.
    batch = max(1, min(512, 262_144 // n))
    for start in range(0, resamples, batch):
        count = min(batch, resamples - start)
        picks = bootstrap_rng.integers(0, n, size=(count, n))
        samples[start : start + count] = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
        signs = 2 * flip_rng.integers(0, 2, size=(count, n)) - 1
        flipped = np.abs((signs * sums).sum(axis=1))
        # Include ties when different addition orders differ only by rounding.
        extreme += int(
            np.count_nonzero(
                (flipped >= observed) | np.isclose(flipped, observed, rtol=1e-14, atol=0)
            )
        )
    low, high = np.percentile(samples, [2.5, 97.5], method="linear")
    return [float(low), float(high)], (1 + extreme) / (resamples + 1)


def compare_evaluations(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric: str,
    *,
    resamples: int = 10000,
    seed: int = 20261119,
    margin: float = 0.02,
) -> dict[str, Any]:
    base, cand = validate_evaluation(baseline), validate_evaluation(candidate)
    for field in (
        "kind",
        "dataset_id",
        "split_id",
        "reference_sha256",
        "scoring_version",
        "query_ids",
        "groups",
    ):
        if base[field] != cand[field]:
            raise ValueError(f"incompatible evaluations: {field} differs")
    if not isinstance(metric, str) or metric not in base["means"]:
        raise ValueError(f"unknown metric {metric!r}")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    margin = finite_number(margin, "margin")
    if not 0 <= margin <= 1:
        raise ValueError("margin must be in [0, 1]")
    qids, groups = base["query_ids"], base["groups"]
    units: dict[str, list[float]] = {}
    deltas = []
    for qid in qids:
        delta = cand["per_query"][qid][metric] - base["per_query"][qid][metric]
        deltas.append(delta)
        unit = qid if groups is None else groups[qid]
        units.setdefault(unit, []).append(delta)
    if len(units) < 2:
        raise ValueError("comparison requires at least two independent resampling units")
    ci, p = _resample(units, resamples, seed)
    if ci[0] >= -margin:
        gate = "pass"
    elif ci[1] < -margin:
        gate = "regression"
    else:
        gate = "inconclusive"
    return {
        "schema_version": 1,
        "kind": base["kind"],
        "dataset_id": base["dataset_id"],
        "split_id": base["split_id"],
        "reference_sha256": base["reference_sha256"],
        "scoring_version": base["scoring_version"],
        "metric": metric,
        "baseline_sha256": reference_digest(base),
        "candidate_sha256": reference_digest(cand),
        "baseline_mean": base["means"][metric],
        "candidate_mean": cand["means"][metric],
        "mean_delta": math.fsum(deltas) / len(qids),
        "ci95": ci,
        "interval_scope": "pairwise",
        "resampling_unit": "query" if groups is None else "group",
        "unit_count": len(units),
        "query_count": len(qids),
        "wins": sum(d > 0 for d in deltas),
        "ties": sum(d == 0 for d in deltas),
        "losses": sum(d < 0 for d in deltas),
        "p_value": p,
        "p_holm": p,
        "family_size": 1,
        "resamples": resamples,
        "seed": seed,
        "margin": margin,
        "gate": gate,
    }
