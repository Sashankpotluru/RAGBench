"""Paired percentile bootstrap intervals, sign flips and noninferiority gates."""

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
) -> tuple[float, list[float], float]:
    # Every binary64 delta is an integer divided by a power of two. A common
    # denominator makes bootstrap totals and sign-flip ties exact even under
    # cancellation. Use machine integers where every possible draw fits.
    ratios = [[delta.as_integer_ratio() for delta in units[key]] for key in sorted(units)]
    denominator = max(d for group in ratios for _, d in group)
    totals = [sum(n * (denominator // d) for n, d in group) for group in ratios]
    mean = sum(totals) / (denominator * sum(map(len, ratios)))
    if not any(totals):
        return mean, [0.0, 0.0], 1.0
    n = len(totals)
    max_draw = max(map(abs, totals)) * n
    sums = np.array(totals, dtype=np.int64 if max_draw <= np.iinfo(np.int64).max else object)
    sizes = np.array([len(group) for group in ratios])
    observed = abs(sum(totals))
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
        sampled_totals = sums[picks].sum(axis=1).tolist()
        sampled_counts = sizes[picks].sum(axis=1).tolist()
        # Python integer division rounds the exact rational mean once, even
        # when the numerator or denominator cannot itself fit in binary64.
        samples[start : start + count] = [
            total / (denominator * queries)
            for total, queries in zip(sampled_totals, sampled_counts, strict=True)
        ]
        signs = 2 * flip_rng.integers(0, 2, size=(count, n)) - 1
        flipped = np.abs((signs * sums).sum(axis=1))
        extreme += int(np.count_nonzero(flipped >= observed))
    low, high = np.percentile(samples, [2.5, 97.5], method="linear")
    return mean, [float(low), float(high)], (1 + extreme) / (resamples + 1)


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
    if base["dataset_id"] == "squad-v1.1" and groups is None:
        raise ValueError("squad-v1.1 comparisons require article groups for every question")
    units: dict[str, list[float]] = {}
    deltas = []
    for qid in qids:
        delta = cand["per_query"][qid][metric] - base["per_query"][qid][metric]
        deltas.append(delta)
        unit = qid if groups is None else groups[qid]
        units.setdefault(unit, []).append(delta)
    if len(units) < 2:
        raise ValueError("comparison requires at least two independent resampling units")
    mean_delta, ci, p = _resample(units, resamples, seed)
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
        "mean_delta": mean_delta,
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
