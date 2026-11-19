import copy

import pytest

from ragbench.compare import compare_evaluations, holm_adjust
from ragbench.evaluation import make_evaluation, reference_digest, validate_evaluation


def evaluation(values, groups=None):
    return make_evaluation(
        kind="answers",
        dataset_id="demo",
        split_id="heldout",
        reference_sha256=reference_digest({"r": 1}),
        groups=groups,
        per_query={q: {"exact_match": x, "token_f1": x} for q, x in values.items()},
    )


def test_identity_and_constant_drop():
    baseline = evaluation(dict.fromkeys("abcd", 1))
    same = compare_evaluations(baseline, baseline, "token_f1")
    assert (same["mean_delta"], same["ci95"], same["gate"]) == (0, [0, 0], "pass")
    assert (same["wins"], same["ties"], same["losses"], same["p_value"]) == (0, 4, 0, 1)
    result = compare_evaluations(baseline, evaluation(dict.fromkeys("abcd", 0)), "token_f1")
    assert (result["ci95"], result["gate"], result["losses"]) == ([-1, -1], "regression", 4)
    assert result["p_value"] == pytest.approx(2 / 16, abs=0.02)


def test_shuffled_query_mapping_preserves_pairing_and_is_reproducible():
    base = evaluation({"a": 0, "b": 1, "c": 0})
    candidate = evaluation({"c": 0, "b": 0, "a": 1})
    candidate["per_query"] = dict(reversed(candidate["per_query"].items()))
    result = compare_evaluations(base, candidate, "token_f1")
    assert compare_evaluations(base, candidate, "token_f1") == result
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 1)
    assert result["gate"] == "inconclusive"


def test_cluster_weighting_and_cluster_sign_flips():
    groups = {"a": "large", "b": "large", "c": "large", "d": "small"}
    base = evaluation({"a": 0, "b": 0, "c": 0, "d": 1}, groups)
    candidate = evaluation({"a": 1, "b": 1, "c": 1, "d": 0}, groups)
    result = compare_evaluations(base, candidate, "token_f1")
    assert result["mean_delta"] == 0.5
    assert result["ci95"] == [-1, 1]
    # The four cluster sign combinations have absolute sums 2 or 4.
    assert result["p_value"] == 1
    assert result["unit_count"] == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("dataset_id", "other"),
        ("split_id", "other"),
        ("reference_sha256", "0" * 64),
        ("groups", {"q": "other", "r": "second"}),
        ("scoring_version", "other"),
        ("kind", "retrieval"),
        ("query_ids", ["other"]),
        ("means", {"exact_match": 1, "token_f1": 0}),
        ("per_query", {"q": {"token_f1": 0}}),
    ],
)
def test_refuse_mismatch_or_tamper(key, value):
    base = evaluation({"q": 0, "r": 0})
    changed = copy.deepcopy(base)
    changed[key] = value
    with pytest.raises(ValueError):
        compare_evaluations(base, changed, "token_f1")


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("reference_sha256", "x"),
        ("dataset_id", ""),
        ("groups", {}),
        ("groups", {"q": ""}),
        ("provenance", {"n": float("nan")}),
        ("per_query", {"q": {"exact_match": True, "token_f1": 0}}),
        ("per_query", {"q": {"exact_match": 2, "token_f1": 0}}),
    ],
)
def test_validate_full_artifact(key, value):
    row = evaluation({"q": 0})
    row[key] = value
    with pytest.raises(ValueError):
        validate_evaluation(row)


def test_canonical_digest_and_query_ids():
    assert reference_digest({"b": 1, "a": 2}) == reference_digest({"a": 2, "b": 1})
    row = evaluation({"z": 0, "a": 1})
    assert row["query_ids"] == ["a", "z"]
    assert row["means"] == {"exact_match": 0.5, "token_f1": 0.5}


def test_holm_stepdown_and_bounds():
    assert holm_adjust(dict(a=0.02, b=0.01, c=0.7, d=0.019)) == pytest.approx(
        dict(a=0.057, b=0.04, c=0.7, d=0.057)
    )
    for invalid in (-0.01, 1.1, float("nan"), True):
        with pytest.raises(ValueError):
            holm_adjust({"p": invalid})


@pytest.mark.parametrize(
    "params", [{"resamples": 0}, {"seed": True}, {"margin": -1}, {"margin": float("nan")}]
)
def test_invalid_comparison_options(params):
    base = evaluation({"q": 1, "r": 1})
    with pytest.raises(ValueError):
        compare_evaluations(base, base, "exact_match", **params)


@pytest.mark.parametrize(
    "values,groups", [({"q": 1}, None), ({"q": 1, "r": 1}, {"q": "a", "r": "a"})]
)
def test_inference_requires_two_independent_units(values, groups):
    base = evaluation(values, groups)
    with pytest.raises(ValueError, match="at least two independent"):
        compare_evaluations(base, base, "exact_match")


def test_bootstrap_uses_query_count_of_each_sampled_cluster():
    groups = {f"x{i}": "large" for i in range(10)}
    groups.update({f"y{i}": f"small{i}" for i in range(5)})
    base = evaluation(dict.fromkeys(groups, 0), groups)
    cand = evaluation({q: int(g == "large") for q, g in groups.items()}, groups)
    result = compare_evaluations(base, cand, "token_f1")
    # In six draws P(K<=2)=.938 and P(K<=3)=.991 for K copies of the large cluster.
    # With three copies the sample has 30 improving queries and three unchanged.
    assert result["ci95"] == pytest.approx([0, 30 / 33])
    assert result["mean_delta"] == pytest.approx(10 / 15)


def test_sign_flip_includes_observed_sample_plus_one():
    base = evaluation(dict.fromkeys("abcdefgh", 0))
    cand = evaluation(dict.fromkeys("abcdefgh", 1))
    result = compare_evaluations(base, cand, "exact_match", resamples=1)
    assert result["p_value"] == 0.5
    assert result["p_holm"] == result["p_value"]
    assert result["family_size"] == 1


def test_gate_includes_exact_lower_margin():
    base = evaluation(dict.fromkeys("ab", 0.02))
    cand = evaluation(dict.fromkeys("ab", 0))
    assert compare_evaluations(base, cand, "token_f1")["gate"] == "pass"
