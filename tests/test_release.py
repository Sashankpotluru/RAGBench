from copy import deepcopy

import pytest

from benchmarks.release import answer_evaluation, comparisons
from benchmarks.verify import check_manifest, descriptor, gate_exit


def test_descriptors_bind_content_and_reject_escape(tmp_path):
    p = tmp_path / "source.json"
    p.write_text("{}")
    d = descriptor(tmp_path, p)
    check_manifest(tmp_path, [d], {"source.json"})
    p.write_text("[]")
    with pytest.raises(ValueError):
        check_manifest(tmp_path, [d], {"source.json"})
    with pytest.raises(ValueError):
        check_manifest(tmp_path, [{**d, "path": "../source.json"}], {"../source.json"})
    with pytest.raises(ValueError):
        check_manifest(tmp_path, [], {"source.json"})


def test_predictions_coverage_metrics_and_gate():
    refs = {"q1": ["yes"], "q2": ["yes"]}
    groups = {"q1": "article1", "q2": "article2"}
    rows = {q: {"prediction": "yes"} for q in refs}

    def evaluate(rows):
        return answer_evaluation(refs, groups, rows, {}, {}, "fixture")

    baseline = evaluate(rows)
    assert baseline["means"] == {"exact_match": 1, "token_f1": 1}
    for missing_or_extra in [{}, {**rows, "extra": {"prediction": "yes"}}]:
        with pytest.raises(ValueError):
            evaluate(missing_or_extra)
    bad = deepcopy(rows)
    for row in bad.values():
        row["prediction"] = "no"
    candidate = evaluate(bad)
    report = comparisons(
        {"closed-book": baseline, "bm25-top1": candidate, "hybrid-top3": candidate},
        primary="token_f1",
    )
    reg = next(
        r
        for r in report
        if r["baseline"] == "closed-book"
        and r["candidate"] == "hybrid-top3"
        and r["metric"] == "token_f1"
    )
    assert reg["family_size"] == 2 and gate_exit(reg) == 1
    assert gate_exit({**reg, "gate": "inconclusive"}) == 2
    assert gate_exit({**reg, "gate": "pass"}) == 0


def test_selection_and_canonical_ranking_cannot_drift(tmp_path):
    from benchmarks.data import sha256_file
    from benchmarks.replay_retrieval import audit_selection, canonical_run
    from benchmarks.retrieval import DENSE, PROTOCOL

    (tmp_path / "tuning.json").write_text("{}")
    picked = {"bm25-tuned": {"id": "winner"}, "hybrid-tuned": {"id": "hybrid"}}
    selected = {
        "split_id": "dev",
        "selection": PROTOCOL["selection"],
        "tuning_sha256": sha256_file(tmp_path / "tuning.json"),
        "systems": {"bm25-fixed": PROTOCOL["fixed_bm25"], "minilm": DENSE, **picked},
    }
    audit_selection(selected, {"split_id": "dev"}, "dev", picked, tmp_path)
    selected["systems"]["bm25-tuned"] = {"id": "held-out-winner"}
    with pytest.raises(ValueError, match="selected"):
        audit_selection(selected, {"split_id": "dev"}, "dev", picked, tmp_path)
    p = tmp_path / "rank.trec"
    p.write_text("q\tQ0\ta\t1\t1.0\trank\nq\tQ0\tb\t2\t0.5\trank\n")
    assert canonical_run(p, ["q"], 2, "rank") == {"q": {"a": 1.0, "b": 0.5}}
    for bad in [
        "q\tQ0\tb\t1\t0.5\trank\nq\tQ0\ta\t2\t1.0\trank\n",
        "q\tQ0\ta\t2\t1.0\trank\n",
        "q\tQ0\ta\t1\tnan\trank\n",
    ]:
        p.write_text(bad)
        with pytest.raises(ValueError):
            canonical_run(p, ["q"], 2, "rank")


@pytest.mark.parametrize("field", ["metric", "ranking", "prediction", "source", "selection"])
def test_referenced_artifact_modification_rejected(tmp_path, field):
    file = tmp_path / f"{field}.json"
    file.write_text('{"value":1}')
    manifest = [descriptor(tmp_path, file)]
    check_manifest(tmp_path, manifest, {file.name})
    file.write_text('{"value":2}')
    with pytest.raises(ValueError, match="hash or size"):
        check_manifest(tmp_path, manifest, {file.name})


def test_published_release_reconstructs():
    from benchmarks.release import ROOT
    from benchmarks.verify import verify

    report = verify(ROOT, ROOT / "results")
    assert report["datasets"]["nfcorpus-beir"]["query_count"] == 323
    assert report["datasets"]["squad-v1.1"]["query_count"] == 200
    assert set(report["datasets"]["squad-v1.1"]["runs"]["answers"]) == {
        "bm25-top1",
        "hybrid-top3",
        "closed-book",
    }
