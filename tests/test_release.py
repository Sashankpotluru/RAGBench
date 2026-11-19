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


def test_same_ids_cannot_rebind_answer_inputs(tmp_path):
    from benchmarks.answer_inputs import input_identity, validate_binding

    (tmp_path / "runs").mkdir()
    (tmp_path / "selected.json").write_text("{}")
    for retriever in ("bm25-tuned", "hybrid-tuned"):
        (tmp_path / "runs" / f"{retriever}.trec").write_text("q\tQ0\tdoc\t1\t1.0\ttag\n")
    inputs = {
        "questions": {"q": "Which island?"},
        "refs": {"q": ["Iceland"]},
        "selection": {"eval_split": "held-out", "eval_ids": ["q"], "eval_groups": {"q": "Islands"}},
        "qrels": {"q": {"doc": 1}},
        "passages": {"doc": "Iceland is an island."},
    }
    runtime = {"source_commit": "a" * 40}
    binding = {
        "schema_version": 1,
        "inputs": input_identity(tmp_path, **inputs),
        "origin": {
            "stage": "before_inference",
            "measurement_source_commit": "a" * 40,
            "binding_source_commit": "a" * 40,
        },
    }
    validate_binding(binding, input_identity(tmp_path, **inputs), runtime)
    for change in [
        {"refs": {"q": ["Greenland"]}},
        {"qrels": {"q": {"other": 1}}},
        {"selection": {**inputs["selection"], "eval_groups": {"q": "Changed"}}},
        {"passages": {"doc": "Wrong source body."}},
    ]:
        with pytest.raises(ValueError, match="binding"):
            validate_binding(binding, input_identity(tmp_path, **{**inputs, **change}), runtime)
    ranking = tmp_path / "runs/hybrid-tuned.trec"
    ranking.write_text(ranking.read_text().replace("doc", "other"))
    with pytest.raises(ValueError, match="binding"):
        validate_binding(binding, input_identity(tmp_path, **inputs), runtime)


def test_runtime_rejects_missing_fields_wrong_types_and_settings():
    from benchmarks.release import ROOT
    from benchmarks.verify import runtime_check
    from ragbench.io import read_json

    machine = read_json(ROOT / "results/answers/runtime.json")
    runtime_check(machine)
    for key in machine:
        with pytest.raises(ValueError):
            runtime_check({k: v for k, v in machine.items() if k != key})
    for key, replacement in [
        ("logical_cpus", True),
        ("torch_interop_threads", 0),
        ("torch_threads", 4.0),
        ("processor", " "),
        ("os", 5),
        ("synchronization", "async"),
        ("tokenizers_parallelism", 0),
        ("versions", {**machine["versions"], "torch": False}),
        ("versions", {"torch": "2.15.0"}),
        ("python", "unknown"),
    ]:
        with pytest.raises(ValueError):
            runtime_check({**machine, key: replacement})


@pytest.mark.parametrize("existing", ["retrieved-passages.json", "input-binding.json"])
def test_resume_partial_input_archive_validates_before_filling_companion(tmp_path, existing):
    from benchmarks.answer_inputs import persist

    args = (
        {"questions_sha256": "a" * 64},
        {"p": "A blue gate."},
        {"stage": "retrospective_source_verification", "binding_source_commit": "b" * 40},
    )
    persist(tmp_path, *args)
    first = tmp_path / existing
    companion = next(p for p in tmp_path.iterdir() if p != first)
    original = first.read_bytes()
    companion_bytes = companion.read_bytes()
    companion.unlink()
    first.write_bytes(original + b" ")
    with pytest.raises(FileExistsError, match=existing):
        persist(tmp_path, *args)
    assert first.read_bytes() == original + b" "
    assert not companion.exists()

    first.write_bytes(original)
    inode = first.stat().st_ino
    persist(tmp_path, *args)
    assert first.stat().st_ino == inode and first.read_bytes() == original
    assert companion.read_bytes() == companion_bytes
    persist(tmp_path, *args)
