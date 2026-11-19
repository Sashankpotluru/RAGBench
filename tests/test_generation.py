import pytest

from benchmarks.generation import build_prompt, context_diagnostics


class TinyTokenizer:
    """One token per non-space character, plus EOS on complete inputs."""

    def __call__(self, text, *, add_special_tokens=True, **kwargs):
        offsets = [(i, i + 1) for i, c in enumerate(text) if not c.isspace()]
        return {
            "input_ids": [ord(text[i]) for i, _ in offsets] + ([1] if add_special_tokens else []),
            "offset_mapping": offsets,
        }


def test_budget_spans_and_gold_independence():
    tok = TinyTokenizer()
    passages = [("first", "a" * 320), ("second", "y" * 200 + " gold"), ("third", "gold")]
    for limit in (1, 3):
        prompt = build_prompt(tok, "Which?", passages[:limit])
        assert len(prompt["input_ids"]) <= 512
        assert prompt["context_tokens"] <= 384
        assert [s["docid"] for s in prompt["spans"]] == ["first", "second"][:limit]
        for s in prompt["spans"]:
            assert s["text"] == dict(passages)[s["docid"]][s["start"] : s["end"]]
        assert context_diagnostics(prompt["spans"], {"third": 1}, ["gold"]) == {
            "gold_passage_supplied": False,
            "answer_string_in_context": False,
        }
        changed = context_diagnostics(prompt["spans"], {"first": 1}, ["a"])
        assert all(changed.values())
        assert build_prompt(tok, "Which?", passages[:limit]) == prompt


def test_reduce_context_for_question_and_reject_overlong():
    tok = TinyTokenizer()
    prompt = build_prompt(tok, "q" * 440, [("d", "c" * 500)])
    assert len(prompt["input_ids"]) <= 512 and 0 < prompt["context_tokens"] < 100
    assert "q" * 440 in prompt["prompt"]
    with pytest.raises(ValueError, match="question"):
        build_prompt(tok, "q" * 513, [])
    closed = build_prompt(tok, "Question?", [])
    assert closed["spans"] == [] and "Context:" not in closed["prompt"]
    assert closed["prompt"].endswith("Question: Question?\nAnswer:")


def test_effective_flan_budget_and_prediction_replay():
    from benchmarks.release import ROOT
    from benchmarks.verify import audit_prediction, replay_tokenizer

    tokenizer = replay_tokenizer(ROOT)

    def encode(text, add_special_tokens=True, **kwargs):
        result = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": result.ids, "offset_mapping": result.offsets}

    question = "Which country? " * 35
    passages = [(str(i), "Glaciers cover the southern island. " * 22) for i in range(3)]
    row = build_prompt(encode, question, passages)
    output = tokenizer.encode("Iceland").ids
    row.update(
        input_tokens=len(row["input_ids"]),
        output_ids=output,
        output_tokens=len(output),
        latency_ns=1,
        prediction="Iceland",
    )
    assert row["input_tokens"] <= 512
    assert 0 < row["context_tokens"] <= 384
    assert len(row["spans"]) >= 2
    audit_prediction(row, question, passages, tokenizer)
    for field, wrong in [("prediction", "France"), ("context_tokens", 0), ("prompt", "different")]:
        with pytest.raises(ValueError):
            audit_prediction({**row, field: wrong}, question, passages, tokenizer)
    with pytest.raises(ValueError, match="question"):
        build_prompt(encode, "unabridged question? " * 300, passages)


def test_replay_rejects_closed_shortened_and_misattributed_context():
    from benchmarks.release import ROOT
    from benchmarks.verify import audit_prediction, replay_tokenizer

    tokenizer = replay_tokenizer(ROOT)

    def encode(text, add_special_tokens=True, **kwargs):
        encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": encoded.ids, "offset_mapping": encoded.offsets}

    original = [("a", "An ice sheet."), ("b", "Glaciers flow downhill.")]
    for incorrect in [
        [],
        original[:1],
        [("a", "An ice"), original[1]],
        [("a", original[1][1]), ("b", original[0][1])],
    ]:
        row = build_prompt(encode, "What flows?", incorrect)
        row.update(
            input_tokens=len(row["input_ids"]),
            output_ids=[1],
            output_tokens=1,
            prediction="",
            latency_ns=1,
        )
        with pytest.raises(ValueError, match="expected prompt"):
            audit_prediction(row, "What flows?", original, tokenizer)


def test_generation_stops_only_at_eos_or_length_limit():
    from benchmarks.release import ROOT
    from benchmarks.verify import audit_prediction, replay_tokenizer

    tokenizer = replay_tokenizer(ROOT)
    question = "Which country?"
    row = build_prompt(lambda t, **kw: {"input_ids": tokenizer.encode(t).ids}, question, [])
    row.update(input_tokens=len(row["input_ids"]), latency_ns=1)
    for ids, valid in [
        ([493], False),
        ([493, 1, 493], False),
        ([493, 1, 1], False),
        ([493, 1], True),
        ([493] * 48, True),
        ([493] * 47 + [1], True),
    ]:
        result = {
            **row,
            "output_ids": ids,
            "output_tokens": len(ids),
            "prediction": tokenizer.decode(ids, skip_special_tokens=True),
        }
        if valid:
            audit_prediction(result, question, [], tokenizer)
        else:
            with pytest.raises(ValueError, match="stopping"):
                audit_prediction(result, question, [], tokenizer)


def test_runner_persists_binding_before_model_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmarks import run_answers as runner
    from ragbench.io import read_json

    machine = {"source_commit": "a" * 40}
    monkeypatch.setattr(runner, "runtime", lambda _: machine)
    monkeypatch.setattr(
        runner, "load_dataset", lambda *args: SimpleNamespace(eval_queries={"q": "Where?"})
    )
    monkeypatch.setattr(
        runner, "source_inputs", lambda *args: ({"aliases_sha256": "b" * 64}, {}, {"d": "Answer."})
    )

    def stop_at_load(_):
        bindings = list(tmp_path.glob("*/answers/input-binding.json"))
        assert len(bindings) == 1
        binding = read_json(bindings[0])
        assert binding["origin"] == {
            "stage": "before_inference",
            "binding_source_commit": "a" * 40,
            "measurement_source_commit": "a" * 40,
        }
        assert binding["inputs"] == {"aliases_sha256": "b" * 64}
        assert read_json(bindings[0].with_name("retrieved-passages.json")) == {"d": "Answer."}
        raise RuntimeError("fixture stop before inference")

    monkeypatch.setattr(runner, "Generator", stop_at_load)
    with pytest.raises(RuntimeError, match="fixture stop"):
        runner.run_answers(tmp_path / "answers", tmp_path / "retrieval", tmp_path / "cache")
    assert not (tmp_path / "answers").exists()
