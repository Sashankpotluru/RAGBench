from benchmarks.select import bm25_grid, select_winner


def test_winner_is_development_only_even_if_evaluation_prefers_other_config():
    development = {"b": 0.2, "a": 0.8}
    evaluation = {"b": 1.0, "a": 0.0}
    assert select_winner(development) == "a"
    assert select_winner(evaluation) == "b"
    evaluation["a"] = 1.0
    assert select_winner(development) == "a"
    assert select_winner({"z": 0.5, "a": 0.5}) == "a"
    assert {(c["k1"], c["b"]) for c in bm25_grid()} == {
        (k, b) for k in (0.6, 1.2, 1.8) for b in (0.0, 0.5, 1.0)
    }


def test_tune_uses_only_development_relevance(monkeypatch):
    from benchmarks import select
    from ragbench.metrics import score_retrieval

    class FixtureIndex:
        def configure(self, configs):
            pass

    def run_for(query, config, index):
        assert query == "Authored development query"
        first = "doc-A" if config.get("k1") == 0.6 else "doc-B"
        second = "doc-B" if first == "doc-A" else "doc-A"
        return {first: 2.0, second: 1.0}

    monkeypatch.setattr(select, "retrieve", run_for)
    dev = {"qid": "Authored development query"}
    dev_qrels = {"qid": {"doc-A": 1}}
    test_qrels = {"held": {"doc-B": 1}}
    index = FixtureIndex()

    def freeze():
        return select.tune(dev, dev_qrels, index, lambda *args: None)["bm25-tuned"]

    winner = freeze()
    assert winner["k1"] == 0.6
    alternate = next(c for c in select.bm25_grid() if c["k1"] == 1.8)

    def held_out_value(c):
        return score_retrieval(test_qrels, {"held": run_for(dev["qid"], c, index)})["held"][
            "ndcg@10"
        ]

    assert held_out_value(alternate) > held_out_value(winner)
    test_qrels["held"] = {"doc-A": 1}
    assert held_out_value(alternate) < held_out_value(winner)
    assert freeze() == winner
