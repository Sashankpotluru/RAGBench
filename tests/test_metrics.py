import math
from pathlib import Path

import pytest

from ragbench.io import read_qrels, read_run
from ragbench.metrics import score_answers, score_retrieval

FIXTURES = Path(__file__).parent / "fixtures"


def test_hand_scored_graded_retrieval():
    qrels = read_qrels(FIXTURES / "graded.tsv")
    scores = score_retrieval(qrels, read_run(FIXTURES / "ranking.trec"))
    actual, ideal = 1 / math.log2(3) + 3 / 2, 3 + 2 / math.log2(3) + 1 / 2
    assert scores["q1"] == pytest.approx(
        {
            "ndcg@10": actual / ideal,
            "recall@5": 2 / 3,
            "recall@10": 2 / 3,
            "recall@100": 2 / 3,
            "mrr@10": 0.5,
            "precision@10": 0.2,
        }
    )
    assert set(scores["q2"].values()) == {0.0}


def test_cutoffs_and_tie_order():
    qrels = {"q": {"d006": 1, "d011": 1, "d101": 1}, "late": {"d011": 1}}
    tied = {f"d{i:03}": 2.0 for i in reversed(range(1, 102))}
    result = score_retrieval(qrels, {"late": tied, "q": tied})
    assert list(result) == ["late", "q"]
    assert result["q"]["recall@5"] == 0
    assert result["q"]["recall@10"] == 1 / 3
    assert result["q"]["recall@100"] == 2 / 3
    assert result["q"]["mrr@10"] == 1 / 6
    assert result["late"]["mrr@10"] == 0
    assert score_retrieval({"z": {"d": 0}}, {"z": {"d": 0}})["z"]["ndcg@10"] == 0


@pytest.mark.parametrize(
    "refs,run",
    [
        ({}, {}),
        ({"q": {}}, {}),
        ({"q": {"d": -1}}, {}),
        ({"q": {"d": True}}, {}),
        ({"q": {"d": 1.0}}, {}),
        ({"q": {"d": 1}}, {"unknown": {}}),
        ({"q": {"d": 1}}, {"q": {"d": float("inf")}}),
        ({"q": {"d": 1}}, {"q": {"d": False}}),
    ],
)
def test_retrieval_invalid(refs, run):
    with pytest.raises(ValueError):
        score_retrieval(refs, run)


def test_squad_v11_aliases_repeated_tokens_and_empty():
    refs = {
        "q1": ["river Nile", "The Nile!"],
        "q2": ["red red fox"],
        "q3": ["the"],
        "q4": ["star"],
        "q5": ["a-b"],
    }
    preds = {"q1": "   NILE.", "q2": "RED fox fox", "q3": "", "q5": "ab"}
    result = score_answers(refs, preds)
    assert result["q1"] == {"exact_match": 1, "token_f1": 1}
    assert result["q2"] == {"exact_match": 0, "token_f1": pytest.approx(2 / 3)}
    assert result["q3"] == {"exact_match": 1, "token_f1": 0}
    assert result["q4"] == {"exact_match": 0, "token_f1": 0}
    assert result["q5"]["exact_match"] == 1


@pytest.mark.parametrize(
    "refs,preds",
    [
        ({}, {}),
        ({"q": []}, {}),
        ({"q": "x"}, {}),
        ({"q": [1]}, {}),
        ({"q": ["x"]}, {"u": ""}),
        ({"q": ["x"]}, {"q": 0}),
    ],
)
def test_invalid_answer_inputs(refs, preds):
    with pytest.raises(ValueError):
        score_answers(refs, preds)


def test_missing_prediction_and_explicit_empty_against_empty_normalized_reference():
    scores = score_answers({"missing": ["the"], "supplied": ["the"]}, {"supplied": ""})
    assert scores == {
        "missing": {"exact_match": 0, "token_f1": 0},
        "supplied": {"exact_match": 1, "token_f1": 0},
    }
