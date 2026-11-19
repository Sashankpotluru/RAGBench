"""Per-query retrieval and SQuAD v1.1 answer metrics."""

import math
import re
import string
from collections import Counter

from .io import identifier, validate_qrels, validate_run

RETRIEVAL_METRICS = ("ndcg@10", "recall@5", "recall@10", "recall@100", "mrr@10", "precision@10")
ANSWER_METRICS = ("exact_match", "token_f1")
_DISCOUNTS = [1 / math.log2(rank + 1) for rank in range(1, 11)]
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _require_known_queries(reference_ids: set[str], supplied_ids: set[str]) -> None:
    unknown = supplied_ids - reference_ids
    if unknown:
        raise ValueError(f"unknown query ID {min(unknown)!r}")


def score_retrieval(
    qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    judgments = validate_qrels(qrels)
    ranking_scores = validate_run(run)
    _require_known_queries(set(judgments), set(ranking_scores))
    result = {}
    for qid in sorted(judgments):
        grades = judgments[qid]
        scores = ranking_scores.get(qid, {})
        ranking = sorted(scores, key=lambda doc: (-scores[doc], doc))[:100]
        gains = [grades.get(doc, 0) for doc in ranking]
        ideal = sorted(grades.values(), reverse=True)[:10]
        relevant = sum(grade > 0 for grade in grades.values())
        hits = [rank for rank, gain in enumerate(gains, 1) if gain > 0]
        # Scaling both DCGs by the best gain also handles very large integer grades.
        ndcg = 0.0
        if ideal[0]:
            dcg = math.fsum(
                (g / ideal[0]) * w for g, w in zip(gains[:10], _DISCOUNTS, strict=False)
            )
            idcg = math.fsum((g / ideal[0]) * w for g, w in zip(ideal, _DISCOUNTS, strict=False))
            ndcg = dcg / idcg
        hit_counts = {k: sum(rank <= k for rank in hits) for k in (5, 10, 100)}
        result[qid] = {
            "ndcg@10": ndcg,
            **{f"recall@{k}": n / relevant if relevant else 0.0 for k, n in hit_counts.items()},
            "mrr@10": 1 / hits[0] if hits and hits[0] <= 10 else 0.0,
            "precision@10": hit_counts[10] / 10,
        }
    return result


def _tokens(answer: str) -> list[str]:
    return _ARTICLES.sub(" ", answer.lower().translate(_PUNCTUATION)).split()


def score_answers(
    references: dict[str, list[str]], predictions: dict[str, str]
) -> dict[str, dict[str, float]]:
    if not isinstance(references, dict) or not references:
        raise ValueError("references must be a nonempty query object")
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a query object")
    for qid, aliases in references.items():
        identifier(qid, "reference query ID")
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(a, str) for a in aliases)
        ):
            raise ValueError(f"expected a nonempty list of reference strings for {qid!r}")
    for qid, answer in predictions.items():
        identifier(qid, "prediction query ID")
        if not isinstance(answer, str):
            raise ValueError(f"expected a prediction string for {qid!r}")
    _require_known_queries(set(references), set(predictions))

    result = {}
    for qid in sorted(references):
        if qid not in predictions:
            result[qid] = {"exact_match": 0.0, "token_f1": 0.0}
            continue
        predicted = _tokens(predictions[qid])
        counts = Counter(predicted)
        best_em, best_f1 = 0.0, 0.0
        for alias in references[qid]:
            expected = _tokens(alias)
            best_em = max(best_em, float(predicted == expected))
            overlap = (counts & Counter(expected)).total()
            if overlap:
                best_f1 = max(best_f1, 2 * overlap / (len(predicted) + len(expected)))
        result[qid] = {"exact_match": best_em, "token_f1": best_f1}
    return result
