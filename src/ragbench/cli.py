"""Command-line imports and paired decisions."""

import argparse
import sys
from pathlib import Path

from .compare import compare_evaluations
from .evaluation import make_evaluation, reference_digest
from .io import read_json, read_qrels, read_run, write_json_atomic
from .metrics import score_answers, score_retrieval


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbench", description="Evaluate and compare recorded predictions."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="score a retrieval run")
    score.add_argument("--qrels", type=Path, required=True)
    score.add_argument("--run", type=Path, required=True)
    answers = commands.add_parser("answers", help="score answer predictions")
    answers.add_argument("--references", type=Path, required=True)
    answers.add_argument("--predictions", type=Path, required=True)
    for command in (score, answers):
        command.add_argument("--dataset-id", required=True)
        command.add_argument("--split-id", required=True)
        command.add_argument("--groups", type=Path)
    compare = commands.add_parser("compare", help="compare paired evaluations")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--metric", required=True)
    compare.add_argument("--margin", type=float, default=0.02)
    compare.add_argument("--resamples", type=int, default=10000)
    compare.add_argument("--seed", type=int, default=20261119)
    for command in (score, answers, compare):
        command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compare":
            result = compare_evaluations(
                read_json(args.baseline),
                read_json(args.candidate),
                args.metric,
                margin=args.margin,
                resamples=args.resamples,
                seed=args.seed,
            )
            write_json_atomic(args.output, result)
            return {"pass": 0, "regression": 1, "inconclusive": 2}[result["gate"]]
        if args.command == "score":
            kind = "retrieval"
            references = read_qrels(args.qrels)
            rows = score_retrieval(references, read_run(args.run))
        else:
            kind = "answers"
            answer_references = read_json(args.references)
            rows = score_answers(answer_references, read_json(args.predictions))
            references = answer_references
        groups = None
        if args.groups is not None:
            groups = read_json(args.groups)
            if not isinstance(groups, dict):
                raise ValueError("--groups requires a JSON object")
        evaluation = make_evaluation(
            kind=kind,
            dataset_id=args.dataset_id,
            split_id=args.split_id,
            reference_sha256=reference_digest(references),
            per_query=rows,
            groups=groups,
        )
        write_json_atomic(args.output, evaluation)
        return 0
    except (OSError, ValueError) as exc:
        print(f"ragbench: invalid input or output: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
