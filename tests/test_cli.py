import json
import subprocess
import sys
from pathlib import Path

from ragbench.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_score_cli_and_error_preserves_file(tmp_path):
    out = tmp_path / "out.json"
    args = [
        "score",
        "--qrels",
        str(FIXTURES / "graded.tsv"),
        "--run",
        str(FIXTURES / "ranking.trec"),
        "--dataset-id",
        "fixture",
        "--split-id",
        "test",
        "--output",
        str(out),
    ]
    assert main(args) == 0
    original = out.read_bytes()
    assert json.loads(original)["means"]["precision@10"] == 0.1
    args[4] = str(tmp_path / "missing.run")
    assert main(args) != 0
    assert out.read_bytes() == original


def test_answers_compare_gate_exit_codes(tmp_path):
    def save(name, data):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(data))
        return str(path)

    refs = save("refs", {"a": ["X"], "b": ["Y"]})
    groups = save("groups", {"a": "first", "b": "second"})
    paths = []
    for i, predictions in enumerate([{"a": "X", "b": "Y"}, {}, {"a": "X"}]):
        path = str(tmp_path / f"result{i}.json")
        assert (
            main(
                [
                    "answers",
                    "--references",
                    refs,
                    "--predictions",
                    save(str(i), predictions),
                    "--dataset-id",
                    "d",
                    "--split-id",
                    "s",
                    "--groups",
                    groups,
                    "--output",
                    path,
                ]
            )
            == 0
        )
        paths.append(path)
    for candidate, status in zip(paths, [0, 1, 2], strict=True):
        out = tmp_path / "report.json"
        assert (
            main(["compare", paths[0], candidate, "--metric", "exact_match", "--output", str(out)])
            == status
        )
        assert (
            json.loads(out.read_bytes())["gate"] == ["pass", "regression", "inconclusive"][status]
        )


def test_installed_console_outside_checkout(tmp_path):
    console = Path(sys.executable).parent / "ragbench"
    result = subprocess.run([console, "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "compare" in result.stdout


def test_answers_missing_key_against_empty_alias_and_squad_comparison(tmp_path, capsys):
    refs = tmp_path / "refs.json"
    preds = tmp_path / "preds.json"
    out = tmp_path / "eval.json"
    compare_out = tmp_path / "compare.json"
    refs.write_text('{"missing":["the"], "supplied":["the"]}')
    preds.write_text('{"supplied":""}')
    assert (
        main(
            [
                "answers",
                "--dataset-id",
                "squad-v1.1",
                "--split-id",
                "s",
                "--references",
                str(refs),
                "--predictions",
                str(preds),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["means"] == {"exact_match": 0.5, "token_f1": 0}
    assert (
        main(
            ["compare", str(out), str(out), "--metric", "exact_match", "--output", str(compare_out)]
        )
        == 3
    )
    assert "article" in capsys.readouterr().err
    assert not compare_out.exists()


def test_compare_cli_accepts_twenty_equal_losses_at_margin(tmp_path):
    from ragbench.evaluation import make_evaluation

    paths = []
    for label, score in [("base", 0.02), ("candidate", 0)]:
        path = tmp_path / (label + ".json")
        artifact = make_evaluation(
            kind="answers",
            dataset_id="synthetic",
            split_id="s",
            reference_sha256="0" * 64,
            per_query={str(i): {"exact_match": 0, "token_f1": score} for i in range(20)},
        )
        path.write_text(json.dumps(artifact))
        paths.append(str(path))
    output = tmp_path / "report.json"
    assert main(["compare", *paths, "--metric", "token_f1", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["ci95"] == [-0.02, -0.02]
