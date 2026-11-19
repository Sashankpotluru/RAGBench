"""Synthetic parser fixtures contain authored text only."""

import hashlib
import zipfile
from dataclasses import replace

import pytest

from benchmarks.data import check_source, parse_beir, parse_squad, validate_dataset


def fixture():
    return {
        "data": [
            {
                "title": f"Article {j}",
                "paragraphs": [
                    {
                        "context": f" Authored {j}. ",
                        "qas": [
                            {"id": f"q{j}-{i}", "question": "Which?", "answers": [{"text": f"{j}"}]}
                            for i in range(5)
                        ],
                    }
                ],
            }
            for j in range(8)
        ]
    }


def test_selection_rule_and_replication():
    raw = fixture()
    data = parse_squad(raw, "digest", dev_articles=3, dev_size=4, eval_size=8)
    assert data == parse_squad(raw, "digest", dev_articles=3, dev_size=4, eval_size=8)

    def hashed(prefix, value):
        return hashlib.sha256((prefix + value).encode()).hexdigest()

    indices = sorted(range(8), key=lambda j: hashed("ragbench-v1-article:", f"Article {j}"))
    for ids, pool, size in [
        (data.dev_queries, indices[:3], 4),
        (data.eval_queries, indices[3:], 8),
    ]:
        candidates = [f"q{j}-{i}" for j in pool for i in range(5)]
        assert (
            list(ids) == sorted(candidates, key=lambda q: hashed("ragbench-v1-question:", q))[:size]
        )
    assert data.documents["squad:0:0"]["text"] == " Authored 0. "
    assert len(data.documents) == 8
    validate_dataset(data)
    with pytest.raises(ValueError, match="query"):
        validate_dataset(replace(data, eval_queries=data.dev_queries))
    with pytest.raises(ValueError, match="group"):
        validate_dataset(
            replace(
                data,
                eval_groups=dict.fromkeys(data.eval_queries, next(iter(data.dev_groups.values()))),
            )
        )
    with pytest.raises(ValueError, match="document"):
        validate_dataset(replace(data, documents={}))


def test_duplicate_question_even_if_outside_selection():
    raw = fixture()
    raw["data"][-1]["paragraphs"][0]["qas"][-1]["id"] = "q0-0"
    with pytest.raises(ValueError, match="duplicate"):
        parse_squad(raw, "digest", dev_articles=3, dev_size=4, eval_size=8)


@pytest.mark.parametrize("name", ["../escape", "/abs", "nfcorpus/../../x", "x\\y", "C:/evil"])
def test_zip_paths_are_checked_even_for_unused_members(tmp_path, name):
    path = tmp_path / "test.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, "x")
    with pytest.raises(ValueError, match="unsafe"):
        parse_beir(path, "digest")


def test_source_tampering(tmp_path):
    path = tmp_path / "a"
    path.write_bytes(b"abc")
    expected = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    check_source(path, expected)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="digest"):
        check_source(path, expected)


def test_duplicate_beir_document(tmp_path):
    path = tmp_path / "small.zip"
    line = '{"_id":"d","text":"Authored.","title":"One"}\n'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("nfcorpus/corpus.jsonl", line * 2)
    with pytest.raises(ValueError, match="duplicate"):
        parse_beir(path, "digest")
