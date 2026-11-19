import pytest

from ragbench.io import MAX_INPUT_BYTES, read_qrels, read_run, write_json_atomic


@pytest.mark.parametrize(
    "reader,suffix,content",
    [
        (read_run, ".json", '{"q":{"d":1,"d":2}}'),
        (read_qrels, ".json", '{"q":{"d":1},"q":{"e":1}}'),
        (read_run, ".trec", "q Q0 d 1 1 x\nq Q0 d 2 2 x"),
        (read_qrels, ".trec", "q 0 d 1\nq 0 d 0"),
        (read_qrels, ".tsv", "query-id\tcorpus-id\tscore\nq\td\t1\nq\td\t2"),
        (read_qrels, ".json", '{"q":{"d":true}}'),
        (read_qrels, ".json", '{"q":{"d":1.5}}'),
        (read_run, ".json", '{"q":{"d":NaN}}'),
        (read_run, ".json", '{"q":{"d":1e999}}'),
        (read_run, ".trec", "q Q0 d 1 -inf x"),
        (read_run, ".trec", "q Q0 d 0 1 x"),
        (read_qrels, ".trec", "q 0 d -2"),
        (read_run, ".trec", "q d 0"),
    ],
)
def test_reject_invalid_records(tmp_path, reader, suffix, content):
    path = tmp_path / ("input" + suffix)
    path.write_text(content)
    with pytest.raises(ValueError):
        reader(path)


def test_trec_qrels_and_json_run(tmp_path):
    qrels, run = tmp_path / "refs.txt", tmp_path / "run.json"
    qrels.write_text("q 0 d 2\nq 0 e 0\n")
    run.write_text('{"q":{"e":-0.25}}')
    assert read_qrels(qrels) == {"q": {"d": 2, "e": 0}}
    assert read_run(run) == {"q": {"e": -0.25}}


def test_byte_limit(tmp_path):
    path = tmp_path / "large.json"
    with path.open("wb") as f:
        f.truncate(MAX_INPUT_BYTES + 1)
    with pytest.raises(ValueError, match="byte"):
        read_run(path)


def test_atomic_write_preserves_previous_output(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text("original")
    with pytest.raises(ValueError):
        write_json_atomic(path, {"v": float("nan")})
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]
