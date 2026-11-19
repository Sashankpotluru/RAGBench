import pytest

from benchmarks.retrieval import fuse, rank


def test_fusion_with_alternate_weight_and_constant():
    sparse = {"a": 3.0, "c": 2.0, "b": 2.0}
    dense = {"c": 5.0, "a": 1.0}
    result = fuse(sparse, dense, dense_weight=0.25, constant=10, top_k=3)
    assert list(result) == ["a", "c", "b"]
    assert result == pytest.approx(
        {"a": 0.75 / 11 + 0.25 / 12, "c": 0.75 / 13 + 0.25 / 11, "b": 0.75 / 12}
    )


def test_fusion_uses_only_top_100_from_each_side():
    many = {f"d{i:03d}": 1.0 for i in range(101)}
    assert "d100" not in fuse(many, {}, dense_weight=0.5, constant=60, top_k=101)
    assert list(rank({"z": 1.0, "a": 1.0, "b": 2.0}, 2)) == ["b", "a"]


def test_cache_rejects_file_from_different_encoder_key(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from benchmarks.retrieval import _vectors
    from ragbench.evaluation import reference_digest

    key = {"revision": "authored", "document_order": ["d"]}
    path = tmp_path / "dense" / (reference_digest(key) + ".npz")
    path.parent.mkdir()
    np.savez(path, key="stale", vectors=np.ones((1, 3), np.float32), lengths=[4], checksum="stale")
    encoder = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(hidden_size=3)))
    with pytest.raises(ValueError, match="stale"):
        _vectors(encoder, ["authored"], key, tmp_path)


def test_encoder_reads_only_verified_model_files(tmp_path, monkeypatch):
    import hashlib
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock

    from benchmarks.retrieval import MODEL, SOURCES, Encoder

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    approved = {"config.json": b"{}", "subdir/fixture.txt": b"authored"}
    for name, content in approved.items():
        path = snapshot / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
    (snapshot / "added_tokens.json").write_text('{"authored-extra": 42}')
    pins = {name: hashlib.sha256(content).hexdigest() for name, content in approved.items()}
    monkeypatch.setitem(SOURCES["models"][MODEL], "files", pins)
    hub = Mock(return_value=str(snapshot))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=hub))

    observed = []

    def load(location, **kwargs):
        # A Hub model ID reopens the whole shared snapshot, including optional files.
        path = snapshot if location == MODEL else Path(location)
        files = {
            p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()
        }
        observed.append((path, files, kwargs))
        return Mock()

    loader = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=loader, AutoModel=loader)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    Encoder(tmp_path)
    assert len(observed) == 2
    assert all(files == approved for _, files, _ in observed)
    assert all(
        kwargs["local_files_only"] and not kwargs["trust_remote_code"] for _, _, kwargs in observed
    )
    assert observed[0][0] == observed[1][0]
    assert hub.call_args.kwargs["local_files_only"]
    assert hub.call_args.kwargs["revision"] == SOURCES["models"][MODEL]["revision"]

    (snapshot / "config.json").write_bytes(b"modified")
    with pytest.raises(ValueError, match="digest mismatch"):
        Encoder(tmp_path)
    assert len(observed) == 2
