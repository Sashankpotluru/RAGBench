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
