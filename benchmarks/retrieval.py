"""Exact corpus-wide BM25 and dense retrieval, followed by optional weighted RRF."""

import copy
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ragbench.evaluation import reference_digest
from ragbench.io import read_json

from .data import SOURCES, Dataset, sha256_file

PROTOCOL = read_json(Path(__file__).with_name("protocol.json"))
DENSE = PROTOCOL["dense"]
MODEL = DENSE["model"]
REVISION = SOURCES["models"][MODEL]["revision"]
type Config = dict[str, Any]
type Scores = dict[str, float]


def rank(scores: Mapping[str, float], top_k: int) -> Scores:
    return {doc: scores[doc] for doc in sorted(scores, key=lambda doc: (-scores[doc], doc))[:top_k]}


def fuse(
    sparse: Scores, dense: Scores, *, dense_weight: float, constant: int, top_k: int = 100
) -> Scores:
    scores: Scores = {}
    for weight, side in [(1 - dense_weight, sparse), (dense_weight, dense)]:
        for position, doc in enumerate(rank(side, 100), 1):
            scores[doc] = scores.get(doc, 0.0) + weight / (constant + position)
    return rank(scores, top_k)


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@contextmanager
def verified_model_view(model_id: str, cache_dir: Path, *, offline: bool = True) -> Iterator[Path]:
    """Yield a private directory containing exactly the manifest files, verified after copying."""
    from huggingface_hub import snapshot_download

    info = SOURCES["models"][model_id]
    snapshot = Path(
        snapshot_download(
            model_id,
            revision=info["revision"],
            cache_dir=cache_dir / "hf",
            allow_patterns=list(info["files"]),
            local_files_only=offline,
        )
    )
    with tempfile.TemporaryDirectory(prefix="ragbench-model-") as temporary:
        view = Path(temporary)
        for name, digest in info["files"].items():
            destination = view / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / name, destination)
            if sha256_file(destination) != digest:
                raise ValueError(f"model digest mismatch for {name}")
        yield view


class Encoder:
    def __init__(self, cache_dir: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        with verified_model_view(MODEL, cache_dir) as view:
            self.tokenizer = AutoTokenizer.from_pretrained(
                view, local_files_only=True, trust_remote_code=False
            )
            self.model = AutoModel.from_pretrained(
                view,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
            )
        self.model.to("cpu").eval()

    def lengths(self, texts: list[str]) -> list[int]:
        encoded = self.tokenizer(texts, padding=False, truncation=False, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(self, texts: list[str]) -> Any:
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=DENSE["max_length"],
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=1)


def index_key(data: Dataset, ids: list[str]) -> dict[str, Any]:
    return {
        "source_sha256": data.fingerprint,
        "document_order_sha256": reference_digest(ids),
        "representation": PROTOCOL["corpus_text"],
        "encoder": DENSE,
        "revision": REVISION,
        "files": SOURCES["models"][MODEL]["files"],
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
        "torch": version("torch"),
        "dtype": "float32",
        "torch_threads": PROTOCOL["timing"]["threads"],
        "device": "cpu",
    }


def _vectors(
    encoder: Encoder, texts: list[str], key: Config, cache: Path
) -> tuple[NDArray[np.float32], list[int]]:
    key_hash = reference_digest(key)
    target = cache / "dense" / (key_hash + ".npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as items:
            vectors = items["vectors"]
            lengths = items["lengths"]
            if (
                str(items["key"]) != key_hash
                or vectors.dtype != np.float32
                or vectors.shape != (len(texts), encoder.model.config.hidden_size)
                or lengths.shape != (len(texts),)
                or str(items["checksum"]) != _array_digest(vectors, lengths)
            ):
                raise ValueError("invalid or stale dense cache")
        return vectors, lengths.tolist()
    vectors_list, counts = [], []
    step = DENSE["document_batch_size"]
    for i in range(0, len(texts), step):
        batch = texts[i : i + step]
        counts.extend(encoder.lengths(batch))
        vectors_list.append(encoder.encode(batch).numpy())
    vectors = np.concatenate(vectors_list)
    lengths = np.asarray(counts, dtype=np.int64)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        item = Path(temp) / "vectors.npz"
        np.savez(
            item,
            key=np.asarray(key_hash),
            checksum=np.asarray(_array_digest(vectors, lengths)),
            lengths=lengths,
            vectors=vectors,
        )
        item.replace(target)
    return vectors, counts


def _array_digest(vectors: NDArray[np.float32], lengths: NDArray[np.int64]) -> str:
    import hashlib

    digest = hashlib.sha256(vectors.tobytes())
    digest.update(lengths.tobytes())
    return digest.hexdigest()


class Index:
    def __init__(self, data: Dataset, encoder: Encoder, cache_dir: Path):
        import torch
        from rank_bm25 import BM25Okapi

        self.ids = sorted(data.documents)
        texts = [data.documents[d]["title"] + "\n" + data.documents[d]["text"] for d in self.ids]
        self.base = BM25Okapi([lexical_tokens(t) for t in texts])
        self.encoder = encoder
        self.key = index_key(data, self.ids)
        vectors, self.lengths = _vectors(encoder, texts, self.key, cache_dir)
        self.vectors = torch.from_numpy(vectors)
        self.sparse_models: dict[str, Any] = {}

    def configure(self, configs: Sequence[Config]) -> None:
        for config in configs:
            if config["kind"] == "bm25":
                model = copy.copy(
                    self.base
                )  # IDF, term counts and length statistics are independent of k1/b.
                model.k1, model.b = config["k1"], config["b"]
                self.sparse_models[config["id"]] = model

    def ranking(self, values: NDArray[Any], top_k: int) -> Scores:
        # Corpus rows are in ascending ID order. Stable sort preserves that order on exact ties.
        indices = np.argsort(-values, kind="stable")[:top_k]
        return {self.ids[i]: float(values[i]) for i in indices}


def retrieve(query: str, config: Config, index: Index, top_k: int = 100) -> Scores:
    if config["kind"] == "bm25":
        model = index.sparse_models[config["id"]]
        return index.ranking(model.get_scores(lexical_tokens(query)), top_k)
    if config["kind"] == "dense":
        vector = index.encoder.encode([query])[0]
        return index.ranking(index.vectors.mv(vector).numpy(), top_k)
    if config["kind"] == "rrf":
        return fuse(
            retrieve(query, config["sparse"], index, 100),
            retrieve(query, DENSE, index, 100),
            dense_weight=config["dense_weight"],
            constant=config["constant"],
            top_k=top_k,
        )
    raise ValueError("unknown retrieval kind")
