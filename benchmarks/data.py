"""Pinned data ingestion. Raw text stays in the local cache."""

import csv
import hashlib
import io
import json
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from ragbench.io import identifier, read_json, validate_qrels

SOURCES = read_json(Path(__file__).with_name("sources.json"))
FILES = {"nfcorpus-beir": "nfcorpus.zip", "squad-v1.1": "squad-dev-v1.1.json"}
MAX_MEMBER = 16 * 1024 * 1024
type Qrels = dict[str, dict[str, int]]


class Document(TypedDict):
    title: str
    text: str


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    fingerprint: str  # SHA-256 of the complete original source file
    dev_split: str
    eval_split: str
    documents: dict[str, Document]
    dev_queries: dict[str, str]
    eval_queries: dict[str, str]
    dev_qrels: Qrels
    eval_qrels: Qrels
    dev_groups: dict[str, str] | None
    eval_groups: dict[str, str] | None
    dev_answers: dict[str, list[str]] | None
    eval_answers: dict[str, list[str]] | None
    article_pools: dict[str, list[str]] | None


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_source(path: Path, record: dict[str, Any]) -> None:
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"source size or digest mismatch: {path.name}")


def get_source(name: str, cache_dir: Path, *, offline: bool = True) -> Path:
    filename = FILES[name]
    record = SOURCES["sources"][filename]
    target = cache_dir / filename
    if target.exists():
        check_source(target, record)
    elif offline:
        raise FileNotFoundError(f"missing verified source {filename}; run scripts/download_data.py")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as temp:
            item = Path(temp) / filename
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--max-filesize",
                    str(record["bytes"]),
                    "--output",
                    str(item),
                    record["url"],
                ],
                check=True,
            )
            check_source(item, record)
            item.replace(target)
    return target


def validate_dataset(data: Dataset) -> None:
    if data.dev_queries.keys() & data.eval_queries.keys():
        raise ValueError("query splits overlap")
    if (
        data.dev_groups
        and data.eval_groups
        and set(data.dev_groups.values()) & set(data.eval_groups.values())
    ):
        raise ValueError("group splits overlap")
    for queries, judgments, groups, answers in [
        (data.dev_queries, data.dev_qrels, data.dev_groups, data.dev_answers),
        (data.eval_queries, data.eval_qrels, data.eval_groups, data.eval_answers),
    ]:
        if judgments.keys() != queries.keys():
            raise ValueError("query and judgment IDs differ")
        if any(
            doc not in data.documents for docs in validate_qrels(judgments).values() for doc in docs
        ):
            raise ValueError("judgment refers to missing document")
        if groups is not None and groups.keys() != queries.keys():
            raise ValueError("group IDs do not cover split")
        if answers is not None and answers.keys() != queries.keys():
            raise ValueError("answer IDs do not cover split")


def _read_rows(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    rows = {}
    for line in archive.read("nfcorpus/" + name).decode("utf8").splitlines():
        row = json.loads(line)
        key = identifier(row["_id"], "record ID")
        if key in rows:
            raise ValueError("duplicate source record ID")
        rows[key] = row
    return rows


def parse_beir(path: Path, digest: str) -> Dataset:
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        if len(members) > 32 or sum(i.file_size for i in members) > 32 * 1024 * 1024:
            raise ValueError("archive exceeds bounds")
        names: set[str] = set()
        for member in members:
            name = member.filename
            if (
                re.search(r"[\\:\x00]", name)
                or any(s in ("", ".", "..") for s in name.rstrip("/").split("/"))
                or member.file_size > MAX_MEMBER
                or name in names
            ):
                raise ValueError("unsafe archive member")
            names.add(name)
        docs: dict[str, Document] = {
            key: {"title": row["title"], "text": row["text"]}
            for key, row in _read_rows(z, "corpus.jsonl").items()
        }
        all_queries = _read_rows(z, "queries.jsonl")
        splits: list[Qrels] = []
        for split in ("dev", "test"):
            judgments: Qrels = {}
            rows = csv.reader(
                io.StringIO(z.read(f"nfcorpus/qrels/{split}.tsv").decode("utf8")), delimiter="\t"
            )
            if next(rows) != ["query-id", "corpus-id", "score"]:
                raise ValueError("unknown BEIR judgment header")
            for qid, doc, grade in rows:
                if qid not in all_queries:
                    raise ValueError("unknown BEIR query ID")
                entries = judgments.setdefault(qid, {})
                if doc in entries:
                    raise ValueError("duplicate BEIR judgment")
                entries[doc] = int(grade)
            splits.append(judgments)
    dev, test = splits
    data = Dataset(
        "nfcorpus-beir",
        digest,
        "official-dev",
        "official-test",
        docs,
        {q: all_queries[q]["text"] for q in dev},
        {q: all_queries[q]["text"] for q in test},
        dev,
        test,
        None,
        None,
        None,
        None,
        None,
    )
    validate_dataset(data)
    return data


def hash_order(prefix: str, item: str) -> str:
    return hashlib.sha256((prefix + item).encode("utf8")).hexdigest()


def parse_squad(
    raw: dict[str, Any],
    digest: str,
    *,
    dev_articles: int = 10,
    dev_size: int = 50,
    eval_size: int = 200,
) -> Dataset:
    docs: dict[str, Document] = {}
    articles: dict[str, list[str]] = {}
    questions: dict[str, str] = {}
    groups: dict[str, str] = {}
    answers: dict[str, list[str]] = {}
    judgments: Qrels = {}
    for a, article in enumerate(raw["data"]):
        title = identifier(article["title"], "article title")
        if title in articles:
            raise ValueError("duplicate article title")
        articles[title] = []
        for p, para in enumerate(article["paragraphs"]):
            doc = f"squad:{a}:{p}"
            docs[doc] = {"title": title, "text": para["context"]}
            for qa in para["qas"]:
                qid = identifier(qa["id"], "question ID")
                if qid in questions:
                    raise ValueError("duplicate SQuAD question ID")
                articles[title].append(qid)
                groups[qid], questions[qid], judgments[qid] = title, qa["question"], {doc: 1}
                answers[qid] = [answer["text"] for answer in qa["answers"]]
                if not answers[qid]:
                    raise ValueError("missing SQuAD answer")
    ordered = sorted(articles, key=lambda title: hash_order("ragbench-v1-article:", title))
    pools = {"dev": ordered[:dev_articles], "eval": ordered[dev_articles:]}

    def choose(pool: list[str], n: int) -> list[str]:
        ids = [qid for title in pool for qid in articles[title]]
        if len(ids) < n:
            raise ValueError("insufficient questions in article pool")
        return sorted(ids, key=lambda q: hash_order("ragbench-v1-question:", q))[:n]

    dev = choose(pools["dev"], dev_size)
    test = choose(pools["eval"], eval_size)

    def subset[T](mapping: dict[str, T], keys: list[str]) -> dict[str, T]:
        return {key: mapping[key] for key in keys}

    data = Dataset(
        "squad-v1.1",
        digest,
        f"project-hash-v1-dev{dev_size}",
        f"project-hash-v1-eval{eval_size}",
        docs,
        subset(questions, dev),
        subset(questions, test),
        subset(judgments, dev),
        subset(judgments, test),
        subset(groups, dev),
        subset(groups, test),
        subset(answers, dev),
        subset(answers, test),
        pools,
    )
    validate_dataset(data)
    return data


def load_dataset(name: str, cache_dir: Path | str) -> Dataset:
    path = get_source(name, Path(cache_dir))
    digest = SOURCES["sources"][path.name]["sha256"]
    if name == "squad-v1.1":
        return parse_squad(read_json(path), digest)
    return parse_beir(path, digest)
