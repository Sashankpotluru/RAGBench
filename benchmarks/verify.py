"""Verify release descriptors and replay from public evidence without neural inference."""

import argparse
import re
import sys
from pathlib import Path

from ragbench.evaluation import reference_digest
from ragbench.io import read_json

from .data import sha256_file
from .generation import PROTOCOL as ANSWER_PROTOCOL
from .generation import build_prompt
from .release import ROOT, derive, exact_ids


def require(condition, label):
    if not condition:
        raise ValueError(f"inconsistent evidence: {label}")


def same(actual, expected, label):
    require(reference_digest(actual) == reference_digest(expected), label)


def confined(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative:
        raise ValueError(f"unsafe evidence path: {relative}")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"evidence outside repository: {relative}")
    return result


def descriptor(root, path):
    relative = str(path.relative_to(root))
    item = confined(root, relative)
    return {"path": relative, "bytes": item.stat().st_size, "sha256": sha256_file(item)}


def check_manifest(root, manifest, expected):
    exact_ids(expected, [d["path"] for d in manifest], "release manifest coverage")
    for d in manifest:
        path = confined(root, d["path"])
        same(d, descriptor(root, path), f"hash or size for {d['path']}")


def check_coverage(root, results, inventory):
    require(not any(p.is_symlink() for p in results.rglob("*")), "symlink in evidence tree")
    expected = {p for p in inventory if (root / p).is_relative_to(results)}
    observed = {
        str(p.relative_to(root))
        for p in results.rglob("*")
        if p.is_file() and p != results / "release.json"
    }
    exact_ids(expected, observed, "results file coverage")


def runtime_check(machine):
    fields = {
        "architecture",
        "device",
        "dtype",
        "logical_cpus",
        "os",
        "processor",
        "python",
        "source_commit",
        "synchronization",
        "tokenizers_parallelism",
        "torch_interop_threads",
        "torch_threads",
        "versions",
    }
    require(isinstance(machine, dict) and set(machine) == fields, "complete runtime fields")
    require(
        all(
            isinstance(machine[key], str) and machine[key].strip()
            for key in ["architecture", "os", "processor", "python", "source_commit"]
        ),
        "runtime platform fields",
    )
    require(
        re.fullmatch(r"3\.(12|13)\.\d+", machine["python"]) is not None
        and re.fullmatch(r"[0-9a-f]{40}", machine["source_commit"]) is not None,
        "runtime identity",
    )
    require(
        all(
            type(machine[key]) is int and machine[key] > 0
            for key in ["logical_cpus", "torch_interop_threads", "torch_threads"]
        ),
        "runtime thread counts",
    )
    require(
        machine["device"] == "cpu"
        and machine["dtype"] == "float32"
        and machine["torch_threads"] == 4
        and machine["tokenizers_parallelism"] is False
        and machine["synchronization"] == "CPU operations synchronous",
        "runtime settings",
    )
    packages = {
        "numpy",
        "rank-bm25",
        "torch",
        "transformers",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
        "pytrec-eval-terrier",
    }
    versions = machine["versions"]
    require(
        isinstance(versions, dict)
        and set(versions) == packages
        and all(
            isinstance(v, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[-+.a-zA-Z0-9]+)?", v)
            for v in versions.values()
        ),
        "runtime dependency versions",
    )
    locks = dict(
        line.split("==", 1)
        for line in (ROOT / "requirements-benchmark.txt").read_text().splitlines()
        if "==" in line and not line.startswith("#")
    )
    require(all(locks.get(pkg) == versions[pkg] for pkg in packages), "runtime dependency pins")


def replay_tokenizer(root):
    from tokenizers import Tokenizer

    tokenizer = root / "benchmarks/vendor/flan-tokenizer.json"
    require(
        sha256_file(tokenizer)
        == "170798cc83dec7e6830d51178ea247259a174975383450f7cd1287a491b55004",
        "effective pinned FLAN tokenizer",
    )
    return Tokenizer.from_file(str(tokenizer))


def audit_answers(folder, root, questions, refs):
    runtime_check(read_json(folder / "runtime.json"))
    same(read_json(folder / "protocol.json"), ANSWER_PROTOCOL, "generation protocol")
    same(read_json(folder / "failures.json"), [], "answer failures")
    exact_ids(refs, questions, "evaluation questions")
    scorer = root / "benchmarks/vendor/squad"
    source = read_json(scorer / "source.json")
    require(
        sha256_file(scorer / "compute_score.py")
        == source["sha256"]
        == "6c27d9393600a8e6c665adb2dc2c966ae7c7a70960064eb032711dc58914956a"
        and sha256_file(scorer / "LICENSE") == source["license"]["sha256"],
        "upstream scorer",
    )


def audit_prediction(row, question, passages, tokenizer):
    require(isinstance(question, str) and question.strip(), "question")

    def encode(text, *, add_special_tokens=True, **kwargs):
        result = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": result.ids, "offset_mapping": result.offsets}

    expected = build_prompt(encode, question, passages)
    same({key: row[key] for key in expected}, expected, "expected prompt and supplied spans")
    same(row["input_tokens"], len(expected["input_ids"]), "input token count")
    output = row["output_ids"]
    require(
        isinstance(output, list)
        and all(type(i) is int and 0 <= i < tokenizer.get_vocab_size() for i in output)
        and type(row["output_tokens"]) is int
        and row["output_tokens"] == len(output)
        and 1 <= len(output) <= 48
        and row["prediction"] == tokenizer.decode(output, skip_special_tokens=True)
        and type(row["latency_ns"]) is int
        and row["latency_ns"] > 0,
        "generation output or latency",
    )
    eos = 1  # Pinned FLAN-T5-small generation config.
    require(
        eos not in output[:-1]
        and (output[-1] == eos or len(output) == ANSWER_PROTOCOL["max_new_tokens"]),
        "greedy stopping condition",
    )


def verify(root, results):
    confined(root, str(results.relative_to(root)))
    report = read_json(confined(root, str((results / "release.json").relative_to(root))))
    inventory = {d["path"] for d in report["manifest"]}
    check_manifest(root, report["manifest"], inventory)
    check_coverage(root, results, inventory)
    release, derived, expected = derive(root, results)
    require(inventory == expected, "declared file inventory")
    same({k: v for k, v in report.items() if k != "manifest"}, release, "release overview")
    for name, value in derived.items():
        same(read_json(results / name), value, name)
    return release


def gate_exit(row):
    return {"pass": 0, "regression": 1, "inconclusive": 2}[row["gate"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--gate", action="store_true", help="also apply the declared release gate")
    args = parser.parse_args()
    try:
        release = verify(ROOT, args.results.resolve())
        print(f"Verified retrieval, generation and paired comparisons; gate {release['gates'][0]}")
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        sys.exit(3)
    if args.gate:
        sys.exit(gate_exit(release["gates"][0]))
