"""Construct exact bounded text inputs and run greedy FLAN-T5 on CPU."""

import hashlib
import time
from pathlib import Path
from typing import Any

from .data import SOURCES
from .retrieval import verified_model_view

MODEL = "google/flan-t5-small"
INSTRUCTION = "Answer the question with a short phrase."
CONFIGS = {
    "closed-book": (None, 0),
    "bm25-top1": ("bm25-tuned", 1),
    "hybrid-top3": ("hybrid-tuned", 3),
}
PROTOCOL = {
    "model": MODEL,
    "revision": SOURCES["models"][MODEL]["revision"],
    "instruction": INSTRUCTION,
    "max_input_tokens": 512,
    "max_context_tokens": 384,
    "max_new_tokens": 48,
    "do_sample": False,
    "num_beams": 1,
    "batch_size": 1,
    "device": "cpu",
    "dtype": "float32",
    "threads": 4,
    "timing": "warm batch-one model.generate only; CPU synchronous; excludes prompt construction, retrieval and text decoding",
}


def render(question, excerpts):
    context = "\n\n".join(excerpts)
    block = "\nContext:\n" + context if context else ""
    return f"{INSTRUCTION}{block}\nQuestion: {question}\nAnswer:"


def build_prompt(tokenizer, question: str, passages: list[tuple[str, str]]) -> dict[str, Any]:
    """Retain ranked paragraph-body prefixes with half-open original character offsets."""

    def encode(text, *, special=True):
        return tokenizer(text, add_special_tokens=special, truncation=False, verbose=False)[
            "input_ids"
        ]

    def assemble(spans):
        texts = [s["text"] for s in spans]
        prompt = render(question, texts)
        ids = encode(prompt)
        context_tokens = len(encode("\n\n".join(texts), special=False)) if texts else 0
        return {
            "prompt": prompt,
            "input_ids": ids,
            "context_tokens": context_tokens,
            "spans": spans,
        }

    def fits(item):
        return len(item["input_ids"]) <= 512 and item["context_tokens"] <= 384

    current = assemble([])
    if not fits(current):
        raise ValueError("question and instruction exceed 512 model tokens")
    for rank, (docid, text) in enumerate(passages, 1):

        def extend(end, *, docid=docid, text=text, rank=rank, previous=current["spans"]):
            span = {
                "docid": docid,
                "rank": rank,
                "start": 0,
                "end": end,
                "source_chars": len(text),
                "text": text[:end],
            }
            return assemble([*previous, span])

        if not text.strip():
            raise ValueError("empty retrieved passage")
        full = extend(len(text))
        if fits(full):
            current = full
            continue
        # Use actual tokenizer offsets for source prefixes. Re-encode the composed
        # prompt and context at each step: tokenization is not generally additive.
        offsets = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )["offset_mapping"]
        ends = sorted({end for _, end in offsets if end > 0})
        low, high = 0, len(ends) - 1
        while low <= high:
            mid = (low + high) // 2
            partial = extend(ends[mid])
            if fits(partial):
                current = partial
                low = mid + 1
            else:
                high = mid - 1
        break
    return current


def context_diagnostics(spans, qrels, answers):
    """Literal casefolded nonempty alias substring, within one supplied span."""
    return {
        "gold_passage_supplied": any(qrels.get(s["docid"], 0) > 0 for s in spans),
        "answer_string_in_context": any(
            alias.strip().casefold() in span["text"].casefold()
            for span in spans
            for alias in answers
            if alias.strip()
        ),
    }


class Generator:
    def __init__(self, cache: Path):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        with verified_model_view(MODEL, cache) as view:
            self.tokenizer = AutoTokenizer.from_pretrained(
                view, local_files_only=True, trust_remote_code=False
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                view,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
            )
        fingerprint = hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        if fingerprint != "170798cc83dec7e6830d51178ea247259a174975383450f7cd1287a491b55004":
            raise ValueError("runtime tokenizer differs from archived effective FLAN tokenizer")
        self.model.to("cpu").eval()

    def predict(self, question, passages):
        import torch

        row = build_prompt(self.tokenizer, question, passages)
        inputs = torch.tensor([row["input_ids"]], device="cpu", dtype=torch.long)
        mask = torch.ones_like(inputs)
        with torch.inference_mode():
            start = time.perf_counter_ns()
            result = self.model.generate(
                input_ids=inputs,
                attention_mask=mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=48,
            )
            latency = time.perf_counter_ns() - start
        # T5 includes a decoder start token preceding its generated sequence.
        ids = result[0, 1:].tolist()
        return {
            **row,
            "input_tokens": len(row["input_ids"]),
            "output_ids": ids,
            "output_tokens": len(ids),
            "prediction": self.tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=True),
            "latency_ns": latency,
        }
