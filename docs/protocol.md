# Retrieval experiment protocol

Each collection uses the same four retrieval systems. Parameters are selected per collection. The runner writes both collections’ query IDs, judgments and SQuAD answers before initializing the encoder. It writes both development selections before processing held-out queries. Gold answers never enter retrieval. A successful run creates a new output directory and refuses to replace existing evidence.

## Source populations

- **NFCorpus (BEIR)** uses all documents in the BEIR abstract collection, all queries judged by the official development and test qrels, and graded judgments. Training qrels are unused. The upstream academic-use terms and the BEIR mirror’s CC BY-SA 4.0 metadata differ. Public evidence includes identifiers, ranks, scores and judgments. Original NFCorpus text stays in the local data cache. [Boteva et al., NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) · [BEIR mirror](https://huggingface.co/datasets/BeIR/nfcorpus)
- **SQuAD v1.1** uses the full public validation paragraph corpus. Each paragraph ID is `squad:<zero-based original article index>:<zero-based original paragraph index>`. Sort articles by SHA-256 of UTF-8 `ragbench-v1-article:` plus the exact title. The first ten form the development pool. The remaining articles form the evaluation pool. Sort each pool’s original question IDs by SHA-256 of UTF-8 `ragbench-v1-question:` plus the original ID and take the first 50 or 200 respectively. Preserve original text, answer aliases and article titles. Each question has one positive paragraph. These are project splits from public validation. Prior model exposure is possible. SQuAD data, including answer labels and article titles, is attributed to Rajpurkar et al. under CC BY-SA 4.0. [Dataset](https://rajpurkar.github.io/SQuAD-explorer/) · [License](https://creativecommons.org/licenses/by-sa/4.0/)

[Source and model digests](../benchmarks/sources.json) identify exact inputs. Every source download uses verified HTTPS. All source bytes and all declared model snapshot files must match their SHA-256 values before use. The model loader copies only manifest-listed files into a private temporary directory, checks each copy, and loads the tokenizer and weights from that directory. The NFCorpus reader reads bounded ZIP members directly in memory. No member is extracted.

## Systems and development selection

The document text for every system is the original title, one newline, and the original body. BM25Okapi from rank-bm25 0.2.2 uses Unicode word tokens from Python’s `re.findall(r"\w+", text.lower())`, repeated query terms, and the library’s default negative-IDF floor (epsilon 0.25). The fixed system uses k1 1.5 and b 0.75. Tune BM25 over k1 0.6, 1.2, 1.8 and b 0, 0.5, 1. [rank-bm25](https://github.com/dorianbrown/rank_bm25)

The MiniLM encoder uses the pinned model in evaluation mode, float32 CPU inference, tokenizer special tokens and at most 256 wordpieces including special tokens. Attention-mask mean-pool final hidden states and L2-normalize. Exact dot products over all documents rank normalized embeddings. Document batches contain at most 16 texts. Queries use single-item batches. Record pre-truncation token counts; document lengths are summarized and query lengths retained by ID. The dense cache key binds source bytes, sorted corpus order, preprocessing, encoder, tokenizer and dependency versions. Stored arrays have an integrity checksum. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

Weighted reciprocal rank fusion uses the top 100 documents from the development-selected BM25 and the top 100 from MiniLM. For each document, add (1 − w)/(c + sparse rank) and w/(c + dense rank), with one-based ranks and a zero contribution on an absent side. Search w 0.25, 0.5, 0.75 and c 10, 60. All systems rank by descending score and ascending document ID on ties. Select each grid’s greatest mean development nDCG@10, breaking ties by ascending configuration ID. Preserve the full tuning table, its top-ten supporting rankings, and the selected configuration before measuring held-out queries. [Machine-readable settings](../benchmarks/protocol.json)

Each held-out evaluation includes the top 100 original scores, canonical ranks and per-query scores for six metrics. Recompute linear-gain nDCG@10, recall@5/10/100, reciprocal rank@10 and precision@10 independently with pytrec-eval-terrier 0.5.11. The independent evaluator uses strictly descending surrogate scores for canonical ranks and a top-ten-only ranking for reciprocal rank. Recompute and compare every development grid’s per-query and mean nDCG and all held-out per-query values and means with an absolute tolerance of 1e-12. The recorded check gives the real maximum difference including floating-point roundoff. [Evaluation contract](evaluation-contract.md) · [Independent trec_eval binding](https://github.com/terrierteam/pytrec_eval)

## Timing

Sort held-out query IDs by SHA-256 of UTF-8 `ragbench-v1-latency:` plus ID and take the first 32. For each system and selected query, issue one unmeasured warmup followed by five serial single-query measurements. Use a monotonic nanosecond clock, four PyTorch intra-op threads, synchronous CPU execution and fresh query processing on each call. Start before query tokenization and finish after final ranked scores. Hybrid timings include both base retrievers and fusion. Models, corpus tokenization and document indexes are already loaded. p50 and p95 use NumPy’s linear percentile over all 160 measured durations. These measurements describe one machine and one corpus per run.

## Reproduction

From a clean Git checkout with Python 3.13, install the local evaluator and benchmark dependencies into an independent virtual environment. The lock captures the measured environment. Model weights and raw corpora remain under the ignored cache. Run the following from the repository root.

```sh
python -m pip install -e . -r requirements-benchmark.txt
sh scripts/reproduce.sh
```

For a populated cache, verify and rerun offline. Supply a new output directory.

```sh
python scripts/download_data.py --offline
python -m benchmarks.run_retrieval --output reproduced/offline-retrieval
```

Replay just the independent checks on the published evidence without loading models or downloading datasets.

```sh
python -m pip install -e . pytrec-eval-terrier==0.5.11
python -m benchmarks.crosscheck results/retrieval
```

The TREC files have six tab-separated columns giving query ID, Q0, document ID, canonical rank, original score and system tag. Development files contain ten ranks per query and evaluation files contain 100. Dataset directories contain `selection.json`, reference JSON, `tuning.json`, `selected.json`, `dev-runs/`, `runs/`, `evaluations/`, `diagnostics/`, `tokenization.json`, `latency.json`, `crosscheck.json` and `provenance.json`. The shared runtime manifest records hardware, versions, threads, dtype and implementation commit.
