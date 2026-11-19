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

## Generated SQuAD answers

Generate on the 200 held-out project questions only. Three configurations use Google FLAN-T5-small at revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`. Closed-book sees only the common instruction and question. BM25 top 1 and hybrid top 3 use the frozen evaluation rankings. Concatenate original paragraph bodies in rank order with two newlines. The prompt consists of “Answer the question with a short phrase.”, an optional “Context:” block, then “Question: {question}” and “Answer:” on separate lines. Gold labels are used after inference. [FLAN-T5-small](https://huggingface.co/google/flan-t5-small/tree/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab)

Encode the complete prompt with the pinned tokenizer, including EOS. Limit the whole input to 512 tokens and joined context to 384 tokens without special tokens. Retain whole ranked paragraphs until the next one exceeds either allowance. For that paragraph, search tokenizer character boundaries for a fitting original-text prefix and stop. Reject a question if the instruction and complete question alone exceed the full budget. Save actual prompts, encoder token IDs and character spans as half-open Unicode code point offsets into the original paragraph body. Separators count toward the context allowance as tokenized. Closed-book omits the context label entirely.

Use model evaluation mode, synchronous CPU float32, four intra-op threads, greedy decoding, one beam, batch one and at most 48 new tokens. The generated token count excludes the initial decoder start and includes EOS when generated. One authored warmup precedes measurement. Each successful row times a single model.generate call after prompt tokenization and before decoding. Answer latency excludes retrieval. Retrieval and answer timings are separate measurements on one machine. A generation or model-loading failure after preflight produces a separate error record. Preflight errors abort before model loading. Any missing response prevents completion.

Compare normalized EM and bag-of-token F1 against all accepted aliases using the maximum over references independently for each metric. Check every query score and both aggregate means against the preserved SQuAD v1.1 evaluation code with absolute tolerance 1e-12. “Gold passage supplied” means a nonempty supplied excerpt originates from a positively judged paragraph. “Answer string in context” means at least one nonempty, whitespace-stripped, casefolded reference is a literal substring of a single actual excerpt after casefolding. It can match inside another word. Neither indicator measures semantic faithfulness. FLAN-T5 may have seen SQuAD-related material during training. The sampled public validation questions provide no assurance of contamination-free testing. SQuAD excerpts, questions, references and article titles are reproduced and excerpted under CC BY-SA 4.0 with attribution to Rajpurkar et al. [Upstream scorer and license](../benchmarks/vendor/NOTICE.md) · [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Paired comparisons and release checks

For each dataset and task, the primary family compares every alternative to the fixed baseline on nDCG@10 for retrieval or token F1 for answers. Other pairs on the primary metric form their own exploratory family. For each secondary metric, all pairs within dataset and task form a single exploratory family. Holm-adjust the two-sided sign-flip p-values within each family. Each unordered pair appears once per metric with baseline and candidate explicit. To view the opposite direction, swap means and wins/losses, negate the mean difference and transform [L,U] into [−U,−L]. The two-sided p-values and their family stay the same. Apply the tolerance to the displayed interval direction.

Pair by query ID. Resample individual NFCorpus queries and entire SQuAD articles, weighting by sampled query count. Bootstrap uncertainty treats the observed units as exchangeable; sign flips assume exchangeability of paired system labels at those units under the null. Report pairwise percentile intervals and the two-sided Monte Carlo test with plus-one correction, each with 10,000 draws at seed 20261119. These comparisons describe the fixed benchmark and do not establish a random population sample. The declared release gate compares hybrid top 3 against closed-book on SQuAD token F1 with loss tolerance 0.02. Pass requires the lower limit to reach −0.02; regression requires the upper limit below −0.02; other intervals are inconclusive. Passing this gate concerns the allowed loss only. [Statistical contract](evaluation-contract.md)

Full reproduction writes a new local evidence tree; both raw runners refuse to overwrite their original runs. To reuse the retained retrieval stage and run only generation and aggregation, run these commands from a clean checkout with benchmark dependencies installed. Choose a new output tree inside the repository and copy the retained retrieval directory into it.

```sh
mkdir reproduced-answers
cp -R results/retrieval reproduced-answers/retrieval
python scripts/download_data.py --model google/flan-t5-small
python -m benchmarks.run_answers --retrieval reproduced-answers/retrieval --output reproduced-answers/answers
python -m benchmarks.release --results reproduced-answers
python -m benchmarks.verify --results reproduced-answers
```

Audit the published release from its public, checksummed evidence. Replay reconstructs development choices, rankings, metrics, diagnostics, token IDs, timing summaries, comparisons, gates and explorer rows. This requires neither raw corpora nor neural weights. SHA-256 binds contents and does not establish authorship or attest hardware timings. Training exposure, original source-document text and hardware measurements cannot be independently inferred from hashes. Token IDs permit checking the exact supplied text and decoded predictions. The immutable original dataset digest and declared source bounds support a separate corpus comparison.

```sh
python -m pip install -e . pytrec-eval-terrier==0.5.11 tokenizers==0.23.3
python -m benchmarks.verify --gate
```

The release manifest uses repository-relative paths with byte sizes and SHA-256 for every evidence file, the pinned sources, dependency lock and scoring/tokenizer references. The gate exits 0 for pass, 1 for regression, 2 for inconclusive and 3 for invalid evidence. The explorer includes all evaluation query IDs and top-ten ranking previews with each row's document ID, rank, score and grade, treating an unjudged document as zero gain. Full top-100 rankings are linked for all eight retrieval runs.

## Measurement input binding

Before future model initialization, the runner checks the evaluation references, groups, question IDs and retrieval records against the verified SQuAD source and the frozen retrieval evidence. It writes `answers/input-binding.json` with the chosen split, digests of the ordered evaluation IDs, questions, article assignments, aliases, qrels, selected configurations, both full retrieval runs and the candidate paragraph archive. The `origin` object identifies the measurement and binding source commits and the stage when the binding was made. SHA-256 of canonical JSON identifies semantic inputs. SHA-256 of file bytes identifies the selected configuration file and ranking files. Each answer evaluation records the binding file digest.

The published 600 predictions predate this binding. Their `origin.stage` is `retrospective_source_verification`. The backfill checked the original committed prediction, reference, runtime and ranking files against their original evidence commit, then reconstructed the binding from the pinned SQuAD source and frozen ranks. It did not rerun the model. The original measurement source commit remains in the runtime record.

The file `answers/retrieved-passages.json` maps only the SQuAD paragraph IDs appearing in these top-one/top-three retrieval inputs to complete original paragraph bodies. These extracts retain CC BY-SA 4.0 attribution to Rajpurkar et al. Offline verification replays the same prompt builder against those original bodies in frozen ranking order and checks the exact expected spans, prompt, token counts and input IDs for each query and configuration. It also verifies that a sub-48-token output ends at EOS, that EOS appears only at the end when present, and that both timing runtimes supply the pinned package versions and synchronous CPU settings. [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
