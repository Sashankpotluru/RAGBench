# Evaluation contract

RAGBench scores recorded predictions on Python 3.12 and 3.13. NumPy supplies the resampling routines. Scoring and comparison run offline after installation.

## Inputs

Each input file must be UTF-8 and at most 64 MiB (67,108,864 bytes). Query, document and group identifiers are nonblank strings. IDs are case-sensitive. JSON object keys must be unique at every level. Duplicate text records for the same query and document are errors, even when their values agree. JSON numbers must be finite. Grade and score fields reject booleans.

| Input | Representation |
| --- | --- |
| Retrieval references | JSON object mapping query ID to document ID to nonnegative integer grade. Also accepts the BEIR header `query-id corpus-id score` and three columns, or four TREC columns `qid iteration docid grade`. Each reference query needs at least one judgment. |
| Retrieval predictions | JSON object mapping query ID to document ID to finite score, or six TREC columns `qid iteration docid rank score tag`. JSON integer scores must be exactly representable in IEEE 754 binary64. JSON scores with a decimal point or exponent and TREC score fields are read at binary64 precision. Rank must be a positive integer. Ranking uses descending score and then ascending document ID. The iteration and tag fields carry no scoring information. |
| Answer references | JSON object mapping query ID to a nonempty list of strings. An empty string can be an accepted alias. |
| Answer predictions | JSON object mapping query ID to one answer string. |
| Groups | Optional JSON object mapping every reference query ID to one nonblank group ID. Supply article IDs for SQuAD and omit groups for query-level NFCorpus sampling. |

Whitespace separates text-format columns. A file named `.json` must parse as JSON. Initial object/array delimiters also select the JSON parser regardless of extension. The reference population must contain at least one query. Both prediction formats may omit queries or provide an empty run. Unexpected query IDs fail validation.

## Metrics

All means give every reference query equal weight. Every metric uses a 0–1 scale.

| Retrieval column | Definition |
| --- | --- |
| `ndcg@10` | Sum of the first ten grades, each divided by log₂(rank + 1), divided by the best possible such sum from all judgments. Gains are linear. Zero ideal gain yields zero. |
| `recall@5`, `recall@10`, `recall@100` | Count of positive-grade documents retrieved through the cutoff divided by the number of positive-grade documents among all judgments for the query. Zero relevant documents yields zero. |
| `mrr@10` | Reciprocal of the first positive-grade rank through rank ten. Zero when no hit is present. The evaluation mean is mean reciprocal rank. |
| `precision@10` | Number of positive-grade documents among the first ten results divided by ten, including for short lists. |

Unjudged documents retain their rank and receive zero gain. Missing query runs receive zeros in all six columns.

Answer scoring lowercases the text, removes ASCII punctuation, replaces English articles (`a`, `an`, `the` at word boundaries) with spaces, and collapses whitespace. For each query, `exact_match` is 1 if any normalized alias matches. `token_f1` is the best multiset token F1 across aliases. F1 is zero when there is no token overlap, including two empty normalized strings. Those two empty strings have exact match 1. An omitted prediction key receives zero for both metrics, including when a reference alias normalizes to an empty string. An explicitly supplied empty string goes through normal scoring. The scoring convention follows the Stanford SQuAD v1.1 reference evaluator, mirrored by Hugging Face at revision `a5db2ad54aaeadf15c7aa1f056bc6f7bbeeb7e5f`, SHA-256 `6c27d9393600a8e6c665adb2dc2c966ae7c7a70960064eb032711dc58914956a`. Its aggregate evaluator reports percentages. Divide by 100 to compare them with these means. [Pinned reference implementation](https://github.com/huggingface/evaluate/blob/a5db2ad54aaeadf15c7aa1f056bc6f7bbeeb7e5f/metrics/squad/compute_score.py)

## Version 1 artifact

| Field | Contents |
| --- | --- |
| `schema_version`, `scoring_version` | Integer `1` and string `ragbench-v1`. |
| `kind` | `retrieval` or `answers`. |
| `dataset_id`, `split_id` | Nonblank identity strings supplied by the caller. |
| `reference_sha256` | Lowercase 64-character SHA-256 digest of the parsed references. The CLI computes it. Python callers must pass the digest for their references. |
| `query_ids` | All query IDs, unique and sorted by ascending Unicode code point. |
| `per_query` | Object keyed by query ID. Each row contains exactly the columns for the selected kind. |
| `means` | Object containing the same columns. Each value equals the accurately summed per-query column divided by the query count, stored at full precision. |
| `groups` | Null for query sampling, or the exact query-to-group map for cluster sampling. |
| `config`, `provenance` | Finite JSON objects for caller-supplied configuration and evidence metadata. Both default to empty objects. |

All fields are required in a saved artifact. The validator rejects extra fields, different column sets, inconsistent means and invalid identities. It returns a canonicalized artifact. The digest uses UTF-8 bytes of Python JSON with sorted object keys, ASCII escapes, no spaces and preserved array order. This definition applies to parsed references, so a JSON and a TREC file expressing the same judgments produce the same reference digest.

## Paired comparisons

The candidate and baseline must have identical task, dataset, split, reference digest, scoring version, query IDs and grouping. Pair the query values by ID. Report the arithmetic mean of the candidate-minus-baseline differences, together with query-level wins, ties and losses. Configuration and provenance may differ. Inference requires at least two independent units, either distinct queries or distinct groups. Comparisons with dataset ID `squad-v1.1` require a complete article map for both retrieval and answer evaluations. For other correlated datasets, callers must provide their own sampling groups.

For a percentile bootstrap replicate, sample as many units as the original count with replacement. Each sampled group contributes all its member queries, including every repeated copy. Divide the sum of differences by the number of queries in that replicate. The 2.5th and 97.5th linearly interpolated percentiles give the **pairwise 95% interval**. The two-sided paired randomization test independently flips a fair sign for each unit. Count absolute permuted sums at least as large as the observed absolute sum and use `(count + 1) / (resamples + 1)`. Bootstrap totals and sign-flip sums use the exact common-denominator integer representation of the input query deltas. Each replicate mean is rounded once to binary64. Exactly tied sign-flip sums count as extreme, including under cancellation. Defaults are 10,000 draws for each procedure and seed 20261119. Each procedure uses its own PCG64 stream from NumPy SeedSequence spawning, with bootstrap first and sign flips second. Sampling units sort by ID before drawing.

With default tolerance 0.02, a lower interval bound at least −0.02 gives `pass`, an upper bound strictly below −0.02 gives `regression`, and the intervening cases give `inconclusive`. The gate depends only on the interval bounds and tolerance.

| Comparison fields | Contents |
| --- | --- |
| Identity | `schema_version`, `scoring_version`, `kind`, `dataset_id`, `split_id`, `reference_sha256`, `metric`, and SHA-256 digests of the complete validated evaluations in `baseline_sha256` and `candidate_sha256`. |
| Estimates | `baseline_mean`, `candidate_mean`, `mean_delta`, `ci95` as [lower, upper], and `interval_scope` equal to `pairwise`. |
| Sampling and counts | `resampling_unit` (`query` or `group`), `unit_count`, `query_count`, `wins`, `ties`, `losses`, `resamples`, `seed`. |
| Decision | `margin`, `gate`, raw `p_value`, Holm-adjusted `p_holm` and `family_size`. |

A standalone CLI comparison declares a family of one. For a larger predeclared family, collect every raw p-value, call `holm_adjust` once on that mapping, then record the returned value and total family size on each comparison. Holm uses the monotone step-down procedure with adjusted probabilities capped at 1.

## Commands

From a checkout with the package installed, the two synthetic fixture queries are enough for a paired comparison.

```sh
ragbench score --qrels tests/fixtures/graded.tsv --run tests/fixtures/ranking.trec \
  --dataset-id fixture --split-id example --output fixture.json
ragbench compare fixture.json fixture.json --metric 'ndcg@10' --margin .02 --output paired.json
```

Score a set of answer strings and declare their article membership with the following command.

```sh
ragbench answers --references references.json --predictions predictions.json \
  --dataset-id squad-v1.1 --split-id project-eval --groups articles.json --output answers.json
```

A valid comparison exits 0 for pass, 1 for regression or 2 for inconclusive, writing the full report in every case. File or validation errors print a diagnostic on stderr and return 3. Argument syntax errors use the standard argparse exit 2. All outputs use same-directory atomic replacement after serialization, and their parent directory must already exist.

## Python entry points

```python
from ragbench.io import read_qrels, read_run
from ragbench.metrics import score_retrieval, score_answers
from ragbench.evaluation import reference_digest, make_evaluation, validate_evaluation
from ragbench.compare import compare_evaluations, holm_adjust
```
