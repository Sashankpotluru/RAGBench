# Sources and licenses

Original RAGBench software is distributed under the [MIT license](LICENSE). The following materials retain their upstream terms.

| Material | Source and terms | Use in this release |
| --- | --- | --- |
| SQuAD v1.1 | Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev and Percy Liang. [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) · [Paper](https://aclanthology.org/D16-1264/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | The project sampled and regrouped public validation questions. SQuAD-derived questions, answer aliases, article titles, selected original paragraphs, truncated excerpts and their adaptations retain CC BY-SA 4.0. Paragraph text derives from Wikipedia. |
| NFCorpus | Boteva, Gholipour Ghalandari, Sokolov and Riezler. [Original collection and academic-use terms](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) · [BEIR mirror](https://huggingface.co/datasets/BeIR/nfcorpus) | The upstream terms and mirror license metadata differ. RAGBench distributes numeric and identifier evidence. Original document and query text stays in the local source cache. |
| all-MiniLM-L6-v2 | Sentence Transformers. [Pinned Apache 2.0 model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) | Encodes documents and questions. Weights are downloaded separately. |
| FLAN-T5-small | Google. [Pinned Apache 2.0 model](https://huggingface.co/google/flan-t5-small/tree/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab) | Generated answer records and effective tokenizer serialization. Weights are downloaded separately. |
| SQuAD evaluation code | Stanford SQuAD authors, via the [Hugging Face Evaluate mirror](https://github.com/huggingface/evaluate/blob/a5db2ad54aaeadf15c7aa1f056bc6f7bbeeb7e5f/metrics/squad/compute_score.py). [Apache 2.0 license](benchmarks/vendor/squad/LICENSE) | Byte-pinned independent scorer for exact match and token F1. [Vendored source and tokenizer notices](benchmarks/vendor/NOTICE.md) |
| Retrieval software | [rank-bm25](https://github.com/dorianbrown/rank_bm25) · [pytrec-eval-terrier](https://github.com/terrierteam/pytrec_eval) · [NIST trec_eval](https://github.com/usnistgov/trec_eval) | BM25 retrieval and independent score verification use installed dependencies. |
