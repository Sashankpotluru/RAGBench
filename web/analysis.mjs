export const names = {
  "squad-v1.1": "SQuAD v1.1",
  "nfcorpus-beir": "NFCorpus",
  retrieval: "Retrieval",
  answers: "Answer generation",
  "bm25-fixed": "Fixed BM25",
  "bm25-tuned": "Tuned BM25",
  minilm: "MiniLM",
  "hybrid-tuned": "Tuned hybrid",
  "bm25-top1": "BM25 top 1",
  "hybrid-top3": "Hybrid top 3",
  "closed-book": "Closed-book",
  token_f1: "Token F1",
  exact_match: "Exact match",
  "ndcg@10": "nDCG@10",
  "mrr@10": "MRR@10",
  "precision@10": "Precision@10",
  "recall@5": "Recall@5",
  "recall@10": "Recall@10",
  "recall@100": "Recall@100",
};
export const name = (key) => names[key] ?? key;
export const systems = {
  answers: ["closed-book", "bm25-top1", "hybrid-top3"],
  retrieval: ["bm25-fixed", "bm25-tuned", "minilm", "hybrid-tuned"],
};
export const metrics = {
  answers: ["token_f1", "exact_match"],
  retrieval: ["ndcg@10", "recall@5", "recall@10", "recall@100", "mrr@10", "precision@10"],
};
export const gateNames = {
  pass: "Within tolerance",
  regression: "Regression",
  inconclusive: "Inconclusive",
};
export const number = (v, places = 2) => (Number.isFinite(v) ? v.toFixed(places) : "Unavailable");
export const points = (v, signed = false) =>
  Number.isFinite(v)
    ? (signed && v > 0 ? "+" : "") + (v * 100).toFixed(2).replace("-", "−")
    : "Unavailable";

export function gate([low, high], margin) {
  if (!Number.isFinite(margin) || margin < 0) return null;
  if (low >= -margin) return "pass";
  return high < -margin ? "regression" : "inconclusive";
}

export function comparison(rows, baseline, candidate, metric, margin) {
  if (baseline === candidate) return null;
  const row = rows.find(
    (r) =>
      r.metric === metric &&
      ((r.baseline === baseline && r.candidate === candidate) ||
        (r.baseline === candidate && r.candidate === baseline)),
  );
  if (!row) return null;
  let result = { ...row };
  if (row.baseline !== baseline) {
    result = {
      ...row,
      baseline,
      candidate,
      baseline_mean: row.candidate_mean,
      candidate_mean: row.baseline_mean,
      baseline_sha256: row.candidate_sha256,
      candidate_sha256: row.baseline_sha256,
      mean_delta: -row.mean_delta,
      ci95: [-row.ci95[1], -row.ci95[0]],
      wins: row.losses,
      losses: row.wins,
    };
  }
  return { ...result, gate: gate(result.ci95, margin), margin };
}

export function resetView(release, dataset, requestedKind) {
  const runs = release.datasets[dataset].runs;
  const kind = Object.hasOwn(runs, requestedKind) ? requestedKind : Object.keys(runs)[0];
  const options = systems[kind].filter((s) => Object.hasOwn(runs[kind], s));
  return {
    dataset,
    kind,
    baseline: options[0],
    candidate: options.length > 1 ? options.at(-1) : null,
    metric: metrics[kind][0],
    margin: 0.02,
    search: "",
    filter: "all",
    sort: "id",
    page: 0,
    selected: null,
  };
}

export function queryRows(queries, state) {
  const term = state.search.trim().toLowerCase();
  const rows = [];
  for (const [id, query] of Object.entries(queries)) {
    if (term && ![id, query.question, query.article].some((s) => s?.toLowerCase().includes(term))) continue;
    const observations = query[state.kind];
    const base = observations?.[state.baseline]?.metrics?.[state.metric];
    const candidate = observations?.[state.candidate]?.metrics?.[state.metric];
    const delta = Number.isFinite(base) && Number.isFinite(candidate) ? candidate - base : null;
    const outcome = delta === null ? "unavailable" : delta === 0 ? "tie" : delta > 0 ? "win" : "loss";
    if (state.filter === "changed" && !["win", "loss"].includes(outcome)) continue;
    if (!["changed", "all", outcome].includes(state.filter)) continue;
    rows.push({ id, query, base, candidate, delta, outcome });
  }
  const byId = (a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  if (state.sort === "id") return rows.sort(byId);
  const direction = state.sort === "gain" ? -1 : 1;
  return rows.sort(
    (a, b) => (a.delta === null) - (b.delta === null) || direction * (a.delta - b.delta) || byId(a, b),
  );
}
