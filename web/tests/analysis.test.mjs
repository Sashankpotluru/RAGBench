import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { comparison, gate, number, points, queryRows, resetView } from "../analysis.mjs";
import { answerDetail } from "../app.mjs";

const read = (file) => JSON.parse(readFileSync(new URL("../../" + file, import.meta.url)));
const release = read("results/release.json");
const answers = read(release.comparisons["squad-v1.1"].answers);
const firstQueries = Object.fromEntries(
  Object.entries(read(release.explorer).datasets["squad-v1.1"].queries).slice(0, 12),
);
const initial = resetView(release, "squad-v1.1", "answers");

test("swapping a measured asymmetric interval changes direction and reclassifies the gate", () => {
  const a = comparison(answers, "closed-book", "hybrid-top3", "token_f1", 0.02);
  const b = comparison(answers, "hybrid-top3", "closed-book", "token_f1", 0.02);
  assert.deepEqual(b.ci95, [-a.ci95[1], -a.ci95[0]]);
  assert.equal(b.mean_delta, -a.mean_delta);
  assert.equal(b.baseline_mean, a.candidate_mean);
  assert.equal(b.baseline_sha256, a.candidate_sha256);
  assert.equal(b.losses, a.wins);
  assert.equal(b.wins, a.losses);
  for (const k of ["p_holm", "p_value", "family", "family_size", "ties", "unit_count"])
    assert.equal(a[k], b[k]);
  assert.equal(a.gate, "pass");
  assert.equal(b.gate, "regression");
  assert.equal(comparison(answers, "closed-book", "closed-book", "token_f1", 0.02), null);
  assert.equal(comparison(answers, "other", "closed-book", "token_f1", 0.02), null);
});
test("loss boundary is inclusive at the lower bound and strict at the upper", () => {
  assert.equal(gate([-0.02, 0.1], 0.02), "pass");
  assert.equal(gate([-0.1, -0.02], 0.02), "inconclusive");
  assert.equal(gate([-0.1, -0.0200001], 0.02), "regression");
  assert.equal(gate([0, 0], 0), "pass");
  for (const v of [-0.01, Infinity, NaN, null]) assert.equal(gate([-0.1, 0.1], v), null);
});
test("changing datasets or tracks clears unavailable comparisons and query state", () => {
  const old = { ...initial, search: "text", filter: "loss", page: 2, selected: "previous" };
  const next = resetView(release, "nfcorpus-beir", old.kind);
  assert.deepEqual(next, {
    dataset: "nfcorpus-beir",
    kind: "retrieval",
    baseline: "bm25-fixed",
    candidate: "hybrid-tuned",
    metric: "ndcg@10",
    margin: 0.02,
    search: "",
    filter: "all",
    sort: "id",
    page: 0,
    selected: null,
  });
  assert.equal(resetView(release, "squad-v1.1", "retrieval").metric, "ndcg@10");
});
test("query filters use recorded per-ID values and leave full-sample comparisons intact", () => {
  const q = Object.keys(firstQueries)[0];
  const source = JSON.stringify(answers);
  const rows = queryRows(firstQueries, { ...initial, sort: "gain" });
  assert.equal(rows.length, 12);
  assert.ok(rows.every((r, i) => i === 0 || rows[i - 1].delta >= r.delta));
  for (const r of rows)
    assert.equal(
      r.delta,
      r.query.answers["hybrid-top3"].metrics.token_f1 - r.query.answers["closed-book"].metrics.token_f1,
    );
  assert.deepEqual(
    queryRows(firstQueries, { ...initial, search: q }).map((r) => r.id),
    [q],
  );
  assert.equal(queryRows(firstQueries, { ...initial, search: "missing-xyz" }).length, 0);
  assert.equal(
    queryRows(firstQueries, { ...initial, filter: "changed", baseline: "hybrid-top3" }).length,
    0,
  );
  const missing = queryRows(firstQueries, { ...initial, candidate: "missing" });
  assert.ok(missing.every((r) => r.delta === null && r.outcome === "unavailable"));
  assert.equal(number(null), "Unavailable");
  assert.equal(points(undefined), "Unavailable");
  assert.equal(points(0.0247, true), "+2.47");
  assert.equal(JSON.stringify(answers), source);
});
test("generated output and excerpt markup remain literal text nodes", () => {
  const probe = "<img src=x onerror=alert(1)>";
  class Element {
    constructor(tag) {
      this.tag = tag;
      this.children = [];
      this.textContent = "";
    }
    append(...nodes) {
      this.children.push(...nodes);
    }
    set innerHTML(_) {
      throw new Error("HTML interpretation requested");
    }
  }
  globalThis.document = { createElement: (t) => new Element(t) };
  try {
    const obs = structuredClone(Object.values(firstQueries)[0].answers["hybrid-top3"]);
    obs.prediction = probe;
    obs.spans[0].text = probe;
    const tree = answerDetail(obs);
    const flatten = (n) => [n, ...n.children.flatMap(flatten)];
    const nodes = flatten(tree);
    assert.deepEqual(
      nodes.filter((n) => n.textContent === probe).map((n) => n.tag),
      ["p", "blockquote"],
    );
    assert.equal(
      nodes.some((n) => n.tag === "img"),
      false,
    );
  } finally {
    delete globalThis.document;
  }
});
