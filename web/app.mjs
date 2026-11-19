import {
  comparison,
  gateNames,
  metrics,
  name,
  number,
  points,
  queryRows,
  resetView,
  systems,
} from "./analysis.mjs";

const $ = (id) => document.getElementById(id);
const upstream = {
  "squad-v1.1": "https://rajpurkar.github.io/SQuAD-explorer/",
  "nfcorpus-beir": "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
};
const pageSize = 8;
let release, explorer, families, state;
let loading = 0;

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function link(label, url) {
  const a = el("a", label);
  a.href = url;
  return a;
}
function links(items) {
  const wrapper = el("div", undefined, "links");
  for (const [label, url] of items) if (url) wrapper.append(link(label, url));
  return wrapper;
}
function svgNode(tag, attrs, text) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
function chart(w, h, label) {
  return svgNode("svg", {
    class: "chart",
    viewBox: [0, 0, w, h].join(" "),
    role: "img",
    "aria-label": label,
  });
}
function line(svg, x1, y1, x2, y2, cls) {
  svg.append(svgNode("line", { x1, y1, x2, y2, class: cls }));
}
function label(svg, x, y, text, anchor = "middle", cls = "") {
  svg.append(svgNode("text", { x, y, "text-anchor": anchor, class: cls }, text));
}
function choices(id, options, current, excluded) {
  const nodes = options.map((key) => {
    const option = el("option", name(key));
    option.value = key;
    option.disabled = key === excluded;
    return option;
  });
  $(id).replaceChildren(...nodes);
  $(id).value = current ?? "";
}
function stat(ledger, term, value) {
  const entry = el("div");
  entry.append(el("dt", term), el("dd", value));
  ledger.append(entry);
}
function makeTable(headers, rows, cls) {
  const table = el("table", undefined, cls);
  const head = table.createTHead().insertRow();
  for (const h of headers) {
    const th = el("th", h);
    th.scope = "col";
    head.append(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of row) tr.insertCell().append(value ?? "Unavailable");
  }
  return table;
}
function intervalPlot(r) {
  const figure = el("figure", undefined, "effect-plot");
  figure.append(el("figcaption", "Pairwise 95% interval · score difference (pp)"));
  const svg = chart(
    500,
    106,
    "Pairwise interval " +
      r.ci95.map((v) => points(v)).join(" to ") +
      " percentage points; mean " +
      points(r.mean_delta) +
      ".",
  );
  const threshold = Number.isFinite(r.margin) ? -r.margin * 100 : null;
  const [lo, hi] = r.ci95.map((v) => v * 100);
  const min = Math.min(lo, threshold ?? 0, 0),
    max = Math.max(hi, 0);
  const range = Math.max(max - min, 1);
  const step = [0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100].find((v) => range / v <= 5) ?? 100;
  const left = Math.floor((min - range * 0.08) / step) * step;
  const right = Math.ceil((max + range * 0.08) / step) * step;
  const x = (v) => 25 + ((v - left) / (right - left)) * 450;
  line(svg, 25, 76, 475, 76, "axis");
  for (let v = left; v <= right + step / 10; v += step) {
    line(svg, x(v), 73, x(v), 80, "axis");
    label(svg, x(v), 98, String(v).replace("-", "−"));
  }
  line(svg, x(0), 28, x(0), 76, "axis");
  if (threshold !== null) {
    line(svg, x(threshold), 24, x(threshold), 77, "cutoff");
    const tx = Math.max(90, Math.min(408, x(threshold)));
    label(svg, tx, 17, "Loss limit " + points(-r.margin) + " pp");
  }
  line(svg, x(lo), 53, x(hi), 53, "whisker");
  for (const v of [lo, hi]) line(svg, x(v), 45, x(v), 61, "whisker");
  svg.append(svgNode("circle", { cx: x(r.mean_delta * 100), cy: 53, r: 6.5, class: "dot" }));
  figure.append(svg);
  return figure;
}
function renderEffect() {
  const rows = families[state.dataset][state.kind];
  const r = comparison(rows, state.baseline, state.candidate, state.metric, state.margin);
  $("effect").replaceChildren();
  $("inference").replaceChildren();
  $("verdict").textContent = "";
  if (!r) {
    $("effect").append(el("p", "Choose two distinct recorded systems with a shared metric.", "empty"));
    return;
  }
  const group = el("div", undefined, "effect-grid");
  const estimate = el("div", undefined, "estimate");
  estimate.append(
    el("h3", name(r.candidate) + " minus " + name(r.baseline)),
    el("strong", points(r.mean_delta, true)),
    el("p", name(r.metric) + " · percentage points"),
  );
  group.append(estimate, intervalPlot(r));
  $("effect").append(group);
  const ledger = el("dl", undefined, "ledger");
  stat(ledger, "Pairwise 95% interval", r.ci95.map((v) => points(v, true)).join(" to ") + " pp");
  stat(ledger, "Holm-adjusted p", number(r.p_holm, 5));
  stat(ledger, "Wins / ties / losses", [r.wins, r.ties, r.losses].join(" / "));
  const units = r.resampling_unit === "group" ? "article clusters" : "queries";
  $("inference").append(
    ledger,
    el(
      "p",
      r.query_count +
        " paired queries. " +
        r.unit_count +
        " " +
        units +
        " resampled. Two-sided p adjusted over " +
        r.family_size +
        (r.family_size === 1 ? " comparison" : " comparisons") +
        " in the " +
        r.analysis +
        " family. Intervals apply to one pair.",
      "note",
    ),
  );
  $("verdict").textContent = r.gate ? gateNames[r.gate] : "Enter 0–100";
  $("verdict").dataset.gate = r.gate ?? "";
  $("gate-note").textContent =
    r.gate === "pass"
      ? "The lower bound meets the allowed loss. Changing tolerance reclassifies the recorded bounds."
      : r.gate === "regression"
        ? "Even the upper bound exceeds the allowed loss. Changing tolerance reclassifies these bounds."
        : r.gate === "inconclusive"
          ? "The bounds leave the tolerance decision unresolved. Changing tolerance reclassifies these bounds."
          : "Enter a loss tolerance between 0 and 100 percentage points.";
}

function qualityChart(rows) {
  const svg = chart(
    420,
    178,
    "Mean " + name(state.metric) + " against warm median latency, milliseconds. Exact values in run table.",
  );
  const valid = rows.filter(([, r]) => Number.isFinite(r.p50_ms) && Number.isFinite(r.means[state.metric]));
  const largest = Math.max(1, ...valid.map(([, r]) => r.p50_ms));
  const step = 10 ** Math.floor(Math.log10(largest));
  const xmax = Math.ceil(largest / step) * step;
  const x = (v) => 45 + (351 * v) / xmax;
  const y = (v) => 140 - 109 * v;
  label(svg, 7, 16, name(state.metric) + " (%)", "start", "data-label");
  for (const val of [0, 0.5, 1]) {
    line(svg, 45, y(val), 396, y(val), "grid");
    label(svg, 35, y(val) + 4, String(val * 100), "end");
  }
  line(svg, 45, 31, 45, 140, "axis");
  line(svg, 45, 140, 396, 140, "axis");
  for (const v of [0, xmax / 2, xmax]) label(svg, x(v), 156, String(v));
  label(svg, 220, 175, "Warm p50 latency (ms)");
  // Key every point to its table row. Nudge letter labels when runs are close.
  const placed = [];
  for (const [i, [id, r]] of rows.entries()) {
    if (!valid.some(([k]) => k === id)) continue;
    const cx = x(r.p50_ms),
      cy = y(r.means[state.metric]);
    const tx = Math.min(394, cx + 8);
    let ty = cy - 7;
    while (placed.some((p) => Math.abs(p.x - tx) < 18 && Math.abs(p.y - ty) < 13)) ty += 14;
    placed.push({ x: tx, y: ty });
    const dot = svgNode("circle", { cx, cy, r: 5, class: "dot" });
    dot.append(
      svgNode(
        "title",
        {},
        name(id) + ", " + points(r.means[state.metric]) + "% at " + number(r.p50_ms) + " ms",
      ),
    );
    svg.append(dot);
    label(svg, tx, ty, String.fromCharCode(65 + i), "start", "data-label");
  }
  return svg;
}
function renderRuns() {
  const d = release.datasets[state.dataset],
    runs = d.runs[state.kind];
  const rows = systems[state.kind].filter((id) => Object.hasOwn(runs, id)).map((id) => [id, runs[id]]);
  $("quality").replaceChildren(qualityChart(rows));
  $("quality-note").textContent = "Quality is the mean over all " + d.query_count + " evaluation queries.";
  const data = rows.map(([id, r], i) => {
    const key = el("span", String.fromCharCode(65 + i), "run-code");
    const title = el("span");
    title.append(key, link(name(id), r.evaluation));
    return [
      title,
      points(r.means[state.metric]),
      number(r.p50_ms),
      number(r.p95_ms),
      r.latency_sample_count ?? "Unavailable",
    ];
  });
  $("run-table").replaceChildren(
    makeTable(["System", name(state.metric) + " (%)", "p50 ms", "p95 ms", "Timings"], data),
  );
  $("runtime-link").href =
    release.inputs[state.kind === "answers" ? "answer_runtime" : "retrieval_runtime"];
  $("timing-note").textContent =
    state.kind === "answers"
      ? "Apple M4 Pro CPU. Batch-one generation after prompt construction and before decoding. Retrieval time is excluded."
      : "Apple M4 Pro CPU. Warm single-query processing through final ranking. Each system timed 32 queries five times with indexes already loaded.";
}

function retrievalDetail(obs, run) {
  const wrap = el("div");
  if (!obs) {
    wrap.append(el("p", "Retrieval observations unavailable."));
    return wrap;
  }
  wrap.append(
    el(
      "p",
      "First relevant rank: " +
        (obs.first_relevant_rank ?? "none in top 100") +
        ". Relevant in top 100: " +
        obs.relevant_in_top100 +
        " of " +
        obs.positive_judgments +
        " judged positives.",
    ),
  );
  const details = el("details");
  details.append(
    el("summary", "Top 10 retrieved documents"),
    makeTable(
      ["Document", "Rank", "Score", "Grade"],
      obs.top10.map(([id, rank, score, grade]) => [id, rank, number(score, 4), grade]),
    ),
    el("p", "Grade 0 includes unjudged documents.", "note"),
  );
  wrap.append(
    details,
    links([
      ["Full top-100 rankings", run.ranking],
      ["Run diagnostics", run.diagnostics],
    ]),
  );
  return wrap;
}
export function answerDetail(obs) {
  const wrap = el("div");
  if (!obs) {
    wrap.append(el("p", "Answer observation unavailable."));
    return wrap;
  }
  wrap.append(
    el("h4", "Generated answer"),
    el("p", obs.prediction === "" ? "(Empty output)" : obs.prediction),
    el(
      "p",
      "Gold passage supplied: " +
        (obs.gold_passage_supplied ? "yes" : "no") +
        ". Reference string in supplied text: " +
        (obs.answer_string_in_context ? "yes" : "no") +
        ".",
      "note",
    ),
    el(
      "p",
      obs.input_tokens +
        " input / " +
        obs.context_tokens +
        " context / " +
        obs.output_tokens +
        " output tokens. Generation " +
        (Number.isFinite(obs.latency_ns) ? number(obs.latency_ns / 1e6) + " ms" : "unavailable") +
        ".",
      "note",
    ),
  );
  const details = el("details");
  details.append(el("summary", "Actual supplied excerpts (" + obs.spans.length + ")"));
  if (!obs.spans.length) details.append(el("p", "No paragraph text entered the prompt."));
  for (const s of obs.spans)
    details.append(
      el(
        "p",
        s.docid +
          " · rank " +
          s.rank +
          " · characters [" +
          s.start +
          ", " +
          s.end +
          ") of " +
          s.source_chars,
      ),
      el("blockquote", s.text),
    );
  wrap.append(details);
  return wrap;
}
function showRecord(row) {
  const root = $("record");
  root.replaceChildren();
  if (!row) return;
  const { id, query } = row;
  const d = release.datasets[state.dataset],
    runs = d.runs[state.kind];
  const article = el("article", undefined, "record");
  const title = el("h3", query.question ?? id);
  title.tabIndex = -1;
  article.append(title);
  if (query.question) article.append(el("p", id + " · " + query.article, "note"));
  if (query.references)
    article.append(
      el("p", "Accepted reference answers: " + [...new Set(query.references)].join(" / "), "refs"),
    );
  const grid = el("div", undefined, "record-grid");
  for (const [label, id] of [
    ["Baseline", state.baseline],
    ["Candidate", state.candidate],
  ]) {
    const side = el("div"),
      obs = query[state.kind]?.[id],
      run = runs[id];
    side.append(
      el("h4", label + " · " + name(id)),
      el(
        "p",
        name(state.metric) +
          ": " +
          points(obs?.metrics[state.metric]) +
          (Number.isFinite(obs?.metrics[state.metric]) ? "%" : ""),
      ),
    );
    if (state.kind === "answers") {
      side.append(answerDetail(obs), links([["Prediction records", run.predictions]]));
      if (run.config.retriever) {
        const retriever = run.config.retriever;
        side.append(
          el("h4", "Underlying retrieval"),
          retrievalDetail(query.retrieval[retriever], d.runs.retrieval[retriever]),
        );
      }
    } else side.append(retrievalDetail(obs, run));
    grid.append(side);
  }
  article.append(
    grid,
    links([
      ["Dataset source", upstream[state.dataset]],
      ["Evaluation judgments", d.retrieval_references],
    ]),
  );
  root.append(article);
}
function renderQueries() {
  if (!explorer) return;
  const queries = explorer.datasets[state.dataset].queries;
  const rows = queryRows(queries, state),
    total = Object.keys(queries).length;
  state.page = Math.min(state.page, Math.max(0, Math.ceil(rows.length / pageSize) - 1));
  const start = state.page * pageSize;
  const slice = rows.slice(start, start + pageSize);
  if (!slice.some((r) => r.id === state.selected)) state.selected = null;
  $("query-n").textContent = total + " evaluation queries";
  $("query-table").replaceChildren();
  if (slice.length) {
    const table = makeTable(
      ["Query", "Baseline (%)", "Candidate (%)", "Δ pp"],
      slice.map((r) => {
        const cell = el("div"),
          b = el("button", r.query.question ?? r.id);
        b.type = "button";
        b.setAttribute("aria-expanded", String(state.selected === r.id));
        b.setAttribute("aria-controls", "record");
        b.addEventListener("click", () => {
          if (state.selected === r.id) {
            state.selected = null;
            renderQueries();
            $("query-table").querySelectorAll("button")[slice.indexOf(r)].focus();
            return;
          }
          state.selected = r.id;
          renderQueries();
          $("record").querySelector("h3")?.focus();
        });
        cell.append(b);
        if (r.query.question) cell.append(el("small", r.id));
        return [cell, points(r.base), points(r.candidate), points(r.delta, true)];
      }),
      "query-table",
    );
    slice.forEach((r, i) => {
      if (state.selected === r.id) table.tBodies[0].rows[i].className = "active";
    });
    $("query-table").append(table);
  } else
    $("query-table").append(
      el("p", "No queries match these filters. Change the outcome or clear the search.", "empty"),
    );
  const pg = $("pages");
  pg.replaceChildren(
    el(
      "span",
      (slice.length ? start + 1 + "–" + (start + slice.length) + " of " : "") +
        rows.length +
        " matching queries",
    ),
  );
  for (const [s, change, disabled] of [
    ["Previous", -1, start === 0],
    ["Next", 1, start + pageSize >= rows.length],
  ]) {
    const b = el("button", s);
    b.disabled = disabled;
    b.onclick = () => {
      state.page += change;
      renderQueries();
      $("pages")
        .querySelector(
          change > 0 ? "button:last-child:not(:disabled)" : "button:first-of-type:not(:disabled)",
        )
        ?.focus();
    };
    pg.append(b);
  }
  showRecord(slice.find((r) => r.id === state.selected));
}
function renderMethod() {
  const d = release.datasets[state.dataset],
    ans = state.kind === "answers";
  const method = el("div", undefined, "method-grid");
  const sample = el("div"),
    audit = el("div");
  sample.append(
    el("h3", "Selection and scoring"),
    el(
      "p",
      state.dataset === "squad-v1.1"
        ? "All 2,067 validation paragraphs are searchable. Article-disjoint pools supplied 50 development questions and 200 evaluation questions from 36 articles. FLAN-T5 may have encountered related training material."
        : "NFCorpus uses the full official development and test sets with graded relevance judgments. Its public records identify queries and documents by ID.",
    ),
    el(
      "p",
      ans
        ? "BM25 top 1 and hybrid top 3 change both the retriever and requested passage count. Their contrast measures the complete configurations. Context diagnostics track supplied excerpts and literal reference strings."
        : "BM25 and fusion settings were selected by development nDCG@10 before evaluation. All systems share the same query IDs and judgments. Unjudged documents receive zero gain.",
    ),
    links([
      ["Dataset source", upstream[state.dataset]],
      ["Selected configurations", d.selected],
      ["Development scores", d.tuning],
      ["Split membership", d.selection],
    ]),
  );
  audit.append(
    el("h3", "Uncertainty and provenance"),
    el(
      "p",
      "Each paired interval uses 10,000 bootstrap resamples at seed 20261119. Holm adjustment applies to the two-sided sign-flip p-values within each declared family. The release gate compares hybrid top 3 with closed-book on SQuAD token F1, allowing a loss of 2 percentage points.",
    ),
    el(
      "p",
      ans
        ? "The 600 original outputs are preserved. Their input binding was reconstructed against the pinned source and original evidence commit. Later runs capture the binding before loading the model."
        : "Download the full rankings and cross-checks to reproduce individual scores and verify the recorded decisions.",
    ),
    links([
      ["Pair estimates", release.comparisons[state.dataset][state.kind]],
      ["Independent score check", ans ? d.answer_crosscheck : d.retrieval_crosscheck],
      ["Input binding", ans && d.answer_inputs],
      ["Source hashes", release.inputs.sources],
      ["Release overview", "./results/release.json"],
      ["Architecture", "./docs/architecture.svg"],
    ]),
  );
  method.append(sample, audit);
  if (state.dataset === "squad-v1.1") {
    const credit = el(
      "p",
      "Questions, reference answers and paragraph extracts adapted from SQuAD v1.1 by Rajpurkar et al. ",
      "note",
    );
    credit.append(link("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"));
    method.append(credit);
  }
  $("method-content").replaceChildren(method);
}
function synchronize() {
  const d = release.datasets[state.dataset],
    runs = d.runs[state.kind];
  choices("dataset", Object.keys(release.datasets), state.dataset);
  choices("kind", Object.keys(d.runs), state.kind);
  const options = systems[state.kind].filter((s) => Object.hasOwn(runs, s));
  choices("baseline", options, state.baseline);
  choices("candidate", options, state.candidate, state.baseline);
  choices("metric", metrics[state.kind], state.metric);
  $("split").textContent = name(state.dataset) + " / " + d.split;
  $("margin").value = Number.isFinite(state.margin) ? String(state.margin * 100) : "";
  $("margin").setAttribute("aria-invalid", String(state.margin === null));
  for (const id of ["search", "filter", "sort"]) $(id).value = state[id];
  $("search").placeholder = state.dataset === "squad-v1.1" ? "Question, article or ID" : "Query ID";
  renderEffect();
  renderRuns();
  renderQueries();
  renderMethod();
}
async function read(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error("HTTP " + response.status);
  return response.json();
}
function retry(id, message, action) {
  const b = el("button", "Retry");
  b.onclick = action;
  $(id).replaceChildren(el("span", message), b);
}
async function loadExplorer(version) {
  $("explorer-controls").hidden = true;
  $("explorer-status").textContent = "Loading query records…";
  try {
    const data = await read(release.explorer);
    if (version !== loading) return;
    explorer = data;
    renderQueries();
    $("explorer-controls").hidden = false;
    $("explorer-status").replaceChildren();
  } catch {
    if (version === loading)
      retry("explorer-status", "Query records could not be loaded.", () => loadExplorer(version));
  }
}
async function load() {
  const version = ++loading;
  $("notebook").hidden = true;
  $("load").hidden = false;
  $("load").textContent = "Loading recorded comparisons…";
  try {
    const r = await read("./results/release.json");
    const files = Object.entries(r.comparisons).flatMap(([dataset, tracks]) =>
      Object.entries(tracks).map(async ([kind, path]) => [dataset, kind, await read(path)]),
    );
    const comparisons = await Promise.all(files);
    if (version !== loading) return;
    release = r;
    explorer = null;
    families = {};
    for (const [dataset, kind, rows] of comparisons) (families[dataset] ??= {})[kind] = rows;
    const declared = release.gates[0];
    state = {
      ...resetView(r, declared.dataset_id, declared.kind),
      baseline: declared.baseline,
      candidate: declared.candidate,
      metric: declared.metric,
      margin: declared.margin,
    };
    const datasets = Object.values(r.datasets);
    const runCount = datasets
      .flatMap((d) => Object.values(d.runs))
      .reduce((n, rs) => n + Object.keys(rs).length, 0);
    $("scope").textContent =
      datasets.length +
      " datasets / " +
      runCount +
      " recorded runs / " +
      datasets.reduce((n, d) => n + d.query_count, 0) +
      " evaluation queries";
    synchronize();
    $("load").hidden = true;
    $("notebook").hidden = false;
    const section = location.hash.slice(1);
    if (["comparison", "runs", "queries", "method"].includes(section)) $(section).scrollIntoView();
    await loadExplorer(version);
  } catch {
    if (version === loading) retry("load", "The recorded comparison files could not be loaded.", load);
  }
}
function start() {
  for (const id of ["dataset", "kind", "baseline", "candidate", "metric"])
    $(id).onchange = (e) => {
      const value = e.target.value;
      if (id === "dataset" || id === "kind") {
        state = resetView(
          release,
          id === "dataset" ? value : state.dataset,
          id === "kind" ? value : state.kind,
        );
      } else {
        if (id === "baseline" && value === state.candidate) state.candidate = state.baseline;
        state[id] = value;
        state.selected = null;
        state.page = 0;
      }
      synchronize();
    };
  $("margin").oninput = (e) => {
    const value = e.target.valueAsNumber;
    state.margin = Number.isFinite(value) && value >= 0 && value <= 100 ? value / 100 : null;
    e.target.setAttribute("aria-invalid", String(state.margin === null));
    renderEffect();
  };
  for (const id of ["search", "filter", "sort"])
    $(id).addEventListener(id === "search" ? "input" : "change", (e) => {
      state[id] = e.target.value;
      state.page = 0;
      state.selected = null;
      renderQueries();
    });
  load();
}
if (typeof document !== "undefined" && document.getElementById("notebook")) start();
