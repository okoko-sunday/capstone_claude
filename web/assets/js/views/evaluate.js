// Evaluate view: confusion matrix, accuracy, per-class precision/recall, mistakes.
// The numbers come from the server (/api/evaluate) so the UI, CLI and notebook
// all use the exact same metric code.

import { api } from "../api.js";
import { LABELS, labelled, on, set, state } from "../store.js";
import { $, download, esc, icon, pct, tooltip } from "../util.js";

let pending;
let lastReport = null;

export function initEvaluate() {
  on("truth", schedule);
  on("items", schedule);
  on("theme", () => lastReport && render(lastReport));
  on("cellFilter", () => lastReport && renderMistakes());
  $("#mistake-mode").addEventListener("change", () => lastReport && renderMistakes());
  $("#matrix-download").addEventListener("click", downloadMatrix);
}

function schedule() {
  clearTimeout(pending);
  pending = setTimeout(refresh, 120);
}

async function refresh() {
  const items = labelled();
  const badge = $("#eval-badge");
  badge.hidden = !items.length;
  badge.textContent = items.length;
  $("#eval-empty").hidden = items.length > 0;
  $("#eval").hidden = !items.length;
  $("#eval-count-card").hidden = !items.length;
  if (!items.length) { lastReport = null; return; }
  try {
    lastReport = await api.evaluate(items.map((i) => ({ true: i.truth, predicted: i.prediction.label })));
    render(lastReport);
  } catch (err) {
    $("#matrix-note").textContent = `Could not compute metrics: ${err.message}`;
  }
}

// Diagonal cells take their class colour, mistakes take blue; more photos = stronger tint.
function cellFill(i, j, v, max) {
  if (v === 0) return "var(--surface-2)";
  const share = Math.round(14 + (v / Math.max(max, 1)) * 34);
  const colour = i === j ? `var(--${LABELS[i]})` : "var(--blue)";
  return `color-mix(in srgb, ${colour} ${share}%, var(--surface))`;
}

function render(r) {
  const m = r.matrix;
  const max = Math.max(...m.flat());
  const grid = $("#matrix");
  const head = (l) => `<i style="background:var(--${l})"></i>${l}`;
  let html = `<div class="corner"></div><div class="axis axis--pred">Predicted</div><div></div>`;
  html += `<div class="corner"></div>${LABELS.map((l) => `<div class="head head--col">${head(l)}</div>`).join("")}<div class="total total--head">Total</div>`;
  LABELS.forEach((t, i) => {
    if (i === 0) html += `<div class="axis axis--true">True</div>`;
    html += `<div class="head head--row">${head(t)}</div>`;
    const rowTotal = m[i].reduce((a, b) => a + b, 0);
    LABELS.forEach((p, j) => {
      const v = m[i][j];
      const border = i === j && v ? `border-color:color-mix(in srgb, var(--${t}) 45%, transparent)` : "";
      html += `<div class="cell${i === j ? " is-diag" : ""}" role="cell" tabindex="0" data-t="${t}" data-p="${p}"
        style="background:${cellFill(i, j, v, max)};${border}"
        aria-label="True ${t}, predicted ${p}: ${v}">${v}<small>${rowTotal ? pct(v / rowTotal, 0) : "—"}</small></div>`;
    });
    html += `<div class="total">${rowTotal}</div>`;
  });
  grid.innerHTML = html;

  grid.querySelectorAll(".cell").forEach((cell) => {
    const t = cell.dataset.t, p = cell.dataset.p;
    const i = LABELS.indexOf(t), j = LABELS.indexOf(p);
    const rowTotal = m[i].reduce((a, b) => a + b, 0);
    tooltip(cell, () => `<strong>True ${t} → predicted ${p}</strong>
      <div class="row"><span>photos</span><strong>${m[i][j]}</strong></div>
      <div class="row"><span>share of real ${t}s</span><strong>${rowTotal ? pct(m[i][j] / rowTotal, 0) : "—"}</strong></div>
      <div class="muted" style="margin-top:6px">${i === j ? "Correct — on the diagonal." : "Click to list these photos."}</div>`);
    const pick = () => {
      const same = state.cellFilter?.true === t && state.cellFilter?.predicted === p;
      set({ cellFilter: same ? null : { true: t, predicted: p } }, "cellFilter");
    };
    cell.addEventListener("click", pick);
    cell.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), pick()));
  });

  const errors = r.errors_excluded ? ` ${r.errors_excluded} unreadable file(s) excluded.` : "";
  $("#matrix-note").innerHTML =
    `<span class="legend-ramp"><span>0</span><span class="ramp"></span><span>${max} photos per cell · stronger colour = more photos</span></span>` +
    `<span>Based on ${r.total} photo${r.total === 1 ? "" : "s"} with a true label.${errors}</span>`;

  $("#acc-value").textContent = pct(r.accuracy);
  $("#acc-sub").innerHTML = `${r.correct} of ${r.total} on the diagonal<br />Macro-F1: ${pct(r.macro_f1)}`;
  $("#eval-count").textContent = r.total;

  // Sparklines from the real labelled photos, in the order they were added:
  // running accuracy, and the running count of correct answers.
  const seq = labelled().map((i) => i.truth === i.prediction.label);
  let ok = 0;
  const running = seq.map((hit, k) => (ok += hit) / (k + 1));
  let c = 0;
  sparkline($("#acc-spark"), running, 0, 1);
  sparkline($("#count-spark"), seq.map((hit) => (c += hit)), 0, Math.max(1, seq.length));

  $("#per-class tbody").innerHTML = r.per_class.map((cl) => {
    const i = LABELS.indexOf(cl.label);
    const tp = m[i][i];
    return `<tr>
      <td><span class="cls"><i class="dot dot--${cl.label}"></i>${cl.label}</span></td>
      <td>${pct(cl.precision, 0)}<span class="frac">${tp}/${cl.predicted}</span></td>
      <td>${pct(cl.recall, 0)}<span class="frac">${tp}/${cl.support}</span></td>
      <td class="num">${pct(cl.f1, 0)}</td></tr>`;
  }).join("");

  renderMistakes();
  renderWorking(r);
}

// ------------------------------------------------------------ show the working
// One-vs-rest counts for each class, straight from the matrix (rows = true, columns = predicted).
function outcomes(m) {
  const total = m.flat().reduce((a, b) => a + b, 0);
  return LABELS.map((label, i) => {
    const tp = m[i][i];
    const fn = m[i].reduce((a, b) => a + b, 0) - tp; // real <label>s called something else
    const fp = m.reduce((a, row) => a + row[i], 0) - tp; // other photos called <label>
    return { label, tp, fp, fn, tn: total - tp - fp - fn, total };
  });
}

const ratio = (a, b) => (b ? a / b : null);
const show = (v) => (v == null ? "undefined (÷ 0)" : pct(v));
const name = (l) => `<span class="cls"><i class="dot dot--${l}"></i>${l}</span>`;

function formulaBox(title, formula, lines, result, note = "") {
  return `<div class="formula">
    <div class="formula__title">${title}</div>
    <div class="formula__eq">${formula}</div>
    <div class="formula__lines">${lines.join("")}</div>
    ${result ? `<div class="formula__result">${result}</div>` : ""}
    ${note ? `<p class="formula__note">${note}</p>` : ""}
  </div>`;
}

const line = (label, working, value) =>
  `<div class="formula__line">${name(label)}<span class="formula__work">${working}</span><strong>${show(value)}</strong></div>`;

function renderWorking(r) {
  const rows = outcomes(r.matrix);
  const total = r.total;
  $("#outcomes tbody").innerHTML = rows.map((o) => `<tr>
      <td>${name(o.label)}</td>
      <td class="num out out--good">${o.tp}</td>
      <td class="num out out--bad">${o.fp}</td>
      <td class="num out out--bad">${o.fn}</td>
      <td class="num out out--good">${o.tn}</td>
      <td class="num muted">${o.total}</td></tr>`).join("");

  const d = rows.find((o) => o.label === "dog");
  $("#outcomes-note").innerHTML =
    `With three classes, each row treats one class as “positive” and the other two as “negative”. Reading the <strong>dog</strong> row: ` +
    `<strong>TP ${d.tp}</strong> real dogs called dog · <strong>FP ${d.fp}</strong> cats or “neither” photos wrongly called dog · ` +
    `<strong>FN ${d.fn}</strong> real dogs called something else · <strong>TN ${d.tn}</strong> non-dogs correctly not called dog. ` +
    `TP + FP + FN + TN = ${d.total}, every labelled photo.`;

  // Per-class values, then the macro average (mean over classes that have photos, as the server computes macro-F1).
  const per = rows.map((o) => {
    const precision = ratio(o.tp, o.tp + o.fp);
    const recall = ratio(o.tp, o.tp + o.fn);
    const f1 = precision != null && recall != null ? ratio(2 * precision * recall, precision + recall) : null;
    return { ...o, precision, recall, f1 };
  });
  const present = per.filter((o) => o.tp + o.fn > 0);
  const macro = (key) => (present.length ? present.reduce((a, o) => a + (o[key] ?? 0), 0) / present.length : null);
  const macroLine = (key) => `Macro average (mean of the ${present.length} class${present.length === 1 ? "" : "es"} with photos) = <strong>${show(macro(key))}</strong>`;

  const tps = rows.map((o) => o.tp);
  const accuracy = formulaBox(
    "Accuracy",
    `correct predictions ÷ all predictions = (TP<sub>cat</sub> + TP<sub>dog</sub> + TP<sub>neither</sub>) ÷ total`,
    [`<div class="formula__line formula__line--wide"><span class="formula__work">(${tps.join(" + ")}) ÷ ${total} = ${r.correct}/${total}</span><strong>${show(ratio(r.correct, total))}</strong></div>`,
     ...rows.map((o) => line(o.label, `(TP + TN) ÷ total = (${o.tp} + ${o.tn}) ÷ ${total}`, ratio(o.tp + o.tn, total)))],
    `Overall accuracy = <strong>${show(ratio(r.correct, total))}</strong>, the diagonal of the matrix ÷ ${total}`,
    `The first line is the accuracy to quote. The (TP + TN) ÷ total lines are each class’s own yes/no accuracy. They come out higher, ` +
    `because every photo is a “true negative” for the classes it isn’t, so TN is large even when the model makes mistakes.`);
  const precision = formulaBox(
    "Precision",
    `TP ÷ (TP + FP) <span class="muted">— of everything it called this class, how much really was?</span>`,
    per.map((o) => line(o.label, `${o.tp} ÷ (${o.tp} + ${o.fp}) = ${o.tp}/${o.tp + o.fp}`, o.precision)),
    macroLine("precision"));
  const recall = formulaBox(
    "Recall",
    `TP ÷ (TP + FN) <span class="muted">— of the real photos of this class, how many did it find?</span>`,
    per.map((o) => line(o.label, `${o.tp} ÷ (${o.tp} + ${o.fn}) = ${o.tp}/${o.tp + o.fn}`, o.recall)),
    macroLine("recall"));
  const f1 = formulaBox(
    "F1-score",
    `2 × precision × recall ÷ (precision + recall) <span class="muted">= 2TP ÷ (2TP + FP + FN)</span>`,
    per.map((o) => line(o.label, `2×${o.tp} ÷ (2×${o.tp} + ${o.fp} + ${o.fn}) = ${2 * o.tp}/${2 * o.tp + o.fp + o.fn}`, o.f1)),
    macroLine("f1"),
    "F1 is high only when precision and recall are both high, so one weak side drags it down.");
  $("#formulas").innerHTML = accuracy + precision + recall + f1;
}

function sparkline(svg, values, lo, hi) {
  if (values.length < 2) { svg.innerHTML = ""; return; }
  const W = 160, H = 56, pad = 6;
  const x = (k) => (k / (values.length - 1)) * W;
  const y = (v) => H - pad - ((v - lo) / Math.max(hi - lo, 1e-9)) * (H - pad * 2);
  const pts = values.map((v, k) => [x(k), y(v)]);
  // Smooth with a simple Catmull-Rom → Bézier conversion.
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let k = 0; k < pts.length - 1; k++) {
    const p0 = pts[k - 1] ?? pts[k], p1 = pts[k], p2 = pts[k + 1], p3 = pts[k + 2] ?? p2;
    d += ` C${p1[0] + (p2[0] - p0[0]) / 6},${p1[1] + (p2[1] - p0[1]) / 6} ${p2[0] - (p3[0] - p1[0]) / 6},${p2[1] - (p3[1] - p1[1]) / 6} ${p2[0]},${p2[1]}`;
  }
  const last = pts[pts.length - 1];
  svg.innerHTML = `<path class="area" d="${d} L${W},${H} L0,${H} Z"/><path class="line" d="${d}"/><circle cx="${last[0]}" cy="${last[1]}" r="3.5"/>`;
}

function renderMistakes() {
  const cf = state.cellFilter;
  const mode = $("#mistake-mode").value;
  const items = labelled().filter((i) =>
    cf ? i.truth === cf.true && i.prediction.label === cf.predicted
       : mode === "all" || i.truth !== i.prediction.label);
  $("#mistake-count").innerHTML = cf
    ? `· true ${cf.true} → predicted ${cf.predicted} (${items.length}) <a href="#" id="clear-cell">show all</a>`
    : `· ${items.length}`;
  $("#clear-cell")?.addEventListener("click", (e) => { e.preventDefault(); set({ cellFilter: null }, "cellFilter"); });

  $("#mistakes").innerHTML = items.length
    ? items.map((i) => {
        const p = i.prediction;
        const right = i.truth === p.label;
        const probs = p.probabilities
          ? LABELS.map((l) => `<span class="prob prob--${l}">${l} ${pct(p.probabilities[l], 0)}</span>`).join("")
          : `<span class="prob">${esc(p.error)}</span>`;
        return `<div class="mistake${right ? " is-right" : ""}">
          <img src="${i.url}" alt="" onerror="this.outerHTML='<div class=ph>${icon("image").replace(/"/g, "&quot;")}</div>'" />
          <div style="min-width:0">
            <div class="mistake__name" title="${esc(i.name)}">${esc(i.name.split("/").pop())}</div>
            <div class="mistake__line">True: <b style="color:var(--${i.truth})">${i.truth}</b> ${icon("arrow")} Predicted: <b style="color:var(--${p.label === "error" ? "critical" : p.label})">${p.label}</b>${p.decided_by === "threshold" ? " · threshold" : ""}</div>
            <div class="mistake__probs">${probs}</div>
          </div>
          <span class="mistake__more" aria-hidden="true">···</span></div>`;
      }).join("")
    : `<p class="muted">${cf ? "No photos in this cell." : "No mistakes — every labelled photo is on the diagonal."}</p>`;
  $$arrowSize();
}

// Inline arrow icons inside text lines should stay small.
function $$arrowSize() {
  document.querySelectorAll(".mistake__line svg").forEach((s) => { s.style.width = "13px"; s.style.height = "13px"; });
}

function downloadMatrix() {
  if (!lastReport) return;
  const rows = [["true \\ predicted", ...LABELS, "total"]];
  lastReport.matrix.forEach((r, i) => rows.push([LABELS[i], ...r, r.reduce((a, b) => a + b, 0)]));
  rows.push([], ["accuracy", lastReport.accuracy.toFixed(4)], ["macro_f1", lastReport.macro_f1.toFixed(4)]);
  download("confusion-matrix.csv", rows.map((r) => r.join(",")).join("\n"));
}
