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
