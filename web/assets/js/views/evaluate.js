// Evaluate view: confusion matrix, accuracy, per-class precision/recall, mistakes.
// The numbers come from the server (/api/evaluate) so the UI, CLI and notebook
// all use the exact same metric code.

import { api } from "../api.js";
import { LABELS, labelled, on, set, state } from "../store.js";
import { $, esc, icon, isDark, pct, tooltip } from "../util.js";

let pending;
let lastReport = null;

export function initEvaluate() {
  on("truth", schedule);
  on("items", schedule);
  on("theme", () => lastReport && render(lastReport));
  on("cellFilter", () => lastReport && renderMistakes());
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
  if (!items.length) { lastReport = null; return; }
  try {
    lastReport = await api.evaluate(items.map((i) => ({ true: i.truth, predicted: i.prediction.label })));
    render(lastReport);
  } catch (err) {
    $("#matrix-note").textContent = `Could not compute metrics: ${err.message}`;
  }
}

// Sequential ramp step (0–6) for a count; 0 is reserved for empty cells.
const step = (v, max) => (v === 0 ? 0 : 1 + Math.round((v / Math.max(max, 1)) * 5));
// Pick ink by the step's actual lightness in each theme.
const inkFor = (s) => (isDark() ? (s >= 5 ? "#0b0b0b" : "#f0efec") : s >= 4 ? "#ffffff" : "#0b0b0b");

function render(r) {
  const m = r.matrix;
  const max = Math.max(...m.flat());
  const grid = $("#matrix");
  const head = (l) => `<i style="background:var(--${l})"></i>${l}`;
  let html = `<div class="corner"></div><div class="axis axis--pred">Predicted</div><div></div>`;
  html += `<div class="corner"></div>${LABELS.map((l) => `<div class="head head--col">${head(l)}</div>`).join("")}<div class="total">total</div>`;
  LABELS.forEach((t, i) => {
    if (i === 0) html += `<div class="axis axis--true">True</div>`;
    html += `<div class="head head--row">${head(t)}</div>`;
    LABELS.forEach((p, j) => {
      const v = m[i][j];
      const s = step(v, max);
      const rowTotal = m[i].reduce((a, b) => a + b, 0);
      html += `<div class="cell${i === j ? " is-diag" : ""}" role="cell" tabindex="0" data-t="${t}" data-p="${p}"
        style="background:var(--seq-${s});color:${inkFor(s)}"
        aria-label="True ${t}, predicted ${p}: ${v}">${v}${rowTotal ? `<small>${pct(v / rowTotal, 0)}</small>` : ""}</div>`;
    });
    html += `<div class="total">${m[i].reduce((a, b) => a + b, 0)}</div>`;
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
    `<span class="legend-ramp"><span>0</span><span class="ramp"></span><span>${max}</span><span>photos per cell · green outline = correct</span></span>` +
    `<span>Based on ${r.total} photo${r.total === 1 ? "" : "s"} with a true label.${errors}</span>`;

  $("#acc-value").textContent = pct(r.accuracy);
  $("#acc-sub").innerHTML = `${r.correct} of ${r.total} on the diagonal · macro-F1 ${pct(r.macro_f1)}`;

  $("#per-class tbody").innerHTML = r.per_class.map((c) => {
    const i = LABELS.indexOf(c.label);
    const tp = m[i][i];
    return `<tr>
      <td><span class="badge badge--${c.label}">${c.label}</span></td>
      <td class="num">${pct(c.precision, 0)}<span class="frac">${tp}/${c.predicted}</span></td>
      <td class="num">${pct(c.recall, 0)}<span class="frac">${tp}/${c.support}</span></td>
      <td class="num">${pct(c.f1, 0)}</td></tr>`;
  }).join("");

  renderMistakes();
}

function renderMistakes() {
  const cf = state.cellFilter;
  const items = labelled().filter((i) =>
    cf ? i.truth === cf.true && i.prediction.label === cf.predicted : i.truth !== i.prediction.label);
  $("#mistake-count").innerHTML = cf
    ? `· true ${cf.true} → predicted ${cf.predicted} (${items.length}) <a href="#" id="clear-cell">show all mistakes</a>`
    : `· ${items.length}`;
  $("#clear-cell")?.addEventListener("click", (e) => { e.preventDefault(); set({ cellFilter: null }, "cellFilter"); });

  $("#mistakes").innerHTML = items.length
    ? items.map((i) => {
        const p = i.prediction;
        const probs = p.probabilities ? LABELS.map((l) => `${l} ${pct(p.probabilities[l], 0)}`).join(" · ") : esc(p.error);
        return `<div class="mistake">
          <img src="${i.url}" alt="" onerror="this.outerHTML='<div class=ph>${icon("image").replace(/"/g, "&quot;")}</div>'" />
          <div>
            <div class="mistake__name" title="${esc(i.name)}">${esc(i.name)}</div>
            <div class="mistake__line">true <span class="badge badge--${i.truth}">${i.truth}</span> → said <span class="badge badge--${p.label}">${p.label}</span> ${p.label !== "error" ? pct(p.confidence, 0) : ""}</div>
            <div class="mistake__line muted">${probs}${p.decided_by === "threshold" ? " · threshold override" : ""}</div>
          </div></div>`;
      }).join("")
    : `<p class="muted">${cf ? "No photos in this cell." : "No mistakes — every labelled photo is on the diagonal."}</p>`;
}
