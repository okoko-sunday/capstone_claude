// Model card: what was trained, on what, how "neither" is decided, and the
// evidence behind that decision (threshold sweep + Approach B study).

import { LABELS, on, state } from "../store.js";
import { $, esc, pct, tooltip } from "../util.js";

const SOURCE_NAMES = {
  oxford_iiit_pet: "Oxford-IIIT Pet",
  caltech101: "Caltech-101",
  imagenette: "Imagenette",
  own: "Own photos",
};

export function initModelCard() {
  on("model", render);
  on("reports", render);
}

function render() {
  const m = state.model;
  if (!m) return;
  const v = m.validation?.tuned;
  const totals = sumCounts(m.train_counts ?? {});
  const valTotals = sumCounts(m.val_counts ?? {});
  const policy = m.policy ?? {};
  const b = state.reports?.approach_b;

  $("#model-card").innerHTML = `
  <div class="mc">
    <div class="card span-4">
      <h3 class="card__title">At a glance</h3>
      <dl class="kv">
        <dt>Approach</dt><dd>${esc(approachName(m.approach))}</dd>
        <dt>Backbone</dt><dd>${esc(m.backbone ?? "—")} (ImageNet)</dd>
        <dt>Input</dt><dd>${m.image_size}×${m.image_size}, letterboxed</dd>
        <dt>Animal threshold</dt><dd>${policy.animal_threshold?.toFixed(2) ?? "—"}</dd>
        <dt>Flip TTA</dt><dd>${policy.tta ? "on" : "off"}</dd>
        <dt>Trained</dt><dd>${m.trained_at ? new Date(m.trained_at).toLocaleString() : "—"}</dd>
        <dt>Val accuracy</dt><dd>${pct(v?.accuracy)}</dd>
        <dt>Val macro-F1</dt><dd>${pct(v?.macro_f1)}</dd>
      </dl>
    </div>

    <div class="card span-8 approach">
      <h3 class="card__title" style="margin:0">How “neither” is decided</h3>
      <div class="pipeline">
        <span class="step">Photo (any size)</span><span class="arrow">→</span>
        <span class="step">EXIF-rotate · RGB · letterbox ${m.image_size}px</span><span class="arrow">→</span>
        <span class="step">${esc(m.backbone ?? "backbone")}</span><span class="arrow">→</span>
        <span class="step">softmax: cat · dog · neither</span><span class="arrow">→</span>
        <span class="step">if max(cat, dog) &lt; ${policy.animal_threshold?.toFixed(2)} → neither</span>
      </div>
      <p><strong>Approach A (three classes)</strong> with a safety net. “Neither” is a trained class built from diverse negatives, so the model can actively say “not a pet” instead of picking the least-bad of two options. The threshold then catches photos where the model leans towards an animal only weakly; it was tuned on validation data for the best macro-F1.</p>
      ${b ? approachB(b) : `<p class="muted">Approach B evidence (two-class + threshold) appears here after running <code>python -m catdog train --two-class</code>.</p>`}
    </div>

    <div class="card span-6" id="datasets">
      <h3 class="card__title">Training data by source</h3>
      ${sourceBars(m.train_counts ?? {})}
      <div class="counts-legend">${LABELS.map((l) => `<span><i style="background:var(--${l})"></i>${l}: <strong>${totals[l] ?? 0}</strong> train · ${valTotals[l] ?? 0} val</span>`).join("")}</div>
    </div>

    <div class="card span-6">
      <h3 class="card__title">Validation macro-F1 by animal threshold</h3>
      ${sweepChart(m.validation?.threshold_sweep ?? [], policy.animal_threshold)}
      <p class="footnote">The chosen threshold is marked. Validation is a held-out slice of the <em>training sources</em>; expect lower numbers on real phone photos.</p>
    </div>

    ${v ? `<div class="card span-6"><h3 class="card__title">Validation per class</h3>
      <div class="scroll-x"><table class="data-table data-table--compact"><thead><tr><th>class</th><th class="num">precision</th><th class="num">recall</th><th class="num">F1</th><th class="num">support</th></tr></thead>
      <tbody>${v.per_class.map((c) => `<tr><td><span class="badge badge--${c.label}">${c.label}</span></td><td class="num">${pct(c.precision)}</td><td class="num">${pct(c.recall)}</td><td class="num">${pct(c.f1)}</td><td class="num">${c.support}</td></tr>`).join("")}</tbody></table></div></div>` : ""}

    <div class="card span-6">
      <h3 class="card__title">Excluded from “neither” — and why</h3>
      <ul class="excl">${Object.entries({ ...(m.exclusions?.caltech101 ?? {}), ...(m.exclusions?.imagenette ?? {}) })
        .map(([k, why]) => `<li><code>${esc(k)}</code><span>${esc(why)}</span></li>`).join("")}</ul>
    </div>
  </div>`;

  wireSweepTooltip(m.validation?.threshold_sweep ?? []);
}

function approachName(a) {
  return a === "A-three-class" ? "A · three classes + threshold" : a ?? "—";
}

function approachB(b) {
  const best = b.threshold_study?.best;
  if (!best) return "";
  const confident = b.threshold_study.negatives_confident_over_90;
  // The conclusion follows the measured numbers — never a canned verdict.
  const verdict = confident >= 0.1 || best.balanced_accuracy < 0.85
    ? "Many negatives were confidently wrong, so no threshold separates them cleanly — the failure the brief warns about."
    : "On this data the threshold separated pets from negatives reasonably well; compare both approaches on your own hold-out set before deciding.";
  return `<p><strong>Approach B, measured.</strong> A cats-vs-dogs model reached ${pct(b.pet_accuracy)} on pets. At its best threshold
    (${best.threshold.toFixed(2)}), ${pct(best.pet_recall)} of pets stayed cat/dog and ${pct(best.neither_recall)} of negatives became “neither”;
    ${pct(confident)} of negatives were called cat or dog with &gt;90% confidence. ${verdict}</p>`;
}

function sumCounts(bySource) {
  const out = {};
  for (const per of Object.values(bySource)) for (const [l, n] of Object.entries(per)) out[l] = (out[l] ?? 0) + n;
  return out;
}

function sourceBars(bySource) {
  const rows = Object.entries(bySource);
  if (!rows.length) return `<p class="muted">No count data in metadata.</p>`;
  const max = Math.max(...rows.map(([, per]) => Object.values(per).reduce((a, b) => a + b, 0)));
  return rows.map(([src, per]) => {
    const total = Object.values(per).reduce((a, b) => a + b, 0);
    return `<div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;font-size:13px"><span>${esc(SOURCE_NAMES[src] ?? src)}</span><span class="muted" style="font-variant-numeric:tabular-nums">${total.toLocaleString()}</span></div>
      <div class="stackbar" style="width:${Math.max(4, (total / max) * 100)}%">
        ${LABELS.filter((l) => per[l]).map((l) => `<span title="${l}: ${per[l]}" style="flex:${per[l]};background:var(--${l})"></span>`).join("")}
      </div></div>`;
  }).join("");
}

// Single-series line chart, inline SVG. One series → no legend; the title names it.
const W = 520, H = 200, PAD = { l: 40, r: 12, t: 12, b: 28 };

function sweepChart(sweep, chosen) {
  if (!sweep.length) return `<p class="muted">No sweep data.</p>`;
  const xs = sweep.map((d) => d.threshold), ys = sweep.map((d) => d.macro_f1 ?? 0);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.max(0, Math.floor(Math.min(...ys) * 10) / 10), y1 = 1;
  const sx = (x) => PAD.l + ((x - x0) / (x1 - x0 || 1)) * (W - PAD.l - PAD.r);
  const sy = (y) => H - PAD.b - ((y - y0) / (y1 - y0 || 1)) * (H - PAD.t - PAD.b);
  const path = sweep.map((d, i) => `${i ? "L" : "M"}${sx(d.threshold).toFixed(1)},${sy(d.macro_f1 ?? 0).toFixed(1)}`).join("");
  const yticks = [y0, (y0 + y1) / 2, y1];
  const pick = sweep.find((d) => Math.abs(d.threshold - chosen) < 1e-6);
  return `<svg id="sweep" viewBox="0 0 ${W} ${H}" style="width:100%;height:auto" role="img" aria-label="Macro-F1 versus threshold">
    ${yticks.map((t) => `<line x1="${PAD.l}" x2="${W - PAD.r}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--grid)" stroke-width="1"/>
      <text x="${PAD.l - 8}" y="${sy(t) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${(t * 100).toFixed(0)}%</text>`).join("")}
    ${[x0, (x0 + x1) / 2, x1].map((t) => `<text x="${sx(t)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--muted)">${t.toFixed(2)}</text>`).join("")}
    <line x1="${PAD.l}" x2="${W - PAD.r}" y1="${H - PAD.b}" y2="${H - PAD.b}" stroke="var(--border-strong)"/>
    <path d="${path}" fill="none" stroke="var(--cat)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    ${pick ? `<line x1="${sx(pick.threshold)}" x2="${sx(pick.threshold)}" y1="${PAD.t}" y2="${H - PAD.b}" stroke="var(--muted)" stroke-dasharray="3 3"/>
      <circle cx="${sx(pick.threshold)}" cy="${sy(pick.macro_f1)}" r="5" fill="var(--cat)" stroke="var(--surface)" stroke-width="2"/>
      <text x="${sx(pick.threshold) + 8}" y="${sy(pick.macro_f1) - 8}" font-size="11.5" font-weight="600" fill="var(--text)">chosen ${pick.threshold.toFixed(2)} · ${pct(pick.macro_f1)}</text>` : ""}
    <line id="sweep-cross" y1="${PAD.t}" y2="${H - PAD.b}" stroke="var(--border-strong)" visibility="hidden"/>
    <circle id="sweep-dot" r="4" fill="var(--cat)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>
    <rect id="sweep-hit" x="${PAD.l}" y="${PAD.t}" width="${W - PAD.l - PAD.r}" height="${H - PAD.t - PAD.b}" fill="transparent"/>
  </svg>`;
}

function wireSweepTooltip(sweep) {
  const svg = $("#sweep");
  if (!svg || !sweep.length) return;
  const hit = $("#sweep-hit"), cross = $("#sweep-cross"), dot = $("#sweep-dot");
  const xs = sweep.map((d) => d.threshold);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let current = sweep[0];
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const vx = ((e.clientX - r.left) / r.width) * W;
    const t = x0 + ((vx - PAD.l) / (W - PAD.l - PAD.r)) * (x1 - x0);
    current = sweep.reduce((a, d) => (Math.abs(d.threshold - t) < Math.abs(a.threshold - t) ? d : a));
    const px = PAD.l + ((current.threshold - x0) / (x1 - x0 || 1)) * (W - PAD.l - PAD.r);
    const ys = sweep.map((d) => d.macro_f1 ?? 0);
    const y0 = Math.max(0, Math.floor(Math.min(...ys) * 10) / 10);
    const py = H - PAD.b - (((current.macro_f1 ?? 0) - y0) / (1 - y0 || 1)) * (H - PAD.t - PAD.b);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", px); dot.setAttribute("cy", py); dot.setAttribute("visibility", "visible");
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); });
  tooltip(hit, () => `<strong>threshold ${current.threshold.toFixed(2)}</strong>
    <div class="row"><span>macro-F1</span><strong>${pct(current.macro_f1)}</strong></div>
    <div class="row"><span>accuracy</span><strong>${pct(current.accuracy)}</strong></div>`);
}
