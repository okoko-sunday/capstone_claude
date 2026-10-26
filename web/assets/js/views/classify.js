// Classify view: upload → predict → piles, grid/table, true-label tagging, export.

import { api } from "../api.js";
import { LABELS, emit, isUnsure, isWrong, on, set, state } from "../store.js";
import { $, $$, IMAGE_EXT, download, esc, filesFromDrop, guessTruth, icon, pct, toast, tooltip } from "../util.js";

const CHUNK = 12; // images per request: keeps progress smooth and requests small
let nextId = 1;

export function initClassify() {
  const zone = $("#dropzone");
  const fileInput = $("#file-input");
  const folderInput = $("#folder-input");

  $("#pick-files").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  $("#pick-folder").addEventListener("click", (e) => { e.stopPropagation(); folderInput.click(); });
  zone.addEventListener("click", () => fileInput.click());
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });

  const fromInput = (input) => () => {
    const files = [...input.files].map((file) => ({ file, name: file.webkitRelativePath || file.name }));
    input.value = "";
    addFiles(files);
  };
  fileInput.addEventListener("change", fromInput(fileInput));
  folderInput.addEventListener("change", fromInput(folderInput));

  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, () => zone.classList.remove("is-over")));
  zone.addEventListener("drop", async (e) => { e.preventDefault(); addFiles(await filesFromDrop(e.dataTransfer)); });
  // Dropping anywhere on the page should not navigate away from the app.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("#filters").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) set({ filter: chip.dataset.filter }, "filter");
  });
  $$(".seg").forEach((b) => b.addEventListener("click", () => set({ layout: b.dataset.layout }, "layout")));
  $("#export-csv").addEventListener("click", exportCsv);
  $("#copy-table").addEventListener("click", copyTable);
  $("#clear").addEventListener("click", clearAll);

  on("items", renderAll);
  on("item", renderItem);
  on("filter", renderAll);
  on("layout", renderAll);
  on("search", renderAll);
}

// ----------------------------------------------------------------- upload
async function addFiles(files) {
  const images = files.filter((f) => IMAGE_EXT.test(f.name));
  const skipped = files.length - images.length;
  if (!images.length) return toast(skipped ? `No images found (${skipped} non-image files skipped)` : "No files selected");
  if (state.health && !state.health.model_loaded) return toast("No model loaded — see the banner above");

  const fresh = images.map(({ file, name }) => ({
    id: nextId++, name, file, url: URL.createObjectURL(file), prediction: null, truth: guessTruth(name),
  }));
  set({ items: [...state.items, ...fresh] }, "items");

  const zone = $("#dropzone");
  zone.classList.add("is-busy");
  $("#progress").hidden = false;
  const started = performance.now();
  let done = 0;
  try {
    for (let i = 0; i < fresh.length; i += CHUNK) {
      const chunk = fresh.slice(i, i + CHUNK);
      progress(done, fresh.length);
      const res = await api.predict(chunk.map((it) => ({ name: it.name, file: it.file })));
      res.predictions.forEach((p, k) => (chunk[k].prediction = p));
      done += chunk.length;
      emit("items");
    }
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    progress(done, fresh.length, `Classified ${done} image${done === 1 ? "" : "s"} in ${secs}s${skipped ? ` · ${skipped} non-image files skipped` : ""}`);
  } catch (err) {
    progress(done, fresh.length, `Error: ${err.message}`);
    // Drop the items that never got a prediction.
    set({ items: state.items.filter((it) => it.prediction || !fresh.includes(it)) }, "items");
  } finally {
    zone.classList.remove("is-busy");
    setTimeout(() => ($("#progress").hidden = true), 900);
  }
}

function progress(done, total, text) {
  $("#progress-bar").style.width = `${(done / total) * 100}%`;
  $("#progress-text").textContent = text ?? `Classifying ${done} / ${total}…`;
}

// --------------------------------------------------------------- rendering
function visible() {
  const f = state.filter;
  const q = state.search;
  return state.items.filter((it) => {
    if (q && !it.name.toLowerCase().includes(q)) return false;
    if (f === "all") return true;
    if (!it.prediction) return false;
    if (f === "unsure") return isUnsure(it);
    if (f === "wrong") return isWrong(it);
    return it.prediction.label === f;
  });
}

function renderAll() {
  const hasItems = state.items.length > 0;
  $("#results-area").hidden = !hasItems;
  $$(".chip").forEach((c) => c.classList.toggle("is-active", c.dataset.filter === state.filter));
  $$(".seg").forEach((b) => b.classList.toggle("is-active", b.dataset.layout === state.layout));
  renderPiles();
  renderChipCounts();
  renderRecent();
  const items = visible();
  const grid = $("#grid");
  grid.hidden = state.layout !== "grid";
  $("#table-wrap").hidden = state.layout !== "table";
  if (state.layout === "grid") {
    grid.replaceChildren(...items.map(card));
    if (!items.length && hasItems) grid.innerHTML = `<p class="muted">No images match ${state.search ? `“${esc(state.search)}”` : "this filter"}.</p>`;
  } else {
    $("#table tbody").replaceChildren(...items.map(row));
  }
}

function renderItem(item) {
  const old = document.querySelector(`[data-id="${item.id}"]`);
  if (old) old.replaceWith(state.layout === "grid" ? card(item) : row(item));
  renderPiles();
  renderChipCounts();
  renderRecent();
}

// Newest first; clicking a thumbnail scrolls to its full result card.
function renderRecent() {
  const recent = [...state.items].reverse().slice(0, 12);
  $("#recent-empty").hidden = recent.length > 0;
  $("#recent").replaceChildren(...recent.map((it) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "thumb";
    const label = it.prediction?.label;
    b.innerHTML = `
      <div class="thumb__media"><img src="${it.url}" alt="" loading="lazy" /><span class="thumb__more" aria-hidden="true">···</span></div>
      <div class="thumb__foot"><i class="dot ${label && label !== "error" ? `dot--${label}` : ""}" style="${!label || label === "error" ? "background:var(--muted)" : ""}"></i><span title="${esc(it.name)}">${esc(it.name.split("/").pop())}</span></div>`;
    b.setAttribute("aria-label", `${it.name}: ${label ?? "classifying"}`);
    b.querySelector("img").addEventListener("error", (e) => e.target.remove());
    b.addEventListener("click", () => {
      if (state.filter !== "all" || state.search) { $("#search").value = ""; set({ filter: "all", search: "" }, "filter"); }
      requestAnimationFrame(() => document.querySelector(`[data-id="${it.id}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
    });
    return b;
  }));
}

function renderPiles() {
  const done = state.items.filter((i) => i.prediction);
  const count = (l) => done.filter((i) => i.prediction.label === l).length;
  const errors = count("error");
  const order = ["dog", "cat", "neither"]; // the brief's three piles
  const tagged = done.filter((i) => i.truth).length;
  const correct = done.filter((i) => i.truth && !isWrong(i)).length;
  $("#piles").innerHTML =
    order.map((l) => `
      <button class="pile pile--${l}" data-filter="${l}">
        <span class="pile__label"><span class="pile__swatch"></span>${l}</span>
        <div class="pile__value">${count(l)}</div>
        <span class="pile__sub">${done.length ? pct(count(l) / done.length, 0) : "—"} of images</span>
      </button>`).join("") +
    `<div class="pile pile--meta">
        <span class="pile__label">${tagged ? "Correct so far" : "Processed"}</span>
        <div class="pile__value">${tagged ? `${correct}<span class="muted" style="font-size:20px">/${tagged}</span>` : done.length}</div>
        <span class="pile__sub">${tagged ? `${state.items.length - tagged} without a true label` : `${state.items.length - done.length} pending`}${errors ? ` · ${errors} unreadable` : ""}</span>
      </div>`;
  $$("#piles .pile[data-filter]").forEach((b) => b.addEventListener("click", () => set({ filter: b.dataset.filter }, "filter")));
}

function renderChipCounts() {
  const n = {
    all: state.items.length,
    unsure: state.items.filter(isUnsure).length,
    wrong: state.items.filter(isWrong).length,
  };
  for (const l of LABELS) n[l] = state.items.filter((i) => i.prediction?.label === l).length;
  $$(".chip").forEach((c) => {
    const label = c.dataset.filter[0].toUpperCase() + c.dataset.filter.slice(1);
    c.innerHTML = `${c.dataset.filter === "wrong" ? "Mistakes" : label}<span class="count">${n[c.dataset.filter]}</span>`;
  });
}

function probBar(p) {
  const bar = document.createElement("div");
  bar.className = "probbar";
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", LABELS.map((l) => `${l} ${pct(p.probabilities[l])}`).join(", "));
  bar.innerHTML = LABELS.map((l) => `<span class="p-${l}" style="width:${(p.probabilities[l] ?? 0) * 100}%"></span>`).join("");
  tooltip(bar, () =>
    LABELS.map((l) => `<div class="row"><span><i style="background:var(--${l})"></i>${l}</span><strong>${pct(p.probabilities[l])}</strong></div>`).join("") +
    (p.decided_by === "threshold" ? `<div class="muted" style="margin-top:6px">Strongest animal score was below the threshold, so this was called “neither”.</div>` : ""));
  return bar;
}

function truthControl(item) {
  const wrap = document.createElement("div");
  wrap.className = "truth";
  wrap.innerHTML = `<span>True label</span><div class="truth__opts" role="group" aria-label="True label for ${esc(item.name)}">${LABELS.map(
    (l) => `<button type="button" data-l="${l}" aria-pressed="${item.truth === l}">${l}</button>`).join("")}</div>`;
  wrap.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    item.truth = item.truth === b.dataset.l ? null : b.dataset.l;
    emit("item", item);
    emit("truth");
  });
  return wrap;
}

function verdict(item) {
  if (!item.prediction || !item.truth || item.prediction.label === "error") return "";
  return isWrong(item)
    ? `<span class="verdict verdict--wrong">${icon("x")}Wrong</span>`
    : `<span class="verdict verdict--right">${icon("check")}Right</span>`;
}

function card(item) {
  const p = item.prediction;
  const el = document.createElement("article");
  el.className = `card result${isWrong(item) ? " is-wrong" : ""}`;
  el.dataset.id = item.id;
  el.innerHTML = `
    <div class="result__media">
      <img src="${item.url}" alt="${esc(item.name)}" loading="lazy" />
      ${p ? `<span class="badge badge--${p.label}">${p.label}</span>` : `<span class="badge" style="--c:var(--muted)">…</span>`}
      ${verdict(item)}
    </div>
    <div class="result__body">
      <div class="result__name" title="${esc(item.name)}">${esc(item.name)}</div>
      ${p && p.label !== "error" ? `
        <div class="result__conf"><span>confidence</span><strong>${pct(p.confidence)}</strong></div>` : ""}
      ${p?.label === "error" ? `<div class="flag">${icon("alert")}${esc(p.error)}</div>` : ""}
      ${p?.decided_by === "threshold" ? `<span class="flag">${icon("alert")}Below animal threshold</span>` : ""}
    </div>`;
  const img = el.querySelector("img");
  img.addEventListener("error", () => {
    // e.g. HEIC: the server can read it, the browser cannot preview it.
    img.replaceWith(Object.assign(document.createElement("div"), { className: "fallback", innerHTML: `${icon("image")}No preview` }));
  });
  const body = el.querySelector(".result__body");
  if (p && p.label !== "error") {
    body.append(probBar(p));
    const legend = document.createElement("div");
    legend.className = "problegend";
    legend.innerHTML = LABELS.map((l) => `<span><i style="background:var(--${l})"></i>${l} ${pct(p.probabilities[l], 0)}</span>`).join("");
    body.append(legend);
  }
  body.append(truthControl(item));
  return el;
}

function row(item) {
  const p = item.prediction;
  const tr = document.createElement("tr");
  tr.dataset.id = item.id;
  const index = state.items.indexOf(item) + 1;
  tr.innerHTML = `
    <td class="num muted">${index}</td>
    <td class="mono">${esc(item.name)}</td>
    <td>${p ? `<span class="badge badge--${p.label}">${p.label}</span>` : "…"}</td>
    <td class="num">${p && p.label !== "error" ? pct(p.confidence) : "—"}</td>
    ${LABELS.map((l) => `<td class="num">${p?.probabilities?.[l] != null ? p.probabilities[l].toFixed(3) : "—"}</td>`).join("")}
    <td><select aria-label="True label">${["", ...LABELS].map((l) => `<option value="${l}" ${item.truth === (l || null) ? "selected" : ""}>${l || "—"}</option>`).join("")}</select></td>
    <td>${item.truth && p && p.label !== "error" ? (isWrong(item) ? `<span class="verdict--wrong">${icon("x")}</span>` : `<span class="verdict--right">${icon("check")}</span>`) : ""}</td>`;
  tr.querySelector("select").addEventListener("change", (e) => {
    item.truth = e.target.value || null;
    emit("item", item);
    emit("truth");
  });
  return tr;
}

// ----------------------------------------------------------------- export
function done() { return state.items.filter((i) => i.prediction); }

function exportCsv() {
  const rows = [["filename", "prediction", "confidence", "p_cat", "p_dog", "p_neither", "decided_by", "true_label"]];
  for (const it of done()) {
    const p = it.prediction;
    rows.push([it.name, p.label, p.confidence.toFixed(4), ...LABELS.map((l) => (p.probabilities[l] ?? "").toString()), p.decided_by, it.truth ?? ""]);
  }
  const csv = rows.map((r) => r.map((v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)).join(",")).join("\n");
  download(`predictions-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`, csv);
}

async function copyTable() {
  const items = done();
  const w = Math.max(8, ...items.map((i) => i.name.length)) + 2;
  const lines = [`${"filename".padEnd(w)}${"prediction".padEnd(12)}${"confidence".padStart(10)}`];
  for (const it of items) {
    const conf = it.prediction.label === "error" ? "—" : pct(it.prediction.confidence);
    lines.push(`${it.name.padEnd(w)}${it.prediction.label.padEnd(12)}${conf.padStart(10)}`);
  }
  try {
    await navigator.clipboard.writeText(lines.join("\n"));
    toast("Table copied to clipboard");
  } catch {
    toast("Clipboard not available in this browser");
  }
}

function clearAll() {
  state.items.forEach((i) => URL.revokeObjectURL(i.url));
  set({ items: [], filter: "all", cellFilter: null }, "items");
  emit("truth");
  $("#progress-text").textContent = "";
}
