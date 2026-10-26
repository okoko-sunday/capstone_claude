// Entry point: wires the views to the store and polls the backend status.

import { api } from "./js/api.js";
import { emit, set, state } from "./js/store.js";
import { $, $$, toast } from "./js/util.js";
import { initClassify } from "./js/views/classify.js";
import { initEvaluate } from "./js/views/evaluate.js";
import { initModelCard } from "./js/views/model-card.js";
import { initMotion } from "./js/motion.js";

function show(view, scroll) {
  $$(".tab").forEach((t) => {
    const active = t.dataset.view === view;
    t.classList.toggle("is-active", active);
    t.setAttribute("aria-selected", active);
  });
  $$(".view").forEach((v) => v.classList.toggle("is-active", v.id === `view-${view}`));
  const label = $(`.tab[data-view="${view}"] span`);
  if (label) $("#crumb").textContent = label.textContent;
  set({ view }, "view");
  const target = scroll && scroll !== "top" ? document.getElementById(scroll) : null;
  if (target) requestAnimationFrame(() => target.scrollIntoView({ behavior: "smooth", block: "start" }));
  else window.scrollTo({ top: 0, behavior: "smooth" });
}

function initNav() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => show(t.dataset.view)));
  document.addEventListener("click", (e) => {
    const link = e.target.closest("[data-goto]");
    if (link) { e.preventDefault(); show(link.dataset.goto, link.dataset.scroll); }
  });
  $("#view-all").addEventListener("click", (e) => {
    e.preventDefault();
    const area = $("#results-area");
    if (!area.hidden) area.scrollIntoView({ behavior: "smooth", block: "start" });
    else toast("No images yet — drop some above");
  });
}

// ------------------------------------------------------------------ theme
function currentTheme() { return document.documentElement.dataset.theme || "system"; }
function applyTheme(choice) {
  if (choice === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
  try { choice === "system" ? localStorage.removeItem("catdog-theme") : localStorage.setItem("catdog-theme", choice); } catch { /* private mode */ }
  syncTheme();
  emit("theme");
}
function syncTheme() {
  const t = currentTheme();
  const dark = t === "system" ? matchMedia("(prefers-color-scheme: dark)").matches : t === "dark";
  $("#theme-light").setAttribute("aria-pressed", !dark);
  $("#theme-dark").setAttribute("aria-pressed", dark);
  $$("[data-theme-set]").forEach((b) => b.classList.toggle("is-active", b.dataset.themeSet === t));
}
function initTheme() {
  $("#theme-light").addEventListener("click", () => applyTheme("light"));
  $("#theme-dark").addEventListener("click", () => applyTheme("dark"));
  $$("[data-theme-set]").forEach((b) => b.addEventListener("click", () => applyTheme(b.dataset.themeSet)));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { syncTheme(); emit("theme"); });
  syncTheme();
}

// --------------------------------------------------------------- settings
function initSettings() {
  const dlg = $("#settings");
  const open = () => { syncTheme(); dlg.showModal(); };
  $("#open-settings").addEventListener("click", open);
  $$("[data-open-settings]").forEach((b) => b.addEventListener("click", open));
  $("#close-settings").addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  $$("#reload-model, [data-reload]").forEach((b) => b.addEventListener("click", reloadModel));
}

async function reloadModel() {
  try {
    const h = await api.reload();
    await refreshStatus();
    toast(h.model_loaded ? "Model loaded" : "Still no model found");
  } catch (err) {
    toast(err.message);
  }
}

// ----------------------------------------------------------------- search
function initSearch() {
  const input = $("#search");
  input.addEventListener("input", () => {
    if (state.view !== "classify") show("classify", "results-area");
    set({ search: input.value.trim().toLowerCase() }, "search");
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { input.value = ""; set({ search: "" }, "search"); input.blur(); }
  });
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); input.focus(); input.select(); }
  });
}

// ----------------------------------------------------------------- status
async function refreshStatus() {
  const pill = $("#status-pill");
  try {
    const health = await api.health();
    set({ health }, "health");
    pill.dataset.state = health.status;
    $("#footer-version").textContent = `v${health.version}`;
    $("#no-model-banner").hidden = health.model_loaded;
    if (health.model_loaded) {
      const [model, reports] = await Promise.all([api.model(), api.reports().catch(() => null)]);
      set({ model }, "model");
      set({ reports }, "reports");
      const acc = model.validation?.tuned?.accuracy;
      const text = `Model ready · ${model.backbone ?? "custom"}${acc != null ? ` · val ${(acc * 100).toFixed(1)}%` : ""}`;
      $("#status-text").textContent = text;
      $("#settings-model").textContent = text;
    } else {
      $("#status-text").textContent = "No model loaded";
      $("#settings-model").textContent = "No model loaded";
      if (health.detail) $("#no-model-detail").textContent = health.detail;
    }
  } catch {
    pill.dataset.state = "offline";
    $("#status-text").textContent = "Server offline";
    $("#settings-model").textContent = "Server offline";
  }
}

initNav();
initTheme();
initSettings();
initSearch();
initClassify();
initEvaluate();
initModelCard();
initMotion();
refreshStatus();

window.__catdog = { state }; // handy for debugging in the console
