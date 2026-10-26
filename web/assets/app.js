// Entry point: wires the views to the store and polls the backend status.

import { api } from "./js/api.js";
import { emit, set, state } from "./js/store.js";
import { $, $$, toast } from "./js/util.js";
import { initClassify } from "./js/views/classify.js";
import { initEvaluate } from "./js/views/evaluate.js";
import { initModelCard } from "./js/views/model-card.js";

function initTabs() {
  const show = (view) => {
    $$(".tab").forEach((t) => {
      const active = t.dataset.view === view;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", active);
    });
    $$(".view").forEach((v) => v.classList.toggle("is-active", v.id === `view-${view}`));
    const tab = $(`.tab[data-view="${view}"] span`);
    if (tab) $("#crumb").textContent = tab.textContent;
    set({ view }, "view");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  $$(".tab").forEach((t) => t.addEventListener("click", () => show(t.dataset.view)));
  document.addEventListener("click", (e) => {
    const link = e.target.closest("[data-goto]");
    if (link) { e.preventDefault(); show(link.dataset.goto); }
  });
}

function initTheme() {
  const btn = $("#theme-toggle");
  const sync = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    btn.querySelector("use").setAttribute("href", dark ? "#i-sun" : "#i-moon");
    return dark;
  };
  btn.addEventListener("click", () => {
    const next = sync() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("catdog-theme", next); } catch { /* private mode */ }
    sync();
    emit("theme");
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { sync(); emit("theme"); });
  sync();
}

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
      $("#fact-backbone").textContent = model.backbone ?? "custom";
      const tau = model.policy?.animal_threshold;
      $("#fact-tau").textContent = tau != null ? tau.toFixed(2) : "—";
      $("#status-text").textContent = `Model ready · ${model.backbone ?? "custom"}${acc != null ? ` · val ${(acc * 100).toFixed(1)}%` : ""}`;
    } else {
      $("#status-text").textContent = "No model loaded";
      if (health.detail) $("#no-model-detail").textContent = health.detail;
    }
  } catch {
    pill.dataset.state = "offline";
    $("#status-text").textContent = "Server offline";
  }
}

function initHero() {
  $("#today").textContent = new Date().toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
  $("#hero-cta").addEventListener("click", (e) => {
    e.preventDefault();
    const zone = $("#dropzone");
    zone.scrollIntoView({ behavior: "smooth", block: "center" });
    zone.focus({ preventScroll: true });
  });
}

initTabs();
initTheme();
initHero();
initClassify();
initEvaluate();
initModelCard();
refreshStatus();

$("#reload-model").addEventListener("click", async () => {
  try {
    const h = await api.reload();
    await refreshStatus();
    toast(h.model_loaded ? "Model loaded" : "Still no model found");
  } catch (err) {
    toast(err.message);
  }
});

window.__catdog = { state }; // handy for debugging in the console
