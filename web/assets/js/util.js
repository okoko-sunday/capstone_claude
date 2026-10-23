// Small DOM and formatting helpers shared by the views.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

export const pct = (v, digits = 1) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
export const icon = (id, cls = "") => `<svg class="${cls}" aria-hidden="true"><use href="#i-${id}"/></svg>`;

export const IMAGE_EXT = /\.(jpe?g|png|bmp|gif|webp|tiff?|heic|heif)$/i;

/** Guess the true label from a path like "dog/x.jpg", "cats/IMG_1.jpg" or "neither_03.png". */
export function guessTruth(path) {
  const tokens = path.toLowerCase().split(/[^a-z]+/).filter(Boolean);
  // Folder names are the most reliable signal, so look at them first.
  for (const t of tokens) {
    if (t === "cat" || t === "cats" || t === "kitten" || t === "kittens") return "cat";
    if (t === "dog" || t === "dogs" || t === "puppy" || t === "puppies") return "dog";
    if (["neither", "other", "others", "none", "negative", "negatives"].includes(t)) return "neither";
  }
  return null;
}

let toastTimer;
export function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 2600);
}

// One shared tooltip that follows the pointer.
export function tooltip(target, html) {
  const tip = $("#tooltip");
  const move = (e) => {
    const pad = 14;
    const { innerWidth: w, innerHeight: h } = window;
    const r = tip.getBoundingClientRect();
    tip.style.left = `${Math.min(e.clientX + pad, w - r.width - 8)}px`;
    tip.style.top = `${Math.min(e.clientY + pad, h - r.height - 8)}px`;
  };
  target.addEventListener("pointerenter", (e) => {
    tip.innerHTML = typeof html === "function" ? html() : html;
    tip.hidden = false;
    move(e);
  });
  target.addEventListener("pointermove", move);
  target.addEventListener("pointerleave", () => (tip.hidden = true));
}

export function download(filename, text, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const isDark = () => {
  const forced = document.documentElement.dataset.theme;
  return forced ? forced === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};

/** Recursively collect files from a drag-and-drop (supports dropped folders). */
export async function filesFromDrop(dataTransfer) {
  const entries = [...dataTransfer.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dataTransfer.files].map((file) => ({ file, name: file.name }));
  const out = [];
  const walk = async (entry, prefix) => {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ file, name: prefix + file.name });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      } while (batch.length);
    }
  };
  for (const entry of entries) await walk(entry, "");
  return out;
}
