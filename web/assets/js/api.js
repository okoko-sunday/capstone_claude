// Thin client for the FastAPI backend. Every network call goes through here.

async function request(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  model: () => request("/api/model"),
  reports: () => request("/api/reports"),
  reload: () => request("/api/model/reload", { method: "POST" }),

  /** files: [{name, file}] → predictions in the same order. */
  async predict(files) {
    const form = new FormData();
    for (const { name, file } of files) form.append("files", file, name);
    return request("/api/predict", { method: "POST", body: form });
  },

  /** items: [{true, predicted}] → confusion matrix + per-class metrics. */
  evaluate: (items) =>
    request("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    }),
};
