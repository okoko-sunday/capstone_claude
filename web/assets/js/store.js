// Minimal observable store: one state object, change events by topic.
//
// item = { id, name, file, url, prediction|null, truth: "cat"|"dog"|"neither"|null }

const listeners = new Map();

export const state = {
  items: [],
  filter: "all",
  layout: "grid",
  view: "classify",
  health: null,
  model: null,
  reports: null,
  cellFilter: null, // { true, predicted } chosen by clicking a matrix cell
};

export function on(topic, fn) {
  if (!listeners.has(topic)) listeners.set(topic, new Set());
  listeners.get(topic).add(fn);
}

export function emit(topic, payload) {
  for (const fn of listeners.get(topic) ?? []) fn(payload);
}

export function set(patch, topic) {
  Object.assign(state, patch);
  emit(topic ?? Object.keys(patch)[0], patch);
}

export const LABELS = ["cat", "dog", "neither"];

export const isWrong = (item) => item.prediction && item.truth && item.prediction.label !== item.truth;
export const isUnsure = (item) =>
  item.prediction && item.prediction.label !== "error" &&
  (item.prediction.confidence < 0.7 || item.prediction.decided_by === "threshold");
export const labelled = () => state.items.filter((i) => i.prediction && i.truth);
