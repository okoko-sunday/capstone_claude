// Landing-page motion: scroll reveals, the example scan loop, pointer tilt,
// card spotlights, the particle field, and the page-wide drag glow.
// Purely decorative — nothing here touches predictions or metrics.

import { $, $$ } from "./util.js";

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

export function initMotion() {
  initReveal();
  initDragGlow();
  initSpotlight();
  if (reduced) {
    $("#scan-text").textContent = "Three classes: dog, cat, neither";
    return;
  }
  initTilt();
  initScanLoop();
  initParticles();
}

// ------------------------------------------------------------ scroll reveal
function initReveal() {
  const els = $$("[data-reveal]");
  if (!("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("is-in")); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("is-in"); io.unobserve(en.target); } });
  }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  els.forEach((e) => io.observe(e));
}

// ------------------------------------------- drag files anywhere = glow wakes
function initDragGlow() {
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes("Files");
  window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { depth++; document.body.classList.add("is-dragging"); } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) document.body.classList.remove("is-dragging"); });
  window.addEventListener("drop", () => { depth = 0; document.body.classList.remove("is-dragging"); });
}

// ---------------------------------------------------- feature card spotlight
function initSpotlight() {
  $$(".feature").forEach((card) => card.addEventListener("pointermove", (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--x", `${e.clientX - r.left}px`);
    card.style.setProperty("--y", `${e.clientY - r.top}px`);
  }));
}

// ------------------------------------------------------- 3D tilt on pointer
function initTilt() {
  const hero = $("#hero");
  const collage = $(".collage");
  if (!hero || !collage || !matchMedia("(pointer: fine)").matches) return;
  let raf = 0;
  hero.addEventListener("pointermove", (e) => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      const r = collage.getBoundingClientRect();
      const mx = Math.max(-1, Math.min(1, (e.clientX - (r.left + r.width / 2)) / (r.width / 1.2)));
      const my = Math.max(-1, Math.min(1, (e.clientY - (r.top + r.height / 2)) / (r.height / 1.2)));
      collage.style.setProperty("--mx", mx.toFixed(3));
      collage.style.setProperty("--my", my.toFixed(3));
    });
  });
  hero.addEventListener("pointerleave", () => { collage.style.setProperty("--mx", 0); collage.style.setProperty("--my", 0); });
}

// ------------------------------------------------- example scan, on a loop
function initScanLoop() {
  const order = ["dog", "cat", "neither"];
  const text = $("#scan-text");
  const nice = { dog: "Dog", cat: "Cat", neither: "Neither — no pet here" };
  let k = 0;
  let timer = 0;
  const visible = () => document.visibilityState === "visible" && $("#view-classify").classList.contains("is-active") && window.scrollY < 700;

  const step = () => {
    if (!visible()) { timer = setTimeout(step, 1200); return; }
    const label = order[k % order.length];
    const shot = $(`.shot[data-label="${label}"]`);
    const tag = $(`.tag[data-for="${label}"]`);
    $$(".shot.is-scanning").forEach((s) => s.classList.remove("is-scanning"));
    $$(".tag.is-found").forEach((t) => t.classList.remove("is-found"));
    void shot.offsetWidth; // restart the CSS animations
    shot.classList.add("is-scanning");
    swap(text, `Scanning example ${(k % order.length) + 1} of 3…`);
    setTimeout(() => { tag.classList.add("is-found"); swap(text, `Example: ${nice[label]} ✓`); }, 1250);
    setTimeout(() => shot.classList.remove("is-scanning"), 2600);
    k++;
    timer = setTimeout(step, 3200);
  };
  timer = setTimeout(step, 1900); // after the intro has landed
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && !timer) step(); });
}

function swap(el, value) {
  el.style.opacity = "0";
  setTimeout(() => { el.textContent = value; el.style.opacity = "1"; }, 180);
}

// ------------------------------------------------------------ particle field
function initParticles() {
  const canvas = $("#particles");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const hero = $("#hero");
  let W = 0, H = 0, dpr = 1, dots = [], running = true, frame = 0;
  const mouse = { x: -9999, y: -9999 };

  const colour = () => (getComputedStyle(document.documentElement).getPropertyValue("--blue").trim() || "#2f5bea");
  let rgb = hexToRgb(colour());

  const resize = () => {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(80, Math.round((W * H) / 9000));
    dots = Array.from({ length: n }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35,
      r: Math.random() * 1.6 + 0.6, tw: Math.random() * Math.PI * 2,
    }));
  };

  const tick = () => {
    if (!running) return;
    frame++;
    ctx.clearRect(0, 0, W, H);
    const [R, G, B] = rgb;
    for (const d of dots) {
      // Gentle drift, a soft push away from the pointer, wrap at the edges.
      const dx = d.x - mouse.x, dy = d.y - mouse.y, dist2 = dx * dx + dy * dy;
      if (dist2 < 140 * 140) { const f = (1 - Math.sqrt(dist2) / 140) * 0.6; d.vx += (dx / 140) * f * 0.08; d.vy += (dy / 140) * f * 0.08; }
      d.vx *= 0.985; d.vy *= 0.985;
      d.vx += (Math.random() - 0.5) * 0.01; d.vy += (Math.random() - 0.5) * 0.01;
      d.x += d.vx; d.y += d.vy;
      if (d.x < -10) d.x = W + 10; if (d.x > W + 10) d.x = -10;
      if (d.y < -10) d.y = H + 10; if (d.y > H + 10) d.y = -10;
    }
    for (let i = 0; i < dots.length; i++) {
      const a = dots[i];
      for (let j = i + 1; j < dots.length; j++) {
        const b = dots[j];
        const dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
        if (d < 110) {
          ctx.strokeStyle = `rgba(${R},${G},${B},${(1 - d / 110) * 0.16})`;
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
      const near = Math.hypot(a.x - mouse.x, a.y - mouse.y) < 160;
      const tw = 0.45 + Math.sin(frame * 0.03 + a.tw) * 0.25;
      ctx.fillStyle = `rgba(${R},${G},${B},${near ? 0.85 : tw})`;
      ctx.beginPath(); ctx.arc(a.x, a.y, near ? a.r + 0.8 : a.r, 0, Math.PI * 2); ctx.fill();
    }
    requestAnimationFrame(tick);
  };

  hero.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top;
  });
  hero.addEventListener("pointerleave", () => { mouse.x = mouse.y = -9999; });

  // Only animate while the hero is on screen and the tab is visible.
  const setRunning = (on) => { if (on && !running) { running = true; requestAnimationFrame(tick); } else if (!on) running = false; };
  new IntersectionObserver(([en]) => setRunning(en.isIntersecting)).observe(hero);
  document.addEventListener("visibilitychange", () => setRunning(document.visibilityState === "visible"));
  new ResizeObserver(resize).observe(hero);
  new MutationObserver(() => (rgb = hexToRgb(colour()))).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  resize();
  requestAnimationFrame(tick);
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const v = h.length === 3 ? h.split("").map((c) => c + c).join("") : h.slice(0, 6);
  const n = parseInt(v, 16);
  return Number.isNaN(n) ? [47, 91, 234] : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
