/**
 * NEXUS motion module.
 *
 * Design principles:
 * - Animate only transform, opacity, and stroke-dashoffset (never layout properties).
 * - Prefer CSS + Web Animations API. No external animation library (WAAPI covers all
 *   catalog items without adding bundle size).
 * - Every animation has: purpose, trigger, cleanup, and a stable data-anim="<id>" selector.
 * - Motion tokens are divided by the playback speed setting.
 */

// Motion tokens (milliseconds)
export const DURATION = {
  fast: 120,
  base: 240,
  slow: 480,
  messageTravel: 700, // full message travel at 1x speed
};

// Easings
export const EASING = {
  out: "cubic-bezier(0.16, 1, 0.3, 1)",      // entrances
  inOut: "cubic-bezier(0.4, 0, 0.2, 1)",     // travel
  spring: "cubic-bezier(0.34, 1.56, 0.64, 1)", // deal bursts
};

// Max concurrent particles; coalesce when speed is high.
export const MAX_PARTICLES = 6;

// ------------------------------------------------------------
// Speed scaling
// ------------------------------------------------------------
export function scaledDuration(ms, speed = 1) {
  return Math.max(60, Math.round(ms / speed));
}

// ------------------------------------------------------------
// Particle manager (caps concurrent particles, tracks WAAPI animations)
// ------------------------------------------------------------
export class ParticleManager {
  constructor() {
    this.active = new Map(); // id => Animation
    this.sequence = 0;
  }

  /**
   * Register a WAAPI animation under a stable id. Returns the id.
   * Caps at MAX_PARTICLES; oldest are cancelled when full.
   */
  register(id, animation) {
    if (this.active.has(id)) {
      this.active.get(id)?.cancel();
      this.active.delete(id);
    }
    if (this.active.size >= MAX_PARTICLES) {
      const oldest = this.active.keys().next().value;
      this.active.get(oldest)?.cancel();
      this.active.delete(oldest);
    }
    this.active.set(id, animation);
    animation.onfinish = () => this.active.delete(id);
    animation.oncancel = () => this.active.delete(id);
    return id;
  }

  cancel(id) {
    this.active.get(id)?.cancel();
    this.active.delete(id);
  }

  cancelAll() {
    for (const [id, anim] of this.active) {
      anim.cancel();
    }
    this.active.clear();
  }

  get count() {
    return this.active.size;
  }
}

export const particleManager = new ParticleManager();

// ------------------------------------------------------------
// Step-request deduplication (AbortController + sequence number)
// ------------------------------------------------------------
export class StepRequester {
  constructor() {
    this.controller = null;
    this.seq = 0;
    this.lastCompletedSeq = -1;
  }

  async step(episodeId, apiFn) {
    // Cancel any in-flight request
    if (this.controller) {
      this.controller.abort();
    }
    this.controller = new AbortController();
    const mySeq = ++this.seq;

    try {
      const res = await apiFn({ signal: this.controller.signal });
      // Ignore stale / out-of-order responses
      if (mySeq < this.lastCompletedSeq) return null;
      this.lastCompletedSeq = mySeq;
      return res;
    } catch (err) {
      if (err.name === "AbortError") return null;
      throw err;
    } finally {
      if (mySeq === this.seq) this.controller = null;
    }
  }

  cancelAll() {
    if (this.controller) {
      this.controller.abort();
      this.controller = null;
    }
    this.seq = 0;
    this.lastCompletedSeq = -1;
  }
}

export const stepRequester = new StepRequester();

// ------------------------------------------------------------
// Visibility pause / resume
// ------------------------------------------------------------
let paused = false;
const resumeCallbacks = [];

export function pauseAnimations() {
  paused = true;
  document.documentElement.classList.add("nexus-paused");
}

export function resumeAnimations() {
  paused = false;
  document.documentElement.classList.remove("nexus-paused");
  for (const cb of resumeCallbacks) cb();
  resumeCallbacks.length = 0;
}

export function isPaused() {
  return paused;
}

export function onResume(cb) {
  resumeCallbacks.push(cb);
}

if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pauseAnimations();
    else resumeAnimations();
  });
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
export function prefersReducedMotion() {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function easeFor(kind) {
  return EASING[kind] || EASING.out;
}

export function toCssDuration(ms) {
  return `${ms}ms`;
}

// Clean up on page unload
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    particleManager.cancelAll();
    stepRequester.cancelAll();
  });
}
