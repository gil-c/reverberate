/** The timeline's arithmetic: seconds to pixels, the ruler, where each lane is.
 *
 * Kept apart from the drawing so it can be checked without a browser.
 */

/** The shortest stretch of a scene the timeline zooms into, in seconds. */
export const MIN_SPAN_S = 5;

/** A window `[start, start + span]` of a scene of `duration`, over `width` pixels. */
export function createTimeMap({ duration, width, start = 0, span = duration }) {
  const total = Math.max(duration, 1e-9);
  const shown = Math.min(total, Math.max(Math.min(MIN_SPAN_S, total), span));
  const first = Math.min(total - shown, Math.max(0, start));
  const pixels = Math.max(width, 1);
  return {
    duration: total,
    width: pixels,
    start: first,
    span: shown,
    end: first + shown,
    toX: (t) => ((t - first) / shown) * pixels,
    /** The time under a pixel, kept inside the scene: a drag past an edge seeks to it. */
    toT: (x) => Math.min(total, Math.max(0, first + (x / pixels) * shown)),
  };
}

/** The window `factor` times as long, the time under pixel `x` staying under it. */
export function zoomed(map, factor, x) {
  const span = Math.min(map.duration, Math.max(Math.min(MIN_SPAN_S, map.duration), map.span * factor));
  const held = map.start + (x / map.width) * map.span;
  return createTimeMap({
    duration: map.duration,
    width: map.width,
    span,
    start: held - (x / map.width) * span,
  });
}

/** The window moved by `pixels`, stopped by the scene's ends. */
export function panned(map, pixels) {
  return createTimeMap({
    duration: map.duration,
    width: map.width,
    span: map.span,
    start: map.start + (pixels / map.width) * map.span,
  });
}

/** The same window over another width. */
export function resized(map, width) {
  return createTimeMap({ duration: map.duration, width, span: map.span, start: map.start });
}

const TICK_STEPS_S = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];

/** `m:ss`, with tenths when `tenths` is set. */
export function formatTime(t, tenths = false) {
  const whole = Math.max(0, t);
  const minutes = Math.floor(whole / 60);
  const seconds = whole - minutes * 60;
  const text = tenths ? seconds.toFixed(1).padStart(4, "0") : String(Math.floor(seconds)).padStart(2, "0");
  return `${minutes}:${text}`;
}

/** The ruler's marks: the finest round step that leaves `minPixels` between two. */
export function ticks(map, minPixels = 64) {
  const wanted = (minPixels / map.width) * map.span;
  const step = TICK_STEPS_S.find((candidate) => candidate >= wanted) || TICK_STEPS_S[TICK_STEPS_S.length - 1];
  const marks = [];
  for (let k = Math.ceil(map.start / step - 1e-9); k * step <= map.end + 1e-9; k++) {
    const t = k * step;
    marks.push({ t, x: map.toX(t), label: formatTime(t, step < 1) });
  }
  return { step, marks };
}

/** Where each lane is drawn: one row per id, top to bottom.
 *
 * A row holds two bands: `activity` (when the source emits) over `movement`
 * (what it does). The listener emits nothing, so its row is all movement.
 */
export function laneLayout(ids, { laneHeight = 22, gap = 2, top = 0, listener = "listener" } = {}) {
  const inner = laneHeight - gap;
  const upper = Math.round(inner * 0.55);
  const rows = ids.map((id, index) => {
    const y = top + index * laneHeight;
    const whole = { y: y + gap / 2, height: inner };
    return {
      id,
      index,
      y,
      height: laneHeight,
      activity: id === listener ? null : { y: whole.y, height: upper },
      movement: id === listener ? whole : { y: whole.y + upper + 1, height: inner - upper - 1 },
    };
  });
  return {
    rows,
    height: top + ids.length * laneHeight,
    /** The lane under a pixel, or null above the first and below the last. */
    at(y) {
      const index = Math.floor((y - top) / laneHeight);
      return index >= 0 && index < rows.length ? rows[index] : null;
    },
  };
}

/** The spans `[start_s, end_s]` that touch `[from, to]`, the spans being in time order. */
export function visible(spans, from, to, startOf = (s) => s.start_s, endOf = (s) => s.end_s) {
  let low = 0;
  let high = spans.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (endOf(spans[middle]) < from) low = middle + 1;
    else high = middle;
  }
  const found = [];
  for (let i = low; i < spans.length && startOf(spans[i]) <= to; i++) found.push(spans[i]);
  return found;
}

/** How a movement span is drawn: its posture at each end, 1 standing and 0 seated,
 *  and whether the source is under way. */
export function posture(span) {
  if (span.type === "rise") {
    const up = span.to === "standing";
    return { from: up ? 0 : 1, to: up ? 1 : 0, moving: true };
  }
  const level = span.height === "seated" ? 0 : 1;
  return { from: level, to: level, moving: span.type === "travel" || span.type === "walk" };
}
