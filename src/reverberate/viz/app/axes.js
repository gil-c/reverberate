/** Graduated axes drawn beside a plot canvas, never on it.
 *
 * The plot's pixels are the datum: the axes live in their own elements, a
 * gutter left of the canvas and a strip under it, with the tick marks
 * pointing outward from the plot's edge. Labels are compact (250, 1k, 16k;
 * 0, 0.5, 1; -30) and the unit is written once, at the far end of its axis.
 *
 * A position is a fraction of the canvas's content box, measured from its
 * top (y) or its left (x); the canvas's 1 px border is allowed for.
 */

/** "250", "1k", "16k", "1.5k": a frequency in hertz, short. */
export function compactHz(hz) {
  if (Math.abs(hz) < 1000) return `${Math.round(hz)}`;
  const k = hz / 1000;
  return `${Number.isInteger(k) ? k : Number(k.toFixed(1))}k`;
}

/** "0", "0.2", "1", "1.5": a time in seconds, short. */
export function compactSeconds(s) {
  return `${Number(s.toFixed(2))}`;
}

/** A step of 1, 2, 4 or 5 times a power of ten giving about `count` intervals over `span`. */
export function niceStep(span, count) {
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 4, 5, 10]) if (m * power >= raw) return m * power;
  return 10 * power;
}

/** Ticks of `step` from `from` up to, not reaching, `to` minus half a step: the end is the unit's. */
export function ticksUpTo(from, to, step) {
  const out = [];
  for (let k = 0; from + k * step < to - step / 2 + 1e-9; k++) out.push(from + k * step);
  return out;
}

/** Wrap `canvas` in a figure with a y gutter and an x strip; returns the setters. */
export function attachAxes(canvas) {
  const figure = document.createElement("div");
  figure.className = "axes";
  canvas.replaceWith(figure);
  const y = document.createElement("div");
  y.className = "axis y";
  const x = document.createElement("div");
  x.className = "axis x";
  const corner = document.createElement("div");
  figure.append(y, canvas, corner, x);

  /** Ticks at fractions `at`, labelled `text`; the last entry may be the unit alone. */
  const fill = (host, marks) => {
    host.replaceChildren(
      ...marks.map(({ at, text, unit }) => {
        const mark = document.createElement("div");
        mark.className = unit ? "mark unit" : "mark";
        // The content box lies 1 px inside the border on each side.
        mark.style.setProperty("--at", `calc(1px + ${at} * (100% - 2px))`);
        const label = document.createElement("span");
        label.textContent = text;
        mark.append(label);
        return mark;
      })
    );
  };

  return {
    figure,
    /** `marks` for the vertical axis, fractions from the top. */
    setY: (marks) => fill(y, marks),
    /** `marks` for the horizontal axis, fractions from the left. */
    setX: (marks) => fill(x, marks),
    clear() {
      y.replaceChildren();
      x.replaceChildren();
    },
  };
}

/** The time axis of a plot of `seconds`: 0, a step, ..., and "s" at the end. */
export function timeMarks(seconds) {
  if (!(seconds > 0)) return [];
  const step = niceStep(seconds, 6);
  return [
    ...ticksUpTo(0, seconds, step).map((s) => ({ at: s / seconds, text: compactSeconds(s) })),
    { at: 1, text: "s", unit: true },
  ];
}

/** The frequency axis of a linear spectrogram to `nyquist`: 0 at the bottom, "Hz" at the top. */
export function frequencyMarks(nyquist) {
  if (!(nyquist > 0)) return [];
  const step = niceStep(nyquist, 6);
  return [
    ...ticksUpTo(0, nyquist, step).map((hz) => ({ at: 1 - hz / nyquist, text: compactHz(hz) })),
    { at: 0, text: "Hz", unit: true },
  ];
}

/** The level axis of a plot `rangeDb` deep: 0 at the top, "dB" at the bottom. */
export function levelMarks(rangeDb, step = 30) {
  return [
    ...ticksUpTo(0, rangeDb, step).map((db) => ({ at: db / rangeDb, text: db ? `-${db}` : "0" })),
    { at: 1, text: "dB", unit: true },
  ];
}
