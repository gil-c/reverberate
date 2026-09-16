/** The three plots of the selected source's response at the listener's cell:
 * a spectrogram and the Schroeder decay of the omnidirectional channel, and
 * where the energy arrives from. The maths runs in a worker on the cell's own
 * channels, so the plots do not depend on the head or on what is decoding,
 * and change only when the cell does; the direction diagram turns with the
 * head.
 *
 * Levels are absolute, against the cell nearest the source: a cell farther
 * away plots lower, which is the point.
 */
import { headYawOfCamera } from "./audio/sh.js";

//: Decibels below the reference the spectrogram and the decay show.
const RANGE_DB = 90;
//: Decibels below its own peak the direction diagram shows.
const POLAR_DB = 30;
const GROUND = "#0e1217";
const LINE = "#262e38";
const EARLY = "#e8a33d";
const WHOLE = "#7c8796";
const LABEL = "rgba(216,222,230,.75)";
//: Seconds between the time ticks, chosen from the response's length.
const timeStep = (seconds) => (seconds > 2 ? 0.5 : seconds > 0.6 ? 0.2 : 0.1);
//: Label size in CSS pixels; the canvases draw at a larger backing size.
const LABEL_PX = 10;

/** A label on a canvas, kept inside it, with a dark backing for contrast. */
function label(g, text, x, y, align = "left", base = "top") {
  const scale = g.canvas.width / Math.max(1, g.canvas.clientWidth || g.canvas.width);
  const size = Math.round(LABEL_PX * scale);
  g.font = `${size}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  g.textAlign = align;
  g.textBaseline = base;
  const width = g.measureText(text).width;
  const left = align === "right" ? x - width : align === "center" ? x - width / 2 : x;
  const top = base === "bottom" ? y - size : base === "middle" ? y - size / 2 : y;
  g.fillStyle = "rgba(14,18,23,.7)";
  g.fillRect(left - scale, top - scale, width + 2 * scale, size + 2 * scale);
  g.fillStyle = LABEL;
  g.fillText(text, x, y);
}

/** Vertical lines and labels every ``timeStep`` along a plot of ``seconds``. */
function timeAxis(g, w, h, seconds, colour) {
  if (!(seconds > 0)) return;
  const step = timeStep(seconds);
  g.strokeStyle = colour;
  g.lineWidth = 1;
  for (let s = step; s < seconds - step / 4; s += step) {
    const x = (s / seconds) * w;
    g.beginPath();
    g.moveTo(x, 0);
    g.lineTo(x, h);
    g.stroke();
    label(g, `${s.toFixed(1)} s`, x + 2, h - 2, "left", "bottom");
  }
}

export function createPlots({ spectrogram, decay, direction, captions, workerUrl }) {
  const sg = spectrogram.getContext("2d");
  const dg = decay.getContext("2d");
  const pg = direction.getContext("2d");
  const worker = new Worker(workerUrl, { type: "module" });
  let nextId = 1;
  let shows = 0;
  let shown = null; // the latest analysis on screen
  const references = new WeakMap(); // field -> { ceilingDb, energyDb }
  let reference = { ceilingDb: 0, energyDb: 0 };
  let cameraYaw = 0;
  let seconds = 0;

  function clear() {
    for (const [g, canvas] of [[sg, spectrogram], [dg, decay], [pg, direction]]) {
      g.fillStyle = GROUND;
      g.fillRect(0, 0, canvas.width, canvas.height);
    }
    for (const caption of Object.values(captions)) caption.textContent = "";
    shown = null;
  }

  const analyse = (channels, order, earlySamples) =>
    new Promise((resolve) => {
      const id = nextId++;
      const onMessage = (event) => {
        if (event.data.id !== id) return;
        worker.removeEventListener("message", onMessage);
        resolve(event.data);
      };
      worker.addEventListener("message", onMessage);
      // Copies: the cell's buffer stays in the field's cache.
      worker.postMessage({ id, channels: channels.map((c) => Float32Array.from(c)), order, earlySamples });
    });

  function drawSpectrogram(spec, sampleRate) {
    const { frames, bins, levels } = spec;
    const image = new ImageData(frames, bins);
    for (let f = 0; f < frames; f++) {
      for (let b = 0; b < bins; b++) {
        const v = Math.max(0, Math.min(1, (levels[f * bins + b] - reference.ceilingDb + RANGE_DB) / RANGE_DB));
        const at = ((bins - 1 - b) * frames + f) * 4;
        image.data[at] = Math.round(20 + 235 * Math.pow(v, 1.4));
        image.data[at + 1] = Math.round(30 + 200 * Math.pow(v, 2.2));
        image.data[at + 2] = Math.round(60 + 120 * v);
        image.data[at + 3] = 255;
      }
    }
    const off = new OffscreenCanvas(frames, bins);
    off.getContext("2d").putImageData(image, 0, 0);
    sg.imageSmoothingEnabled = true;
    sg.drawImage(off, 0, 0, spectrogram.width, spectrogram.height);
    const nyquist = sampleRate / 2;
    sg.strokeStyle = "rgba(216,222,230,.3)";
    sg.lineWidth = 1;
    for (const hz of [4000, 8000, 12000, 16000, 20000]) {
      if (hz >= nyquist) continue;
      const y = spectrogram.height * (1 - hz / nyquist);
      sg.beginPath();
      sg.moveTo(0, y);
      sg.lineTo(spectrogram.width, y);
      sg.stroke();
      label(sg, `${hz / 1000} kHz`, 2, y, "left", "middle");
    }
    timeAxis(sg, spectrogram.width, spectrogram.height, seconds, "rgba(216,222,230,.18)");
    captions.spectrogram.textContent =
      `0 – ${seconds.toFixed(2)} s · 0 – ${(nyquist / 1000).toFixed(0)} kHz · ` +
      `${RANGE_DB} dB · peak ${shown.peakDb.toFixed(1)} dB`;
  }

  function drawDecay(curve) {
    const w = decay.width;
    const h = decay.height;
    dg.fillStyle = GROUND;
    dg.fillRect(0, 0, w, h);
    dg.strokeStyle = LINE;
    dg.lineWidth = 1;
    for (const db of [10, 30, 60, 90]) {
      const y = Math.min((h * db) / RANGE_DB, h - 1);
      dg.beginPath();
      dg.moveTo(0, y);
      dg.lineTo(w, y);
      dg.stroke();
    }
    timeAxis(dg, w, h, seconds, LINE);
    // The level scale is absolute: 0 dB is the energy at the cell nearest the source.
    for (const db of [10, 30, 60]) {
      label(dg, `-${db} dB`, w - 2, (h * db) / RANGE_DB, "right", "middle");
    }
    dg.strokeStyle = "#6fd18a";
    dg.lineWidth = 2;
    dg.beginPath();
    for (let i = 0; i < curve.length; i++) {
      const x = (i / (curve.length - 1)) * w;
      const y = Math.min(h, ((reference.energyDb - curve[i]) / RANGE_DB) * h);
      i ? dg.lineTo(x, y) : dg.moveTo(x, y);
    }
    dg.stroke();
    // T30 from the -5 to -35 dB stretch of the cell's own decay, doubled.
    const at = (db) => {
      for (let i = 0; i < curve.length; i++) if (curve[i] - curve[0] <= db) return (i / curve.length) * seconds;
      return null;
    };
    const t5 = at(-5);
    const t35 = at(-35);
    const t30 = t5 !== null && t35 !== null ? 2 * (t35 - t5) : null;
    captions.decay.textContent =
      `0 – ${seconds.toFixed(2)} s · ${RANGE_DB} dB · energy ${curve[0].toFixed(1)} dB` +
      (t30 !== null ? ` · T30 ${t30.toFixed(2)} s` : "");
  }

  /** The direction diagram, front up, turned for the head. */
  function drawDirection() {
    const w = direction.width;
    const h = direction.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) / 2 - 6;
    pg.fillStyle = GROUND;
    pg.fillRect(0, 0, w, h);
    pg.strokeStyle = LINE;
    pg.lineWidth = 1;
    for (const db of [10, 20, 30]) {
      pg.beginPath();
      pg.arc(cx, cy, radius * (1 - db / POLAR_DB), 0, 2 * Math.PI);
      pg.stroke();
    }
    pg.beginPath();
    pg.moveTo(cx, cy - radius);
    pg.lineTo(cx, cy + radius);
    pg.moveTo(cx - radius, cy);
    pg.lineTo(cx + radius, cy);
    pg.stroke();
    for (const db of [10, 20]) {
      label(pg, `-${db} dB`, cx + 2, cy - radius * (1 - db / POLAR_DB), "left", "middle");
    }
    label(pg, "front", cx, 1, "center", "top");
    label(pg, "back", cx, h - 1, "center", "bottom");
    label(pg, "left", 1, cy, "left", "middle");
    label(pg, "right", w - 1, cy, "right", "middle");
    if (!shown) return;
    const psi = headYawOfCamera(cameraYaw);
    const curve = (levels, colour, width) => {
      const top = Math.max(...levels);
      pg.strokeStyle = colour;
      pg.lineWidth = width;
      pg.beginPath();
      for (let k = 0; k <= levels.length; k++) {
        const level = levels[k % levels.length];
        const r = radius * Math.max(0, 1 - (top - level) / POLAR_DB);
        const relative = (2 * Math.PI * k) / levels.length - psi;
        const x = cx - Math.sin(relative) * r;
        const y = cy - Math.cos(relative) * r;
        k ? pg.lineTo(x, y) : pg.moveTo(x, y);
      }
      pg.stroke();
    };
    curve(shown.polar.whole, WHOLE, 1.5);
    curve(shown.polar.early, EARLY, 2);
    const early = shown.polar.early;
    let peak = 0;
    for (let k = 1; k < early.length; k++) if (early[k] > early[peak]) peak = k;
    let degrees = ((((2 * Math.PI * peak) / early.length - psi) * 180) / Math.PI) % 360;
    if (degrees > 180) degrees -= 360;
    if (degrees < -180) degrees += 360;
    const side = degrees > 0 ? "left" : "right";
    captions.direction.textContent =
      `early · whole · ${POLAR_DB} dB · peak ${Math.abs(degrees).toFixed(0)}° ${side}`;
  }

  return {
    clear,
    /** The cell nearest the source sets the top of the level scales. */
    async setReference(field) {
      const [x, y, z] = field.index.source_position;
      const position = field.cellAt(x, y, z);
      if (position === null) return;
      const result = await analyse(await field.cell(position), field.index.order, 1);
      let ceilingDb = -Infinity;
      for (const level of result.spectrogram.levels) ceilingDb = Math.max(ceilingDb, level);
      references.set(field, { ceilingDb, energyDb: result.energyDb });
      if (shown && shown.field === field) {
        reference = references.get(field);
        drawSpectrogram(shown.spectrogram, shown.sampleRate);
        drawDecay(shown.decay);
      }
    },
    /** Plot one cell of a field; a later call wins over one still analysing. */
    async show(field, position, earlySamples) {
      const mine = ++shows;
      const result = await analyse(await field.cell(position), field.index.order, earlySamples);
      if (mine !== shows) return;
      shown = { ...result, field, sampleRate: field.index.sample_rate_hz };
      reference = references.get(field) || reference;
      seconds = field.index.samples / field.index.sample_rate_hz;
      drawSpectrogram(shown.spectrogram, shown.sampleRate);
      drawDecay(shown.decay);
      drawDirection();
    },
    setHead(yaw) {
      cameraYaw = yaw;
      drawDirection();
    },
  };
}
