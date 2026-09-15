/** The decode, off the main thread: an ambisonic response, a head
 * orientation and a decoder in; two ear responses out.
 *
 * Two instances of this script run, in two roles the first message sets:
 *
 * - **early**: a cell's first `EARLY_MS` are kept as spectra and re-decoded
 *   for every head orientation. At order 7 that is 680 rotation weights over
 *   one 8192-bin spectrum and 128 complex multiply-accumulates: 5 to 25 ms.
 * - **late**: the rest of the response. Decoded at the head orientation of
 *   the moment when a cell is entered, then again,
 *   exactly, once the head has been still for `STILL_MS`. A late decode is
 *   ten times the early one, so it runs in steps that yield between them and
 *   can be cancelled by a later message when the head moves again.
 *
 * All of it is linear: the rotation is applied to spectra, which it mixes
 * exactly as it mixes the samples, and the sum of the two parts with their
 * complementary fades is the whole response to rounding.
 */
import { fft, nextPowerOfTwo } from "./fft.js";
import { channelCount, rotationBlocks } from "./sh.js";

let decoder = null;
const kept = new Map(); // key -> { size, samples, re[], im[], fade }
//: Late spectra of one cell are tens of megabytes; keep few.
const KEEP_LATE = 3;
const cancelled = new Set();
let current = null; // the late render in progress, if any

function spectraOf(channels, size) {
  const re = channels.map((channel) => {
    const out = new Float32Array(size);
    out.set(channel);
    return out;
  });
  const im = channels.map(() => new Float32Array(size));
  for (let ch = 0; ch < channels.length; ch++) fft(re[ch], im[ch]);
  return { re, im };
}

function decoderSpectra(size) {
  if (!decoder.spectra.has(size)) {
    const ears = [];
    for (let ear = 0; ear < 2; ear++) {
      const taps = [];
      for (let ch = 0; ch < decoder.channels; ch++) {
        const start = (ear * decoder.channels + ch) * decoder.taps;
        taps.push(decoder.filters.subarray(start, start + decoder.taps));
      }
      ears.push(spectraOf(taps, size));
    }
    decoder.spectra.set(size, ears);
  }
  return decoder.spectra.get(size);
}

/** Rotate kept spectra by the head matrix, block by block; null head = as is. */
function rotate(entry, head, size) {
  if (!head) return { re: entry.re, im: entry.im };
  const blocks = rotationBlocks(decoder.order, head);
  const re = entry.re.map(() => new Float32Array(size));
  const im = entry.im.map(() => new Float32Array(size));
  for (const { offset, size: n, matrix } of blocks) {
    for (let i = 0; i < n; i++) {
      const outRe = re[offset + i];
      const outIm = im[offset + i];
      for (let j = 0; j < n; j++) {
        const weight = matrix[i * n + j];
        if (Math.abs(weight) < 1e-12) continue;
        const inRe = entry.re[offset + j];
        const inIm = entry.im[offset + j];
        for (let b = 0; b < size; b++) {
          outRe[b] += weight * inRe[b];
          outIm[b] += weight * inIm[b];
        }
      }
    }
  }
  return { re, im };
}

/** One ear from rotated spectra, as a time signal of `size`. */
function ear(rotated, index, size) {
  const spectra = decoderSpectra(size)[index];
  const accRe = new Float32Array(size);
  const accIm = new Float32Array(size);
  for (let ch = 0; ch < decoder.channels; ch++) {
    const fr = spectra.re[ch];
    const fi = spectra.im[ch];
    const xr = rotated.re[ch];
    const xi = rotated.im[ch];
    for (let i = 0; i < size; i++) {
      accRe[i] += xr[i] * fr[i] - xi[i] * fi[i];
      accIm[i] += xr[i] * fi[i] + xi[i] * fr[i];
    }
  }
  fft(accRe, accIm, true);
  return accRe;
}

//: The two parts meet in a raised-cosine crossfade, sin^2 in and cos^2 out,
//: which sums to one and has no corner at either end.
const fadeIn = (i, fade) => Math.sin((Math.PI / 2) * (i / fade)) ** 2;

function trim(signal, entry) {
  const out = Float32Array.from(signal.subarray(0, entry.samples + decoder.taps - 1));
  for (let i = 0; i < entry.fade && i < out.length; i++) out[i] *= fadeIn(i, entry.fade);
  return out;
}

function keep(message) {
  const samples = message.channels[0].length;
  const size = nextPowerOfTwo(samples + decoder.taps - 1);
  kept.set(message.key, {
    size,
    samples,
    fade: message.crossfade || 0,
    ...spectraOf(message.channels, size),
  });
  if (message.limit) {
    while (kept.size > message.limit) kept.delete(kept.keys().next().value);
  }
}

/** The early decode, in one go. */
function renderEarly(message) {
  const started = performance.now();
  const entry = kept.get(message.cell);
  if (!entry) {
    self.postMessage({ type: "brir", id: message.id, missing: message.cell });
    return;
  }
  const rotated = rotate(entry, message.head, entry.size);
  const early = [0, 1].map((index) => {
    const out = Float32Array.from(ear(rotated, index, entry.size).subarray(0, entry.samples + decoder.taps - 1));
    // Faded out where the late part fades in.
    const start = message.fadeStart;
    const fade = message.crossfade;
    if (fade) {
      for (let i = start; i < out.length; i++) {
        out[i] *= i >= start + fade ? 0 : 1 - fadeIn(i - start, fade);
      }
    }
    return out;
  });
  self.postMessage(
    { type: "brir", id: message.id, early, ms: performance.now() - started },
    [early[0].buffer, early[1].buffer]
  );
}

/** The late decode, in steps that yield so a cancel can land between them. */
function renderLate(message) {
  const started = performance.now();
  const entry = kept.get(message.key);
  if (!entry) {
    self.postMessage({ type: "late", id: message.id, missing: message.key });
    return;
  }
  const steps = [];
  let rotated = null;
  const ears = [];
  steps.push(() => {
    rotated = rotate(entry, message.head, entry.size);
  });
  steps.push(() => ears.push(trim(ear(rotated, 0, entry.size), entry)));
  steps.push(() => ears.push(trim(ear(rotated, 1, entry.size), entry)));
  let at = 0;
  const step = () => {
    if (cancelled.has(message.id)) {
      cancelled.delete(message.id);
      current = null;
      self.postMessage({ type: "late", id: message.id, cancelled: true });
      return;
    }
    steps[at++]();
    if (at < steps.length) {
      setTimeout(step, 0);
      return;
    }
    current = null;
    self.postMessage(
      { type: "late", id: message.id, key: message.key, brir: ears, ms: performance.now() - started },
      [ears[0].buffer, ears[1].buffer]
    );
  };
  current = message.id;
  setTimeout(step, 0);
}

self.onmessage = (event) => {
  const message = event.data;
  switch (message.type) {
    case "decoder":
      decoder = {
        order: message.order,
        channels: message.channels,
        taps: message.taps,
        filters: message.filters,
        spectra: new Map(),
      };
      if (decoder.channels !== channelCount(decoder.order)) {
        throw new Error(`decoder has ${decoder.channels} channels for order ${decoder.order}`);
      }
      return;
    case "cell":
    case "keep":
      keep(message);
      return;
    case "forget":
      kept.delete(message.key);
      return;
    case "cancel":
      if (current === message.id) cancelled.add(message.id);
      return;
    case "render":
      if (message.cell !== undefined) renderEarly(message);
      else renderLate(message);
      return;
    default:
      return;
  }
};
