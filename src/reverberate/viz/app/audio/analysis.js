/** The two plots' numbers, from a response: Schroeder decay and a short-time
 * magnitude spectrogram. Pure functions, shared by the page and the workers.
 */
import { fft } from "./fft.js";

/** Schroeder decay in dB, `points` values, of one ear of a response. */
export function decay(signal, points = 256) {
  let total = 0;
  for (let i = 0; i < signal.length; i++) total += signal[i] * signal[i];
  const out = new Float32Array(points);
  if (total <= 0) return out.fill(-120);
  let tail = total;
  let next = 0;
  for (let i = 0; i < signal.length; i++) {
    const at = Math.floor((i * points) / signal.length);
    while (next <= at && next < points) {
      out[next] = 10 * Math.log10(Math.max(tail / total, 1e-12));
      next++;
    }
    tail -= signal[i] * signal[i];
  }
  while (next < points) out[next++] = -120;
  return out;
}

/** A short-time magnitude spectrogram, `frames` x `bins` bytes, `rangeDb` deep. */
export function spectrogram(signal, frames = 256, bins = 128, rangeDb = 70) {
  const size = bins * 2;
  const hop = Math.max(1, Math.floor((signal.length - size) / frames));
  const levels = new Float32Array(frames * bins);
  const re = new Float32Array(size);
  const im = new Float32Array(size);
  let top = -Infinity;
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    for (let i = 0; i < size; i++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);
      re[i] = (signal[start + i] || 0) * w;
      im[i] = 0;
    }
    fft(re, im);
    for (let b = 0; b < bins; b++) {
      const db = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-30);
      levels[f * bins + b] = db;
      if (db > top) top = db;
    }
  }
  const out = new Uint8Array(frames * bins);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.max(0, Math.min(255, Math.round(((levels[i] - top + rangeDb) / rangeDb) * 255)));
  }
  return { frames, bins, levels: out };
}

/** The whole response's left ear: the early part faded into the late one. */
export function joined(early, late, start) {
  const whole = new Float32Array(Math.max(early.length, start + late.length));
  whole.set(early);
  for (let i = 0; i < late.length; i++) whole[start + i] += late[i];
  return whole;
}
