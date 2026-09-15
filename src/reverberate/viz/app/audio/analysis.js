/** The plots' numbers, from the ambisonic response of one cell: the
 * omnidirectional channel's Schroeder decay and short-time spectrum, in
 * absolute decibels so a cell farther from the source plots lower, and the
 * direction the energy arrives from. Pure functions, run in a worker.
 */
import { fft } from "./fft.js";
import { channelCount, realSH } from "./sh.js";

/** Schroeder decay of a signal in dB re unit energy, `points` values. */
export function decay(signal, points = 256) {
  let total = 0;
  for (let i = 0; i < signal.length; i++) total += signal[i] * signal[i];
  const out = new Float32Array(points).fill(-200);
  let tail = total;
  let next = 0;
  for (let i = 0; i < signal.length; i++) {
    const at = Math.floor((i * points) / signal.length);
    while (next <= at && next < points) {
      out[next] = 10 * Math.log10(Math.max(tail, 1e-20));
      next++;
    }
    tail -= signal[i] * signal[i];
  }
  return out;
}

/** A short-time spectrum, `frames` x `bins` values in dB re a unit impulse. */
export function spectrogram(signal, frames = 256, bins = 128) {
  const size = bins * 2;
  const hop = Math.max(1, Math.floor((signal.length - size) / frames));
  const levels = new Float32Array(frames * bins);
  const re = new Float32Array(size);
  const im = new Float32Array(size);
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    for (let i = 0; i < size; i++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);
      re[i] = (signal[start + i] || 0) * w;
      im[i] = 0;
    }
    fft(re, im);
    for (let b = 0; b < bins; b++) {
      levels[f * bins + b] = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-20);
    }
  }
  return { frames, bins, levels };
}

/** The channel covariance over `[from, to)` samples, `channels` squared. */
export function covariance(channels, from, to) {
  const n = channels.length;
  const out = new Float64Array(n * n);
  for (let a = 0; a < n; a++) {
    const x = channels[a];
    for (let b = 0; b <= a; b++) {
      const y = channels[b];
      let sum = 0;
      for (let t = from; t < to; t++) sum += x[t] * y[t];
      out[a * n + b] = sum;
      out[b * n + a] = sum;
    }
  }
  return out;
}

/** Max-rE beam weights by degree: the narrowest beam without side lobes
 *  this order allows, Legendre polynomials at the order's own angle. */
export function maxReWeights(order) {
  const cosine = Math.cos((137.9 * Math.PI) / 180 / (order + 1.51));
  const p = [1, cosine];
  for (let l = 2; l <= order; l++) p.push(((2 * l - 1) * cosine * p[l - 1] - (l - 1) * p[l - 2]) / l);
  const weights = new Float64Array(channelCount(order));
  for (let l = 0; l <= order; l++) for (let m = -l; m <= l; m++) weights[l * l + l + m] = p[l];
  return weights;
}

/** Energy in dB arriving from `count` azimuths of the horizontal plane,
 *  counter clockwise from the frame's front, out of a channel covariance. */
export function polar(cov, order, count = 72) {
  const n = channelCount(order);
  const weights = maxReWeights(order);
  const out = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    const azimuth = (2 * Math.PI * k) / count;
    const beam = realSH(order, Math.cos(azimuth), Math.sin(azimuth), 0);
    for (let i = 0; i < n; i++) beam[i] *= weights[i];
    let energy = 0;
    for (let a = 0; a < n; a++) {
      let row = 0;
      for (let b = 0; b < n; b++) row += cov[a * n + b] * beam[b];
      energy += beam[a] * row;
    }
    out[k] = 10 * Math.log10(Math.max(energy, 1e-20));
  }
  return out;
}

/** Everything the plots show of one cell. `earlySamples` bounds the early
 *  polar diagram, the direct sound and first reflections. */
export function analyse(channels, order, earlySamples) {
  const omni = channels[0];
  let peak = 0;
  for (let i = 0; i < omni.length; i++) peak = Math.max(peak, Math.abs(omni[i]));
  const early = covariance(channels, 0, Math.min(earlySamples, omni.length));
  const whole = covariance(channels, 0, omni.length);
  const edc = decay(omni);
  return {
    spectrogram: spectrogram(omni),
    decay: edc,
    polar: { early: polar(early, order), whole: polar(whole, order) },
    peakDb: 20 * Math.log10(Math.max(peak, 1e-10)),
    energyDb: edc[0],
  };
}
