/** What a saccade measures as: how far the filter a listener hears moves
 * from one frame to the next.
 *
 * `trace` is one entry per frame, each the gain-weighted sum of every
 * response sounding at that instant -- both ears -- with the propagation
 * delay in front of them in samples and, where the walk traced it, the
 * energy of half a second of the late part. A listener who walks and turns
 * smoothly moves every one of these smoothly; the second difference, the
 * jerk, is zero for a straight line and rings at every step of a staircase,
 * so it is the jerk that is reported:
 *
 * - **level**, the energy of the filter: a response swapped for one of a
 *   different gain steps it.
 * - **timbre**, how far its spectrum moves per frame, in dB per band: two
 *   responses a millisecond apart comb filter each other while they fade.
 * - **reverberation**, the late part's energy: it dips whenever the late
 *   part's convolver is given a new buffer and starts from nothing.
 * - **interaural delay and level** of the direct sound: steps of the
 *   direction.
 * - **arrival**, where the response starts plus the delay line in front of
 *   it: the time the sound takes, which should slide as the walk does.
 *
 * Clicks are not here. Through a noise signal a detector misses them as an
 * ear does; they were measured in the browser, through a tone (ADR 0013).
 */
import { fft, nextPowerOfTwo } from "../../src/reverberate/viz/app/audio/fft.js";

const percentile = (values, fraction) => {
  if (!values.length) return 0;
  const sorted = Float64Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.round(fraction * (sorted.length - 1)))];
};
const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);

//: How sure of an interaural delay the correlation has to be before the
//: number is used. Below it the two ears do not agree well enough for a
//: delay to mean anything -- the direct sound is buried under a reflection --
//: and the reading is the estimator's noise, not the engine's.
const SURE_ENOUGH = 0.25;

/** Where a response starts: the first sample past a hundredth of its energy.
 *  The loudest sample would not do; two reflections of nearly equal height
 *  trade places as the head turns. */
function onset(left, right) {
  let total = 0;
  for (let i = 0; i < left.length; i++) total += left[i] ** 2 + right[i] ** 2;
  let running = 0;
  for (let i = 0; i < left.length; i++) {
    const here = left[i] ** 2 + right[i] ** 2;
    if (running + here >= total * 0.01) return i - 1 + (here > 0 ? (total * 0.01 - running) / here : 0);
    running += here;
  }
  return 0;
}

/** A one-pole low pass at `hz`. Its phase is the same in both ears, so it
 *  moves no interaural delay. */
function lowpass(signal, hz, sampleRate) {
  const a = Math.exp((-2 * Math.PI * hz) / sampleRate);
  const out = new Float64Array(signal.length);
  let state = 0;
  for (let i = 0; i < signal.length; i++) out[i] = state = (1 - a) * signal[i] + a * state;
  return out;
}

/** The interaural delay of the direct sound, below 1.6 kHz where an ear
 *  takes its delay from, over the 2.5 ms after the onset; with how sure the
 *  correlation is of it. Positive means the right ear is late. */
function interaural(left, right, at, sampleRate) {
  const from = Math.max(0, Math.floor(at) - 16);
  const count = Math.min(left.length - from, Math.round(0.0025 * sampleRate));
  if (count <= 4) return { lag: 0, sure: 0 };
  const [l, r] = [left, right].map((ear) => lowpass(Float64Array.from(ear.subarray(from, from + count)), 1600, sampleRate));
  const maxLag = Math.round(0.0011 * sampleRate);
  const correlation = (lag) => {
    let sum = 0;
    for (let i = Math.max(0, -lag); i < Math.min(count, count - lag); i++) sum += l[i] * r[i + lag];
    return sum;
  };
  let best = 0;
  let bestValue = -Infinity;
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    const value = correlation(lag);
    if (value > bestValue) [best, bestValue] = [lag, value];
  }
  const before = correlation(best - 1);
  const after = correlation(best + 1);
  const curvature = before - 2 * bestValue + after;
  let el = 0;
  let er = 0;
  for (let i = 0; i < count; i++) [el, er] = [el + l[i] ** 2, er + r[i] ** 2];
  const energy = Math.sqrt(el * er);
  return {
    lag: best + (Math.abs(curvature) > 1e-30 ? (0.5 * (before - after)) / curvature : 0),
    // A peak against the search edge is no peak.
    sure: energy > 0 && Math.abs(best) < maxLag ? bestValue / energy : 0,
  };
}

/** Every number this file exists for, from one walk's trace. */
export function measureTrace(trace, { sampleRate, bands = 512 }) {
  const size = nextPowerOfTwo(trace[0].ears[0].length);
  const energy = [];
  const ild = [];
  const itd = [];
  const sure = [];
  const arrival = [];
  const step = [];
  let previous = null;
  for (const { ears, delaySamples } of trace) {
    const [left, right] = ears;
    let el = 0;
    let er = 0;
    for (let i = 0; i < left.length; i++) [el, er] = [el + left[i] ** 2, er + right[i] ** 2];
    energy.push(10 * Math.log10(Math.max(el + er, 1e-30)));
    ild.push(10 * Math.log10(Math.max(er, 1e-30) / Math.max(el, 1e-30)));
    const at = onset(left, right);
    arrival.push(at + delaySamples);
    const heard = interaural(left, right, at, sampleRate);
    itd.push(heard.lag);
    sure.push(heard.sure >= SURE_ENOUGH);
    // The spectrum, in bands, both ears together.
    const now = new Float64Array(bands);
    const fold = Math.max(1, Math.floor(size / 2 / bands));
    for (const ear of ears) {
      const re = new Float32Array(size);
      const im = new Float32Array(size);
      re.set(ear);
      fft(re, im);
      for (let b = 0; b < bands; b++) for (let i = b * fold; i < (b + 1) * fold; i++) now[b] += re[i] ** 2 + im[i] ** 2;
    }
    for (let b = 0; b < bands; b++) now[b] = 10 * Math.log10(now[b] + 1e-30);
    if (previous) step.push(Math.sqrt(now.reduce((sum, v, b) => sum + (v - previous[b]) ** 2, 0) / bands));
    previous = now;
  }
  const jerk = (series, valid = null) => {
    const out = [];
    for (let i = 2; i < series.length; i++) {
      if (valid && !(valid[i] && valid[i - 1] && valid[i - 2])) continue;
      out.push(Math.abs(series[i] - 2 * series[i - 1] + series[i - 2]));
    }
    return out;
  };
  const round = (value, digits = 2) => Number(value.toFixed(digits));
  const reverb = trace[0].reverbDb === undefined ? null : trace.map((frame) => frame.reverbDb);
  return {
    levelJerkDb: round(percentile(jerk(energy), 0.99)),
    timbreStepDb: round(percentile(step, 0.99)),
    reverbJerkDb: reverb ? round(percentile(jerk(reverb), 0.99)) : null,
    // The mean for the delay: a staircase has no jerk on its treads, so a
    // median would read zero for the very shape this looks for, and a
    // handful of frames defeat any estimator, which a ninety-ninth would read.
    itdJerkSamples: round(mean(jerk(itd, sure)), 3),
    ildJerkDb: round(percentile(jerk(ild), 0.99)),
    arrivalJerkSamples: round(percentile(jerk(arrival), 0.99), 1),
  };
}
