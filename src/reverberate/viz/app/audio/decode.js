/** The decode itself: an ambisonic response, a head orientation and a
 * decoder in; two ear responses out. No worker, no messages, no clock.
 *
 * `brir.worker.js` is the message shell around this module; the tests drive
 * the same functions with no browser.
 *
 * All of it is linear: the rotation is applied to spectra, which it mixes
 * exactly as it mixes the samples, and the sum of an early part and a late
 * part with their complementary fades is the whole response to rounding.
 *
 * **The two parts are faded into each other on the field, before the decode,
 * not on the two ears after it.** The decoder spreads every sample over its
 * 512 taps, centred on the 256th, so a fade put on the decoded early part at
 * the seam removes field samples from well before the seam, which the late
 * part, starting at the seam, never had: the two parts summed to the whole
 * response only far from the seam, and missed it by half its size there.
 * Faded on the field, the parts sum to the field, and by linearity their
 * decodes to its decode.
 *
 * Every spectrum here is that of a real signal, so bins above the Nyquist
 * one are the conjugates of those below and are never computed: the arrays
 * hold `size / 2 + 1` bins, and `ear` mirrors them back before the inverse
 * transform.
 */
import { fft, nextPowerOfTwo } from "./fft.js";
import { channelCount, rotationBlocks } from "./sh.js";

//: The two parts meet in a raised-cosine crossfade, sin^2 in and cos^2 out,
//: which sums to one and has no corner at either end.
const fadeIn = (i, fade) => Math.sin((Math.PI / 2) * (i / fade)) ** 2;

//: How far before the decoder's first audible tap the early part starts.
//: The decoder is the same for every cell, so this needs no room for spread.
const DECODER_GUARD = 32;

//: How far before a cell's own onset its response is cut, samples. The
//: latency is measured on one cell and the others vary around it; this is
//: the margin that keeps every cell's direct sound inside its response.
export const LATENCY_GUARD = 96;

/** Forward transforms of `channels`, zero padded to `size`, half spectra. */
function spectraOf(channels, size) {
  const half = size / 2 + 1;
  const re = [];
  const im = [];
  const workRe = new Float32Array(size);
  const workIm = new Float32Array(size);
  for (const channel of channels) {
    workRe.set(channel);
    workRe.fill(0, channel.length);
    workIm.fill(0);
    fft(workRe, workIm);
    re.push(Float32Array.from(workRe.subarray(0, half)));
    im.push(Float32Array.from(workIm.subarray(0, half)));
  }
  return { re, im };
}

/** Copies of `channels` faded in over their first `rise` samples and out
 *  over the `fall` samples from `fallFrom`, as the seam wants them. */
function windowed(channels, rise, fallFrom, fall) {
  if (!rise && !fall) return channels;
  return channels.map((channel) => {
    const out = Float32Array.from(channel);
    for (let i = 0; i < rise && i < out.length; i++) out[i] *= fadeIn(i, rise);
    for (let i = fallFrom; fall && i < out.length; i++) {
      out[i] *= i >= fallFrom + fall ? 0 : 1 - fadeIn(i - fallFrom, fall);
    }
    return out;
  });
}

/** Samples of quiet run-up at the start of the binaural decoder's filters.
 *
 * The decoder is designed centred, its energy peaking half way along its
 * taps, so everything it decodes comes out five milliseconds late. That is
 * latency the listener hears and nothing needs: the early part is started at
 * the first tap within 40 dB of the filters' peak, less `DECODER_GUARD`, and
 * the late part is placed as much earlier to meet it.
 */
export function decoderLead(filters, channels, taps) {
  const energy = new Float64Array(taps);
  for (let row = 0; row < 2 * channels; row++) {
    for (let tap = 0; tap < taps; tap++) energy[tap] += filters[row * taps + tap] ** 2;
  }
  let peak = 0;
  for (const value of energy) peak = Math.max(peak, value);
  let first = 0;
  while (first < taps && energy[first] <= peak * 1e-4) first++;
  return Math.max(0, first - DECODER_GUARD);
}

/** The samples between the geometric arrival and where a response starts.
 *
 * A solver's response does not start when the direct sound geometrically
 * arrives: its source pulse has a delay of its own, a few hundred samples,
 * the same at every cell, and like the decoder's it is latency nothing needs.
 * Measured on the omnidirectional channel of the cell nearest the source,
 * where the direct sound is clearest, as the first sample within 40 dB of its
 * peak, less `LATENCY_GUARD`.
 */
export function solverLatency(omni, directSamples) {
  let peak = 0;
  for (let i = 0; i < omni.length; i++) peak = Math.max(peak, Math.abs(omni[i]));
  let onset = 0;
  while (onset < omni.length && Math.abs(omni[onset]) <= peak * 0.01) onset++;
  return Math.max(0, Math.round(onset - directSamples) - LATENCY_GUARD);
}

export function createDecoder() {
  let decoder = null;
  const kept = new Map(); // key -> { size, samples, re[], im[] }

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
  function rotate(entry, head) {
    if (!head) return entry;
    const half = entry.size / 2 + 1;
    const re = entry.re.map(() => new Float32Array(half));
    const im = entry.im.map(() => new Float32Array(half));
    for (const { offset, size: n, matrix } of rotationBlocks(decoder.order, head)) {
      for (let i = 0; i < n; i++) {
        const outRe = re[offset + i];
        const outIm = im[offset + i];
        for (let j = 0; j < n; j++) {
          const weight = matrix[i * n + j];
          if (Math.abs(weight) < 1e-12) continue;
          const inRe = entry.re[offset + j];
          const inIm = entry.im[offset + j];
          for (let b = 0; b < half; b++) {
            outRe[b] += weight * inRe[b];
            outIm[b] += weight * inIm[b];
          }
        }
      }
    }
    return { re, im };
  }

  /** One ear from rotated half spectra, as a time signal of `size`. */
  function ear(rotated, index, size) {
    const spectra = decoderSpectra(size)[index];
    const half = size / 2 + 1;
    const accRe = new Float32Array(size);
    const accIm = new Float32Array(size);
    for (let ch = 0; ch < decoder.channels; ch++) {
      const fr = spectra.re[ch];
      const fi = spectra.im[ch];
      const xr = rotated.re[ch];
      const xi = rotated.im[ch];
      for (let i = 0; i < half; i++) {
        accRe[i] += xr[i] * fr[i] - xi[i] * fi[i];
        accIm[i] += xr[i] * fi[i] + xi[i] * fr[i];
      }
    }
    // The product of two real signals' spectra is Hermitian too; the upper
    // half is the conjugate mirror of the lower, not another computation.
    for (let i = 1; i < half - 1; i++) {
      accRe[size - i] = accRe[i];
      accIm[size - i] = -accIm[i];
    }
    fft(accRe, accIm, true);
    return accRe;
  }

  return {
    /** Samples the early part is advanced by; see `decoderLead`. */
    get lead() {
      return decoder ? decoder.lead : 0;
    },
    /** The filters, once; every kept spectrum stays valid, the decoder's do not. */
    setDecoder({ order, channels, taps, filters }) {
      if (channels !== channelCount(order)) {
        throw new Error(`decoder has ${channels} channels for order ${order}`);
      }
      decoder = { order, channels, taps, filters, spectra: new Map(), lead: decoderLead(filters, channels, taps) };
    },
    /** Transform one cell's channels and keep them under `key`.
     *
     * `fadeIn` fades the first samples in, for a late part; `fadeOutFrom`
     * and `fadeOut` fade an early part out, ending at zero. See the seam, at
     * the top of this file.
     */
    keep({ key, channels, fadeIn: rise = 0, fadeOutFrom = 0, fadeOut: fall = 0, limit = 0 }) {
      const samples = channels[0].length;
      const size = nextPowerOfTwo(samples + decoder.taps - 1);
      kept.set(key, { size, samples, ...spectraOf(windowed(channels, rise, fadeOutFrom, fall), size) });
      while (limit && kept.size > limit) kept.delete(kept.keys().next().value);
    },
    forget: (key) => kept.delete(key),
    /** The early part of a kept cell at a head, in one go, advanced by the
     *  decoder's lead; the late part is placed that much earlier to meet it
     *  (`spatial.js`). */
    renderEarly({ key, head }) {
      const entry = kept.get(key);
      if (!entry) return null;
      const rotated = rotate(entry, head);
      const end = entry.samples + decoder.taps - 1;
      return [0, 1].map((index) => Float32Array.from(ear(rotated, index, entry.size).subarray(decoder.lead, end)));
    },
    /** The late part as three steps, so a caller can yield between them. */
    lateSteps({ key, head }) {
      const entry = kept.get(key);
      if (!entry) return null;
      const end = entry.samples + decoder.taps - 1;
      let rotated = null;
      const ears = [];
      return {
        ears,
        steps: [
          () => {
            rotated = rotate(entry, head);
          },
          () => ears.push(Float32Array.from(ear(rotated, 0, entry.size).subarray(0, end))),
          () => ears.push(Float32Array.from(ear(rotated, 1, entry.size).subarray(0, end))),
        ],
      };
    },
  };
}
