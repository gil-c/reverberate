/** The early part's convolution: one history of the input, many filters.
 *
 * A `ConvolverNode` given a new response has no past. It convolves only the
 * input that arrives after, as if the voice had been switched on at that
 * instant, and every tap of the response adds a small step as the input
 * reaches it, heard at whatever gain the fade in has reached. A real early
 * response carries energy across all of its 150 ms, so the steps spread over
 * the whole fade and no fade removes them (ADR 0013).
 *
 * Here the input is kept once, as the spectra of its last few hundred
 * milliseconds, and every filter is applied to that same history. A new
 * response therefore sounds from its first sample exactly as it would had it
 * always been there, and a crossfade between two responses is a crossfade
 * between two outputs that are both complete: nothing is switched on.
 *
 * Uniformly partitioned overlap-save: blocks of `BLOCK` samples, the render
 * quantum of Web Audio, so a block of output is computed from the block of
 * input that arrives with it and no latency is added. A filter is cut into
 * blocks, each transformed at twice the length (`partitionFilter`, run in the
 * decode worker, off the audio thread); each block of input is transformed
 * once and kept in a ring of spectra, the frequency-domain delay line; an
 * output block is the sum over partitions of the delay line times the filter,
 * inverse transformed. The two ears share one inverse transform, packed as
 * the real and imaginary parts of one complex signal.
 *
 * Which filter sounds at what gain is `ring.js`, exactly as it was for the
 * convolvers: exact-zero fades summing to one. A slot is still given a new
 * filter only once it is silent, because swapping the filter under a sounding
 * slot is an instant change of filter, and that clicks too.
 *
 * Pure: no worklet, no messages, no clock of its own. `early.worklet.js` is
 * the shell, and `tests/test_partitioned_js.py` checks this against direct
 * convolution.
 */
import { fft } from "./fft.js";
import { createRing } from "./ring.js";

//: Samples per block: Web Audio's render quantum.
export const BLOCK = 128;
const SIZE = 2 * BLOCK;
const BINS = BLOCK + 1;

/** A filter's two ears, cut into blocks and transformed, ready to convolve.
 *
 * `re` and `im` hold `[ear][partition][bin]`, the lower half of each
 * transform only: the filter is real, and the upper half is the conjugate.
 */
export function partitionFilter(ears) {
  const partitions = Math.ceil(ears[0].length / BLOCK);
  const re = new Float32Array(2 * partitions * BINS);
  const im = new Float32Array(2 * partitions * BINS);
  const workRe = new Float32Array(SIZE);
  const workIm = new Float32Array(SIZE);
  for (let ear = 0; ear < 2; ear++) {
    for (let p = 0; p < partitions; p++) {
      workRe.fill(0);
      workIm.fill(0);
      workRe.set(ears[ear].subarray(p * BLOCK, Math.min(ears[ear].length, (p + 1) * BLOCK)));
      fft(workRe, workIm);
      const base = (ear * partitions + p) * BINS;
      re.set(workRe.subarray(0, BINS), base);
      im.set(workIm.subarray(0, BINS), base);
    }
  }
  return { partitions, re, im };
}

export function createPartitionedConvolver({ slots = 6, fade = 0.12, sampleRate = 48000 } = {}) {
  const ring = createRing(slots, fade);
  const filters = new Array(slots).fill(null);
  // The delay line: spectra of the input's recent blocks, newest at `head`.
  let lineRe = [new Float32Array(BINS)];
  let lineIm = [new Float32Array(BINS)];
  let head = 0;
  const previous = new Float32Array(BLOCK);
  const workRe = new Float32Array(SIZE);
  const workIm = new Float32Array(SIZE);
  const leftRe = new Float32Array(BINS);
  const leftIm = new Float32Array(BINS);
  const rightRe = new Float32Array(BINS);
  const rightIm = new Float32Array(BINS);

  /** Room for `count` blocks of history, oldest first to go. Growing it
   *  keeps what is there; the blocks it adds are silence. */
  function capacity(count) {
    const old = lineRe.length;
    if (count <= old) return;
    const re = [];
    const im = [];
    for (let age = count - 1; age >= 0; age--) {
      if (age < old) {
        re.push(lineRe[(head - age + old) % old]);
        im.push(lineIm[(head - age + old) % old]);
      } else {
        re.push(new Float32Array(BINS));
        im.push(new Float32Array(BINS));
      }
    }
    lineRe = re;
    lineIm = im;
    head = count - 1;
  }

  return {
    /** A new filter, from `partitionFilter`, taking over from `time`. */
    setFilter(filter, time) {
      capacity(filter.partitions);
      const { slot } = ring.take(time);
      filters[slot] = filter;
    },
    /** Everything fades to silence from `time`. */
    silence(time) {
      ring.silence(time);
    },
    /** One block: `input` in, `left` and `right` out, the block starting at
     *  `time` seconds on the clock `setFilter` was given times on. */
    process(input, left, right, time) {
      // The input block, transformed with the one before it, into the line.
      workRe.set(previous, 0);
      workRe.set(input, BLOCK);
      workIm.fill(0);
      fft(workRe, workIm);
      previous.set(input);
      const size = lineRe.length;
      head = (head + 1) % size;
      lineRe[head].set(workRe.subarray(0, BINS));
      lineIm[head].set(workIm.subarray(0, BINS));
      left.fill(0);
      right.fill(0);
      const step = 1 / sampleRate;
      for (let slot = 0; slot < slots; slot++) {
        const filter = filters[slot];
        if (!filter) continue;
        if (ring.gainAt(slot, time) === 0 && ring.gainAt(slot, time + BLOCK * step) === 0) continue;
        leftRe.fill(0);
        leftIm.fill(0);
        rightRe.fill(0);
        rightIm.fill(0);
        const { partitions, re, im } = filter;
        const depth = Math.min(partitions, size);
        for (let p = 0; p < depth; p++) {
          const xr = lineRe[(head - p + size) % size];
          const xi = lineIm[(head - p + size) % size];
          const l = p * BINS;
          const r = (partitions + p) * BINS;
          for (let k = 0; k < BINS; k++) {
            const a = xr[k];
            const b = xi[k];
            leftRe[k] += a * re[l + k] - b * im[l + k];
            leftIm[k] += a * im[l + k] + b * re[l + k];
            rightRe[k] += a * re[r + k] - b * im[r + k];
            rightIm[k] += a * im[r + k] + b * re[r + k];
          }
        }
        // Both ears through one inverse transform: left + i right. Each is
        // the spectrum of a real signal, so its upper half is the conjugate
        // of the lower, and the packed signal's real and imaginary parts come
        // back out as the two ears.
        for (let k = 0; k < BINS; k++) {
          workRe[k] = leftRe[k] - rightIm[k];
          workIm[k] = leftIm[k] + rightRe[k];
        }
        for (let k = 1; k < BLOCK; k++) {
          workRe[SIZE - k] = leftRe[k] + rightIm[k];
          workIm[SIZE - k] = rightRe[k] - leftIm[k];
        }
        fft(workRe, workIm, true);
        // Overlap-save: the second half is the block's output.
        for (let n = 0; n < BLOCK; n++) {
          const gain = ring.gainAt(slot, time + n * step);
          left[n] += gain * workRe[BLOCK + n];
          right[n] += gain * workIm[BLOCK + n];
        }
      }
    },
  };
}
