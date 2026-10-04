/** A scene's sound, decoded: the engine's order 7 stream, a head, two ears.
 *
 * The server sends what the signal engine rendered, 64 channels in the
 * scene's fixed frame. Everything between that and the headphones is here,
 * and it is the decode the page already does for a solver's field
 * (`audio/decode.js`): the field turned by the head's matrix
 * (`audio/sh.js`, `rotationBlocks`), then each channel through the decoder's
 * filter for each ear (`viz/decoders.py`), summed. Nothing is rendered.
 *
 * **The head turns the filters, not the stream.** An ear is
 * `sum_i F_i * (R c)_i`, which is `sum_j (sum_i R_ij F_i) * c_j`: turning the
 * 128 filters by the transpose is the same sum, costs a few milliseconds on
 * the page's thread once per movement of the head, and leaves the audio
 * thread one fixed job, 64 channels into two ears.
 *
 * **That job is overlap-save in blocks of 512.** The window of a block is
 * the 512 samples before it and its own; each channel is transformed once,
 * multiplied by its two filters, and the two sums are transformed back. The
 * transforms of a block are spread over the four render quanta before it
 * sounds, sixteen channels each, and the filters are applied in the last of
 * them: a movement of the head is heard within a block, 10.7 ms, and the
 * stream itself, known seconds ahead, is not delayed at all.
 *
 * **Nothing is switched.** A new head is a crossfade over one block between
 * the two decodes of the same samples; a new mix (a solo, a mute) is a
 * crossfade over one block between the two streams under the same head; a
 * start is faded in and a stop faded out.
 *
 * Pure: no worklet, no messages, no clock. `sound.worklet.js` is the shell.
 */
import { fft } from "../audio/fft.js";
import { headYawOfCamera, rotationBlocks } from "../audio/sh.js";

//: Samples per block, and per render quantum of Web Audio.
export const BLOCK = 512;
export const QUANTUM = 128;
const SIZE = 2 * BLOCK;
export const BINS = BLOCK + 1;
const PHASES = BLOCK / QUANTUM;

/** The head's matrix for a camera yaw and pitch and a roll, radians: the
 * recipe's `Rz(yaw) Ry(-pitch) Rx(roll)`, which for no roll is `headMatrix`
 * of `audio/sh.js`. A positive roll lowers the right ear. Row-major. */
export function headMatrix3(cameraYaw, cameraPitch, roll = 0) {
  const psi = headYawOfCamera(cameraYaw);
  const cz = Math.cos(psi);
  const sz = Math.sin(psi);
  const cy = Math.cos(-cameraPitch);
  const sy = Math.sin(-cameraPitch);
  const cx = Math.cos(roll);
  const sx = Math.sin(roll);
  return [
    cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
    sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
    -sy, cy * sx, cy * cx,
  ];
}

/** The decoder's filters as half spectra of a block's transform:
 * `re` and `im` hold `[ear][channel][bin]`. */
export function filterSpectra({ channels, taps, filters }) {
  if (taps > BLOCK + 1) throw new Error(`a decoder of ${taps} taps does not fit blocks of ${BLOCK}`);
  const re = new Float32Array(2 * channels * BINS);
  const im = new Float32Array(2 * channels * BINS);
  const workRe = new Float32Array(SIZE);
  const workIm = new Float32Array(SIZE);
  for (let row = 0; row < 2 * channels; row++) {
    workRe.fill(0);
    workIm.fill(0);
    workRe.set(filters.subarray(row * taps, (row + 1) * taps));
    fft(workRe, workIm);
    re.set(workRe.subarray(0, BINS), row * BINS);
    im.set(workIm.subarray(0, BINS), row * BINS);
  }
  return { channels, re, im };
}

/** The filters turned by the head `M`: `G_j = sum_i R_ij F_i`, degree by degree.
 * Written into `into` when given, which must be of the spectra's size. */
export function rotateSpectra(spectra, order, M, into = null) {
  const { channels } = spectra;
  const out = into || { re: new Float32Array(spectra.re.length), im: new Float32Array(spectra.im.length) };
  out.re.fill(0);
  out.im.fill(0);
  for (const { offset, size, matrix } of rotationBlocks(order, M)) {
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const weight = matrix[i * size + j];
        if (Math.abs(weight) < 1e-12) continue;
        for (let ear = 0; ear < 2; ear++) {
          const from = (ear * channels + offset + i) * BINS;
          const to = (ear * channels + offset + j) * BINS;
          for (let b = 0; b < BINS; b++) {
            out.re[to + b] += weight * spectra.re[from + b];
            out.im[to + b] += weight * spectra.im[from + b];
          }
        }
      }
    }
  }
  return out;
}

/** Whether the chunks of `store` hold every sample of `[from, to)`. */
function covers(store, from, to) {
  let t = from;
  while (t < to) {
    let next = -1;
    for (const chunk of store.chunks) {
      if (chunk.start <= t && chunk.start + chunk.frames > t) next = Math.max(next, chunk.start + chunk.frames);
    }
    if (next < 0) return false;
    t = next;
  }
  return true;
}

/** The stream's decoder: chunks of frames in, quanta of two ears out.
 *
 * `total` is the scene's length in samples. Chunks are `[frame][channel]`
 * float32, as the server sends them, pushed under a generation: `start` opens
 * one at a sample, and a chunk of a later generation than the one playing is
 * a new mix, crossfaded to as soon as it covers the block to come. The
 * decoder waits (silence, the clock stopped) from a start, and whenever it
 * runs out of samples, until a push says `go`.
 */
export function createStreamDecoder({ channels }) {
  const half = channels * BINS;
  const spectra = () => ({ re: new Float32Array(half), im: new Float32Array(half) });
  const X = [spectra(), spectra()]; // the block to come: of the stream playing, of the one coming
  const workRe = new Float32Array(SIZE);
  const workIm = new Float32Array(SIZE);
  const accRe = new Float32Array(SIZE);
  const accIm = new Float32Array(SIZE);
  // Two decodes of the block to come, crossfaded when something changed.
  const made = [0, 1].map(() => [new Float32Array(BLOCK), new Float32Array(BLOCK)]);
  const fifo = [new Float32Array(BLOCK), new Float32Array(BLOCK)];
  let fifoRead = BLOCK;
  // What was sounding when the stream was stopped or moved, faded out over a quantum.
  const tail = [new Float32Array(QUANTUM), new Float32Array(QUANTUM)];
  let tailPending = false;

  let filters = null;
  let nextFilters = null;
  const spent = []; // filters no longer used, for the thread that made them
  let stores = []; // [playing] or [playing, coming]
  let total = 0;
  let running = false;
  let waiting = true;
  let position = 0; // the first sample of the block to come
  let phase = 0;
  let ready = false;
  let blockStores = 1; // how many stores the block to come is made from
  let fresh = true; // the block to come follows silence: it is faded in
  let played = 0;
  let ended = false;
  let starved = false;

  function gather(store, channel, from) {
    workRe.fill(0);
    workIm.fill(0);
    for (const chunk of store.chunks) {
      const low = Math.max(from, chunk.start);
      const high = Math.min(from + SIZE, chunk.start + chunk.frames);
      for (let n = low; n < high; n++) workRe[n - from] = chunk.data[(n - chunk.start) * channels + channel];
    }
  }

  function transform(store, into, first, count) {
    for (let channel = first; channel < first + count; channel++) {
      gather(store, channel, position - BLOCK);
      fft(workRe, workIm);
      into.re.set(workRe.subarray(0, BINS), channel * BINS);
      into.im.set(workIm.subarray(0, BINS), channel * BINS);
    }
  }

  /** The block to come at one stream and one set of filters: its two ears. */
  function decode(x, g, out) {
    for (let ear = 0; ear < 2; ear++) {
      accRe.fill(0);
      accIm.fill(0);
      for (let channel = 0; channel < channels; channel++) {
        const a = channel * BINS;
        const b = (ear * channels + channel) * BINS;
        for (let i = 0; i < BINS; i++) {
          const xr = x.re[a + i];
          const xi = x.im[a + i];
          const gr = g.re[b + i];
          const gi = g.im[b + i];
          accRe[i] += xr * gr - xi * gi;
          accIm[i] += xr * gi + xi * gr;
        }
      }
      for (let i = 1; i < BLOCK; i++) {
        accRe[SIZE - i] = accRe[i];
        accIm[SIZE - i] = -accIm[i];
      }
      fft(accRe, accIm, true);
      out[ear].set(accRe.subarray(BLOCK, SIZE));
    }
  }

  function fadeOut() {
    const count = Math.min(QUANTUM, BLOCK - fifoRead);
    if (!running || count <= 0) return;
    for (let ear = 0; ear < 2; ear++) {
      tail[ear].fill(0);
      for (let i = 0; i < count; i++) tail[ear][i] = fifo[ear][fifoRead + i] * (1 - (i + 1) / count);
    }
    tailPending = true;
    fifoRead = BLOCK;
  }

  /** What the block to come needs of a store: its own samples, to the scene's end. */
  const holds = (store) => covers(store, Math.max(position, 0), Math.min(position + BLOCK, total));

  function work() {
    if (phase === 0) {
      // A newer mix takes over as soon as it holds the block; the old one is
      // kept for this block only, to fade out of.
      const coming = stores.length > 1 && holds(stores[1]);
      const playing = stores.length > 0 && holds(stores[0]);
      if (!playing && !coming) {
        if (fifoRead >= BLOCK) {
          starved = true;
          waiting = true;
          fresh = true;
        }
        return;
      }
      if (coming && !playing) stores.shift();
      blockStores = coming && playing ? 2 : 1;
    }
    const first = Math.floor((phase * channels) / PHASES);
    const count = Math.floor(((phase + 1) * channels) / PHASES) - first;
    for (let s = 0; s < blockStores; s++) transform(stores[s], X[s], first, count);
    phase += 1;
    if (phase < PHASES) return;
    // The last quantum before the block sounds: the head as it is now.
    decode(X[0], filters, made[0]);
    let faded = false;
    if (blockStores === 2 || nextFilters) {
      decode(X[blockStores - 1], nextFilters || filters, made[1]);
      faded = true;
    }
    for (let ear = 0; ear < 2; ear++) {
      const out = made[0][ear];
      if (faded) {
        const to = made[1][ear];
        for (let i = 0; i < BLOCK; i++) out[i] += (to[i] - out[i]) * ((i + 1) / BLOCK);
      }
      if (fresh) for (let i = 0; i < BLOCK; i++) out[i] *= (i + 1) / BLOCK;
    }
    if (nextFilters) {
      spent.push(filters);
      filters = nextFilters;
      nextFilters = null;
    }
    if (blockStores === 2) stores.shift();
    fresh = false;
    ready = true;
  }

  return {
    /** The scene's length, samples. */
    setTotal(samples) {
      total = samples;
    },
    /** The filters for a head; the newest wins, and takes a block to fade in. */
    setFilters(next) {
      if (!filters) filters = next;
      else {
        if (nextFilters) spent.push(nextFilters);
        nextFilters = next;
      }
    },
    /** Filters that are no longer used, to hand back to whoever made them. */
    takeSpent: () => spent.splice(0),
    /** A stream from `sample`, under generation `gen`: silent until a push says go. */
    start(sample, gen) {
      fadeOut();
      stores = [{ gen, chunks: [] }];
      position = sample;
      phase = 0;
      ready = false;
      fifoRead = BLOCK;
      running = true;
      waiting = true;
      fresh = true;
      ended = false;
      starved = false;
    },
    /** Stop, fading what is sounding out over one quantum. */
    stop() {
      fadeOut();
      running = false;
    },
    push({ gen, start, frames, data, go }) {
      const store = stores.find((s) => s.gen === gen);
      if (store) store.chunks.push({ start, frames, data });
      else if (stores.length && gen > stores[stores.length - 1].gen) {
        // Only the newest mix is worth fading to.
        if (stores.length > 1 && !(phase > 0 && blockStores === 2)) stores.pop();
        stores.push({ gen, chunks: [{ start, frames, data }] });
      } else return false;
      if (go) {
        waiting = false;
        starved = false;
      }
      return true;
    },
    /** One render quantum. Returns whether scene samples were written. */
    process(left, right) {
      left.fill(0);
      right.fill(0);
      if (tailPending) {
        left.set(tail[0].subarray(0, left.length));
        right.set(tail[1].subarray(0, right.length));
        tailPending = false;
      }
      if (!running || !filters) return false;
      if (fifoRead >= BLOCK && ready) {
        fifo[0].set(made[0][0]);
        fifo[1].set(made[0][1]);
        fifoRead = 0;
        position += BLOCK;
        phase = 0;
        ready = false;
        // What lies wholly before the next window is done with.
        for (const store of stores) {
          store.chunks = store.chunks.filter((chunk) => chunk.start + chunk.frames > position - BLOCK);
        }
        if (position - BLOCK >= total) {
          running = false;
          ended = true;
          return false;
        }
      }
      let wrote = false;
      if (fifoRead < BLOCK) {
        const count = Math.min(left.length, BLOCK - fifoRead);
        for (let i = 0; i < count; i++) {
          left[i] = fifo[0][fifoRead + i];
          right[i] = fifo[1][fifoRead + i];
        }
        fifoRead += count;
        played += count;
        wrote = true;
      }
      if (!ready && !waiting) work();
      return wrote;
    },
    state: () => ({
      running,
      waiting,
      starved,
      ended,
      played,
      // The sample being heard now.
      position: position - (BLOCK - fifoRead),
      buffered: stores.length ? stores[stores.length - 1].chunks.reduce((n, c) => Math.max(n, c.start + c.frames), 0) : 0,
    }),
  };
}
