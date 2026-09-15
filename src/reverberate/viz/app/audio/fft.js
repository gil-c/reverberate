/** A radix-2 complex FFT on split real and imaginary arrays.
 *
 * Small and dependency free. Everything the app transforms is a power of two
 * it chose itself, so no other size is supported. Used from the worker for
 * the decode and from the plots; never on the audio thread.
 */
const tables = new Map();

function table(n) {
  let entry = tables.get(n);
  if (entry) return entry;
  const cos = new Float32Array(n / 2);
  const sin = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = Math.sin((2 * Math.PI * i) / n);
  }
  const rev = new Uint32Array(n);
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  entry = { cos, sin, rev };
  tables.set(n, entry);
  return entry;
}

/** In-place forward (`inverse` false) or inverse transform of `re`, `im`. */
export function fft(re, im, inverse = false) {
  const n = re.length;
  if (n & (n - 1)) throw new Error(`fft size ${n} is not a power of two`);
  const { cos, sin, rev } = table(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step];
        const wi = inverse ? sin[k * step] : -sin[k * step];
        const a = start + k;
        const b = a + half;
        const tr = re[b] * wr - im[b] * wi;
        const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/** The smallest power of two at or above `n`. */
export function nextPowerOfTwo(n) {
  let size = 1;
  while (size < n) size <<= 1;
  return size;
}
