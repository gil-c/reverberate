/** The engine's convolutions, as filters over time rather than as sound.
 *
 * What a listener is filtered by at any instant is the gain-weighted sum of
 * every response sounding then; its trajectory over a walk is what "smooth"
 * means, and it has no noise floor. The gains are `ring.js`'s own, the ones
 * the page plays, so this cannot disagree with the engine about which slot
 * sounds how loud.
 *
 * `hot` is the early convolver of `partitioned.js`: a response is applied to
 * the input's whole past, so it sounds complete from its first sample.
 * Otherwise a slot is a `ConvolverNode`, which convolves only what arrives
 * after its buffer, so a response that has sounded for `t` seconds is heard
 * as its first `t` seconds.
 */
import { createRing } from "../../src/reverberate/viz/app/audio/ring.js";

export function createPair({ sampleRate, fade, count, hot = false }) {
  const ring = createRing(count, fade);
  // Every response a slot has held: from when, until when, and the caller's
  // own label for it (`weighted` averages the labels).
  const held = Array.from({ length: count }, () => []);
  // The ring keeps only each slot's latest curve; the earlier ones are kept
  // here, as they are played, so a gain can be read at any past time.
  const curves = Array.from({ length: count }, () => []);

  const holding = (slot, at) => held[slot].findLast((entry) => entry.from <= at && at < entry.until);

  function gainAt(slot, t) {
    const curve = curves[slot].findLast((candidate) => candidate.start <= t);
    if (!curve) return 0;
    const last = curve.samples.length - 1;
    const position = Math.min(last, ((t - curve.start) / curve.duration) * last);
    const below = Math.min(last - 1, Math.floor(position));
    return curve.samples[below] + (curve.samples[below + 1] - curve.samples[below]) * (position - below);
  }

  return {
    /** A new response, taking over from `at` seconds. */
    swap(at, response, about = null) {
      const { slot, curves: played } = ring.take(at);
      const previous = held[slot][held[slot].length - 1];
      if (previous) previous.until = at;
      held[slot].push({ from: at, until: Infinity, response, about });
      for (const curve of played) curves[curve.slot].push(curve);
    },
    /** The filter this pair applies at `at`, its first `length` samples,
     *  added into `into`. */
    effective(at, length, into) {
      for (let slot = 0; slot < count; slot++) {
        const entry = holding(slot, at);
        const gain = gainAt(slot, at);
        if (!entry || gain === 0) continue;
        const sounded = hot ? Infinity : Math.round((at - entry.from) * sampleRate);
        for (let ear = 0; ear < 2; ear++) {
          const response = entry.response[ear];
          const end = Math.min(length, response.length, sounded + 1);
          for (let i = 0; i < end; i++) into[ear][i] += gain * response[i];
        }
      }
      return into;
    },
    /** The gain-weighted mean of the labels of the responses sounding at
     *  `at`: for the early part, the head yaw the listener is hearing. */
    weighted(at) {
      let total = 0;
      let sum = 0;
      for (let slot = 0; slot < count; slot++) {
        const entry = holding(slot, at);
        if (!entry || entry.about === null) continue;
        const gain = gainAt(slot, at);
        total += gain;
        sum += gain * entry.about;
      }
      return total > 1e-9 ? sum / total : null;
    },
    /** How many responses took over. */
    swaps: () => held.reduce((sum, entries) => sum + entries.length, 0),
  };
}
