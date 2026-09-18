/** Which slot the next response goes to, and every slot's gain.
 *
 * A response takes over by fading in while the ones before it fade out, so
 * they must be in different slots, and a slot must be silent before it is
 * given a new response: a `ConvolverNode` given a buffer is cleared, and a
 * filter swapped under a sounding slot of the early convolver is an instant
 * change of filter; either is a click. With a fade longer than the interval
 * between responses -- which is what keeps the filter moving rather than
 * stepping -- several slots are fading out at once.
 *
 * So the ring schedules every gain itself, analytically. The early part's
 * convolver (`partitioned.js`) reads the gains straight off it, sample by
 * sample; `engine.js` plays its curves on the tail's `GainNode`s; and the
 * offline model in `tests/js/graph.mjs` plays the same curves, so none of
 * them can disagree about which slot sounds how loud. Two rules:
 *
 * - **A response that is superseded fades to exactly zero over one fade, from
 *   wherever it was, and is never touched again.** The ring used to re-ramp
 *   every fading convolver at every swap instead. That also sums to one, but
 *   it restarts each fade from where it had got to, so a gain approaches zero
 *   geometrically and never arrives: at four convolvers a slot was still 21 dB
 *   down when it was written to, and that was the faint crackle left in a walk
 *   once everything louder was gone. Before that, the same never-arriving
 *   fades made every convolver look equally busy, and a rule choosing the one
 *   free longest collapsed the ring to two slots.
 * - **The incoming response takes whatever the outgoing ones leave**, so the
 *   gains sum to one at every instant however the fades overlap. When they
 *   did not already sum to one -- the first response, or one after a silence
 *   -- the shortfall closes by the same shape over one fade from when it
 *   opened, whatever responses arrive meanwhile. Closing it afresh at each
 *   response instead is the geometric trap again: half of it left per swap.
 *
 * A slot is reused in turn, and only once its fade is over; a response that
 * arrives before any slot is free goes to the quietest one, which is the only
 * case left in which a convolver is written to while audible.
 */

//: Samples per fade, as `setValueCurveAtTime` takes them. The engine and the
//: model both interpolate linearly between them, as the browser does.
const POINTS = 64;

/** The shape of a fade: a smootherstep, an S with no first or second
 * derivative at either end, and `shape(t) + shape(1 - t)` is one. It measured
 * smoother than the raised cosine it replaced on every timbre number.
 *
 * The gains sum to one rather than squaring to one on purpose: two responses a
 * head update apart are nearly the same signal, and the square root fade,
 * measured, doubles the level jerk of a turn.
 */
function shape(t) {
  const u = Math.max(0, Math.min(1, t));
  return u * u * u * (u * (u * 6 - 15) + 10);
}

export function createRing(count, fade) {
  if (count < 2) throw new Error(`a ring needs at least two slots, not ${count}`);
  // Each slot's current schedule: a curve from `start`, held at both ends.
  const slots = Array.from({ length: count }, () => ({ start: 0, duration: 0, samples: null }));
  let live = null;
  let turn = -1;
  // What the gains still fall short of one by, closing over one fade from
  // `start`: after the first response, or after a silence.
  let deficit = null;
  const deficitAt = (t) =>
    deficit && t < deficit.start + fade ? deficit.amount * (1 - shape((t - deficit.start) / fade)) : 0;

  /** A slot's gain at `t`, as the audio thread will have it. */
  function gainAt(index, t) {
    const { start, duration, samples } = slots[index];
    if (!samples) return 0;
    const last = samples.length - 1;
    if (t <= start) return samples[0];
    if (t >= start + duration) return samples[last];
    const position = ((t - start) / duration) * last;
    const below = Math.min(last - 1, Math.floor(position));
    return samples[below] + (samples[below + 1] - samples[below]) * (position - below);
  }

  /** One fade from `now`, sampled as `value(t)`. */
  const curve = (now, value) =>
    Float32Array.from({ length: POINTS }, (_, i) => value(now + (fade * i) / (POINTS - 1)));

  function schedule(index, now, duration, samples) {
    slots[index] = { start: now, duration, samples };
    return { slot: index, start: now, duration, samples };
  }

  /** Fade `index` from wherever it is to exactly zero, over one fade from `now`. */
  function fadeOut(index, now) {
    const from = gainAt(index, now);
    return schedule(index, now, fade, curve(now, (t) => from * (1 - shape((t - now) / fade))));
  }

  const silent = (index, now) => gainAt(index, now) === 0 && gainAt(index, Infinity) === 0;

  /** The next slot in turn that is silent, or failing that the quietest. */
  function pick(now) {
    for (let step = 1; step <= count; step++) {
      const index = (turn + step) % count;
      if (index !== live && silent(index, now)) return index;
    }
    let quietest = null;
    for (let index = 0; index < count; index++) {
      if (index === live) continue;
      if (quietest === null || gainAt(index, now) < gainAt(quietest, now)) quietest = index;
    }
    return quietest;
  }

  return {
    count,
    fade,
    gainAt,
    /** The slot sounding at full gain, or null before the first response. */
    get live() {
      return live;
    },
    /** A new response at `now`: the slot it goes to, and the curves to play.
     *
     * `curves` holds the new schedule of every slot whose gain changes -- the
     * one going and the one coming -- each to be played from `start` over
     * `duration`, replacing whatever that slot was doing. Every other slot
     * carries on down the fade it was already given.
     */
    take(now) {
      const next = pick(now);
      const curves = [];
      // Nothing was live: this is the first response, or the first after a
      // silence, and whatever the others still hold is all that sounds.
      const fresh = live === null;
      if (!fresh) curves.push(fadeOut(live, now));
      const others = (t) => {
        let sum = 0;
        for (let index = 0; index < count; index++) if (index !== next) sum += gainAt(index, t);
        return sum;
      };
      // The shortfall opens only then. It is not detected by adding the gains
      // up: curves sampled on different grids interpolate to within a few
      // thousandths of each other, and a shortfall restarted on that at every
      // response closes by a fraction per response, which is the geometric
      // trap again.
      if (fresh) deficit = { start: now, amount: Math.max(0, 1 - others(now)) };
      const samples = curve(now, (t) => Math.max(0, Math.min(1, 1 - others(t) - deficitAt(t))));
      curves.push(schedule(next, now, fade, samples));
      turn = next;
      live = next;
      return { slot: next, curves };
    },
    /** Everything fades to zero from `now`; the curves to play, as `take`. */
    silence(now) {
      const curves = [];
      for (let index = 0; index < count; index++) {
        if (!silent(index, now)) curves.push(fadeOut(index, now));
      }
      live = null;
      deficit = null;
      return curves;
    },
  };
}
