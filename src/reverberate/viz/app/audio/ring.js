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

/** A curve's value at `t`: linear between its samples, as the browser plays
 *  `setValueCurveAtTime`, and held at both ends. */
export function curveValue(curve, t) {
  const { start, duration, samples } = curve;
  const last = samples.length - 1;
  if (t <= start) return samples[0];
  if (t >= start + duration) return samples[last];
  const position = ((t - start) / duration) * last;
  const below = Math.min(last - 1, Math.floor(position));
  return samples[below] + (samples[below + 1] - samples[below]) * (position - below);
}

export function createRing(count, fade) {
  if (count < 2) throw new Error(`a ring needs at least two slots, not ${count}`);
  // Each slot's schedule, a curve from `start` held at both ends, in storage
  // of its own that every new schedule of that slot overwrites. The ring runs
  // on the audio thread, inside the early convolver, where every allocation
  // is garbage the collector may stop the audio to take away.
  const slots = Array.from({ length: count }, (_, slot) => ({
    slot,
    start: 0,
    duration: fade,
    samples: new Float32Array(POINTS),
    scheduled: false,
  }));
  const played = [];
  const taken = { slot: 0, curves: played };
  let live = null;
  let turn = -1;
  // What the gains still fall short of one by, closing over one fade from
  // `deficitStart`: after the first response, or after a silence.
  let deficitStart = -Infinity;
  let deficitAmount = 0;
  const deficitAt = (t) =>
    t < deficitStart + fade ? deficitAmount * (1 - shape((t - deficitStart) / fade)) : 0;

  /** A slot's gain at `t`, as the audio thread will have it. */
  const gainAt = (index, t) => (slots[index].scheduled ? curveValue(slots[index], t) : 0);

  /** The sum of every slot's gain at `t` but `except`'s. */
  function others(except, t) {
    let sum = 0;
    for (let index = 0; index < count; index++) if (index !== except) sum += gainAt(index, t);
    return sum;
  }

  const moment = (now, i) => now + (fade * i) / (POINTS - 1);

  /** Fade `index` from wherever it is to exactly zero, over one fade from `now`. */
  function fadeOut(index, now) {
    const slot = slots[index];
    const from = gainAt(index, now);
    for (let i = 0; i < POINTS; i++) slot.samples[i] = from * (1 - shape(i / (POINTS - 1)));
    slot.start = now;
    slot.scheduled = true;
    played.push(slot);
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
     * one going and the one coming -- each `{ slot, start, duration, samples }`
     * to be played from `start`, replacing whatever that slot was doing; every
     * other slot carries on down the fade it was already given. They are the
     * ring's own storage, good until the next call: `setValueCurveAtTime`
     * copies them, and anything that keeps them must too.
     */
    take(now) {
      played.length = 0;
      const next = pick(now);
      // Nothing was live: this is the first response, or the first after a
      // silence, and whatever the others still hold is all that sounds.
      const fresh = live === null;
      if (!fresh) fadeOut(live, now);
      // The shortfall opens only then. It is not detected by adding the gains
      // up: curves sampled on different grids interpolate to within a few
      // thousandths of each other, and a shortfall restarted on that at every
      // response closes by a fraction per response, which is the geometric
      // trap again.
      if (fresh) {
        deficitStart = now;
        deficitAmount = Math.max(0, 1 - others(next, now));
      }
      const slot = slots[next];
      for (let i = 0; i < POINTS; i++) {
        const t = moment(now, i);
        slot.samples[i] = Math.max(0, Math.min(1, 1 - others(next, t) - deficitAt(t)));
      }
      slot.start = now;
      slot.scheduled = true;
      played.push(slot);
      turn = next;
      live = next;
      taken.slot = next;
      return taken;
    },
    /** Everything fades to zero from `now`; the curves to play, as `take`. */
    silence(now) {
      played.length = 0;
      for (let index = 0; index < count; index++) if (!silent(index, now)) fadeOut(index, now);
      live = null;
      deficitAmount = 0;
      return played;
    },
  };
}
