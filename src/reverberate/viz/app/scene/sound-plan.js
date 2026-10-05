/** A scene's sound, scheduled: the clock the timeline follows, and which chunks to fetch.
 *
 * Nothing here plays, fetches or draws; `sound.js` does, with these, and
 * `tests/js/sound.mjs` drives them without a browser.
 */

/** The audio clock handed to `transport.setClock`, in seconds.
 *
 * It advances by the samples of the scene that were played and by nothing
 * else: while the stream waits for a chunk that is still being rendered the
 * clock stands still, and the timeline with it. `report` is what the audio
 * thread says once a block; between two reports the clock runs on the
 * context's own time, by no more than `ahead`, so a picture drawn sixty
 * times a second does not move in steps of a block.
 *
 * A stream that is moved (a seek, a start) is not the old one continued:
 * `hold` freezes the clock at the instant the transport anchored on it, and
 * `begin` restarts it from the audio thread's count when the new stream is
 * taken, so no sample of the old stream moves the new time.
 */
export function createSoundClock({ sampleRate, contextTime, ahead = 0.03 }) {
  let base = 0; // the clock when the current stream began, seconds
  let from = null; // the audio thread's count when it began; null while held
  let played = 0;
  let at = 0;
  let running = false;
  let last = 0;
  const read = () => {
    if (from === null) return base;
    const lead = running ? Math.min(Math.max(contextTime() - at, 0), ahead) : 0;
    return base + (played - from) / sampleRate + lead;
  };
  return {
    /** Never runs backwards, whatever the reports' jitter. */
    now() {
      last = Math.max(last, read());
      return last;
    },
    hold() {
      base = this.now();
      from = null;
    },
    /** The new stream is taken; `count` is the audio thread's when it was. */
    begin(count) {
      from = count;
      played = count;
      running = false;
    },
    /** From the audio thread: samples of the scene played so far, when, and whether more follow. */
    report(count, when, isRunning) {
      if (from === null || count < from) return;
      played = count;
      at = when;
      running = isRunning;
    },
    held: () => from === null,
  };
}

/** Which chunks the page asks the server for, and under which generation.
 *
 * A generation is one stream: it changes with every seek, start, solo, mute
 * and switch, and an answer of an older generation is dropped. Chunks are
 * asked in order from the one under the cursor, `ahead` of them at most,
 * `parallel` at a time; the audio thread holds what was pushed.
 */
export function createFetchPlan({ chunkSamples, chunks, ahead = 6, parallel = 2 }) {
  let gen = 0;
  let pushed = new Set();
  let flying = new Set();
  const at = (sample) => Math.min(chunks - 1, Math.max(0, Math.floor(sample / chunkSamples)));
  return {
    gen: () => gen,
    chunkAt: at,
    /** A new stream: nothing of the old one counts. */
    restart() {
      gen += 1;
      pushed = new Set();
      flying = new Set();
      return gen;
    },
    /** The chunks to ask for now, the stream being at `sample`. */
    next(sample) {
      const first = at(sample);
      const wanted = [];
      for (let index = first; index < Math.min(chunks, first + ahead); index++) {
        if (flying.size + wanted.length >= parallel) break;
        if (!pushed.has(index) && !flying.has(index)) wanted.push(index);
      }
      for (const index of wanted) flying.add(index);
      return wanted;
    },
    /** An answer came: whether it is still wanted. */
    arrived(generation, index) {
      if (generation !== gen) return false;
      flying.delete(index);
      pushed.add(index);
      return true;
    },
    /** An answer did not come: it may be asked again. */
    failed(generation, index) {
      if (generation === gen) flying.delete(index);
    },
    /** Chunks held without a gap from the one under `sample`. */
    buffered(sample) {
      let count = 0;
      for (let index = at(sample); index < chunks && pushed.has(index); index++) count += 1;
      return count;
    },
    /** Whether the stream may sound: its first chunk is in, or after it ran
     *  dry, `resume` chunks or everything to the scene's end. */
    mayGo(sample, { starved = false, resume = 4 } = {}) {
      const held = this.buffered(sample);
      const left = chunks - at(sample);
      return held >= Math.min(starved ? resume : 1, left);
    },
  };
}

/** Runs of chunks `[first, stop)` as spans of seconds. */
export function chunkSpans(ranges, chunkSamples, sampleRate, duration) {
  return ranges.map(([first, stop]) => [
    (first * chunkSamples) / sampleRate,
    Math.min(duration, (stop * chunkSamples) / sampleRate),
  ]);
}

/** The runs of chunks present in every one of `lists`, each a list of `[first, stop)`. */
export function commonRanges(lists) {
  if (!lists.length) return [];
  let common = lists[0];
  for (const list of lists.slice(1)) {
    const next = [];
    let i = 0;
    let j = 0;
    while (i < common.length && j < list.length) {
      const low = Math.max(common[i][0], list[j][0]);
      const high = Math.min(common[i][1], list[j][1]);
      if (high > low) next.push([low, high]);
      if (common[i][1] < list[j][1]) i += 1;
      else j += 1;
    }
    common = next;
  }
  return common;
}

// --- the level the page plays at, and whether it clips ---------------------------

/** The level control's default, dB, and its two ends.
 *
 * A pack is physical: a sample of 1 is 86 dB SPL, a voice at 1 m about 60.
 * At 0 dB three of a scene's fourteen sources already peaked at -3.8 dB re
 * full scale in the two ears, and fourteen pass it. At -12 dB the page's
 * full scale stands for 98 dB SPL: the scene keeps its own levels, 12 dB
 * lower in the file, and the listener makes that up on the volume knob.
 * `render/check/measure.py` holds the same number for the files it writes.
 */
export const FULL_SCALE_SPL_DB = 86;
export const DEFAULT_LEVEL_DB = -12;
export const LEVEL_RANGE_DB = [-60, 20];

/** The control's value as a gain; what is not a number is the default. */
export function levelGain(db) {
  const value = Number(db);
  const [low, high] = LEVEL_RANGE_DB;
  const held = db === "" || db === null || !Number.isFinite(value) ? DEFAULT_LEVEL_DB : Math.min(high, Math.max(low, value));
  return 10 ** (held / 20);
}

/** The largest sample of two ears' blocks, or `held` if that is larger. */
export function peakOf(left, right, held = 0) {
  let peak = held;
  for (let i = 0; i < left.length; i += 1) {
    const a = Math.abs(left[i]);
    const b = Math.abs(right[i]);
    if (a > peak) peak = a;
    if (b > peak) peak = b;
  }
  return peak;
}

/** Whether what leaves the page passes full scale, said for `hold` seconds after it did.
 *
 * `report` takes the largest sample the audio thread made since it last
 * said, before the level control, the control's gain, and the time. Nothing
 * here changes the sound: an output past full scale is clipped by the
 * browser, and the page says so rather than limiting what the owner hears.
 */
export function createClipMeter({ hold = 3 } = {}) {
  let until = -Infinity;
  let worst = 0; // the largest output since the last reset, linear
  let events = 0;
  let over = false;
  return {
    report(peak, gain, now) {
      const out = Math.abs(peak) * gain;
      if (!Number.isFinite(out)) return;
      worst = Math.max(worst, out);
      if (out > 1) {
        if (!over) events += 1;
        over = true;
        until = now + hold;
      } else {
        over = false;
      }
    },
    /** What the page shows at `now`. `overDb` is how far past full scale the worst was. */
    state(now) {
      return {
        clipping: now < until,
        events,
        peakDb: worst > 0 ? 20 * Math.log10(worst) : -Infinity,
        overDb: worst > 1 ? 20 * Math.log10(worst) : 0,
      };
    },
    reset() {
      until = -Infinity;
      worst = 0;
      events = 0;
      over = false;
    },
  };
}

/** A level in dB as a meter's fill, 0 to 1, over `range` dB under `top`. */
export function meterFill(db, top, range = 50) {
  if (db === null || db === undefined || !Number.isFinite(db)) return 0;
  return Math.min(1, Math.max(0, (db - (top - range)) / range));
}

// --- what is drawn of a step's arrivals (`arrivals.js`) ---------------------------

// The bead of the strongest arrival, and of one 40 dB under it, metres.
const BEAD_M = [0.14, 0.03];
const RANGE_DB = 40;
// The tail's shell: its radius where no energy comes from, and what the
// direction of most energy adds.
const SHELL_M = [0.25, 0.45];
// Kinds 2 and 3 are diffracted; they take the colour past the orders'.
const DIFFRACTED_ORDER = 6;

/** What is drawn of one arrival: where its line ends and how large its bead is. */
export function arrivalGlyphs(step, soundSpeed) {
  const count = step.delay_s.length;
  let top = 0;
  for (const gain of step.gain) top = Math.max(top, gain);
  const glyphs = [];
  for (let i = 0; i < count; i++) {
    const reach = step.delay_s[i] * soundSpeed;
    const db = top > 0 && step.gain[i] > 0 ? 20 * Math.log10(step.gain[i] / top) : -Infinity;
    const share = Math.min(1, Math.max(0, 1 + db / RANGE_DB));
    glyphs.push({
      end: [0, 1, 2].map((axis) => step.listener[axis] + step.direction[3 * i + axis] * reach),
      radius: BEAD_M[1] + (BEAD_M[0] - BEAD_M[1]) * share,
      order: step.kind[i] >= 2 ? DIFFRACTED_ORDER : step.order[i],
      db,
    });
  }
  return glyphs;
}

/** The shell's vertices for a tail's energies on the grid, `[x, y, z, ...]`. */
export function shellPoints(listener, directions, energy) {
  const points = new Float32Array(directions.length);
  for (let i = 0; i < energy.length; i++) {
    const radius = SHELL_M[0] + SHELL_M[1] * Math.sqrt(energy[i]);
    for (let axis = 0; axis < 3; axis++) points[3 * i + axis] = listener[axis] + directions[3 * i + axis] * radius;
  }
  return points;
}
