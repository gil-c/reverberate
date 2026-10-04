/** The scene's clock, and which sources are heard: the seam sound attaches to.
 *
 * Nothing here draws and nothing here knows the DOM. The scene view moves its
 * markers on `tick`; an audio stage starts, stops and relocates its stream on
 * `play`, `pause`, `seek` and `speed`, and reads `mix.audible(id)` to know
 * what to render. Both read the same `time()`, so a marker and a sound cannot
 * be at two instants.
 *
 * The clock is whatever `now` returns, in seconds: the page's own by default.
 * A stage that plays samples hands in its audio clock with `setClock`, and
 * the picture then follows the sound and not the other way round.
 */
export const SPEEDS = [1, 4, 16];

function createEvents() {
  const handlers = new Map();
  return {
    /** Listen to `event`; the function returned stops listening. */
    on(event, handler) {
      if (!handlers.has(event)) handlers.set(event, new Set());
      handlers.get(event).add(handler);
      return () => handlers.get(event).delete(handler);
    },
    emit(event, detail) {
      for (const handler of handlers.get(event) || []) handler(detail);
    },
  };
}

/** Scene time against a clock.
 *
 * Events, each with `{ t, speed, playing, duration }`:
 * `load` (a scene of a new length, at rest at zero), `play`, `pause`, `seek`
 * (a jump, playing or not), `speed`, `end` (the scene ran out; a `pause`
 * follows) and `tick` (once per frame drawn, and after every other event).
 */
export function createTransport({ now = () => performance.now() / 1000 } = {}) {
  const events = createEvents();
  let clock = now;
  let duration = 0;
  let playing = false;
  let speed = 1;
  // Scene time `anchorT` was true at clock time `anchorAt`.
  let anchorT = 0;
  let anchorAt = 0;

  const time = () => (playing ? Math.min(duration, anchorT + (clock() - anchorAt) * speed) : anchorT);
  const state = () => ({ t: time(), speed, playing, duration });
  const anchor = (t) => {
    anchorT = Math.min(duration, Math.max(0, t));
    anchorAt = clock();
  };
  const say = (event) => {
    events.emit(event, state());
    if (event !== "tick") events.emit("tick", state());
  };

  const transport = {
    on: events.on,
    time,
    state,
    /** A scene of `seconds`: stopped, at its start. */
    load(seconds) {
      playing = false;
      duration = Math.max(0, seconds || 0);
      anchor(0);
      say("load");
    },
    play() {
      if (playing || duration <= 0) return;
      // Play at the end starts over: there is nothing after it to play.
      anchor(anchorT >= duration ? 0 : anchorT);
      playing = true;
      say("play");
    },
    pause() {
      if (!playing) return;
      anchor(time());
      playing = false;
      say("pause");
    },
    toggle() {
      if (playing) transport.pause();
      else transport.play();
    },
    seek(t) {
      anchor(Number.isFinite(t) ? t : 0);
      say("seek");
    },
    setSpeed(value) {
      if (!(value > 0) || value === speed) return;
      anchor(time());
      speed = value;
      say("speed");
    },
    /** Called once per frame by whoever draws; stops the scene at its end. */
    tick() {
      if (playing && time() >= duration) {
        anchor(duration);
        playing = false;
        say("end");
        say("pause");
        return;
      }
      say("tick");
    },
    /** Another clock, in seconds: scene time carries over without a jump. */
    setClock(next) {
      const t = time();
      clock = next;
      anchor(t);
    },
  };
  return transport;
}

/** Solo and mute per source.
 *
 * A source is audible when it is soloed, or when nothing is soloed and it is
 * not muted: a solo wins over a mute elsewhere, as on a mixing desk. `change`
 * carries `{ solo, mute, audible }`, three arrays of ids.
 */
export function createMix() {
  const events = createEvents();
  let ids = [];
  const solo = new Set();
  const mute = new Set();
  const audible = (id) => (solo.size ? solo.has(id) : !mute.has(id));
  const state = () => ({ solo: [...solo], mute: [...mute], audible: ids.filter(audible) });
  const flip = (set, id) => {
    if (!ids.includes(id)) return;
    if (set.has(id)) set.delete(id);
    else set.add(id);
    events.emit("change", state());
  };
  return {
    on: events.on,
    state,
    audible,
    isSolo: (id) => solo.has(id),
    isMuted: (id) => mute.has(id),
    /** The sources of a new scene: every one heard. */
    setSources(list) {
      ids = [...list];
      solo.clear();
      mute.clear();
      events.emit("change", state());
    },
    toggleSolo: (id) => flip(solo, id),
    toggleMute: (id) => flip(mute, id),
  };
}
