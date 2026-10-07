/** A clock with a player's manners, for a page on which nothing sounds.
 *
 *   const clock = createClock(180);
 *   createTransport(element, clock, { level: false, head: false, bar: false });
 *   createTimeline(element, scene, clock);
 *
 * It plays, stops, seeks and loops as `createPlayer` does and says so with
 * the same events (`state`, `time`), so that a transport, a timeline and a
 * scene view cannot tell: a recipe is looked at with the parts a scene is
 * listened to with.
 */
export function createClock(duration) {
  const listeners = {};
  const emit = (name, detail) => (listeners[name] || []).forEach((fn) => fn(detail));
  let playing = false;
  let looping = false;
  let loop = null;
  let position = 0; // seconds, when it last started or stopped
  let since = 0; // performance.now() when it last started

  const time = () => (playing ? Math.min(position + (performance.now() - since) / 1000, duration) : position);
  const state = () => ({ playing, looping, loop: loop ? [...loop] : null, duration, levelDb: 0, turned: { yaw: 0, pitch: 0 }, following: true, kind: null, failure: "" });
  const begin = (seconds) => {
    position = Math.max(0, Math.min(seconds, duration));
    since = performance.now();
  };

  function frame() {
    if (playing) {
      const now = time();
      const end = looping && loop ? loop[1] : duration;
      if (now >= end) {
        if (looping) begin(loop ? loop[0] : 0);
        else {
          playing = false;
          position = 0;
          emit("state", state());
        }
      }
      emit("time", time());
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  const clock = {
    on(name, fn) {
      (listeners[name] = listeners[name] || []).push(fn);
      return clock;
    },
    play() {
      if (playing) return;
      playing = true;
      const from = looping && loop && (position < loop[0] || position >= loop[1]) ? loop[0] : position;
      begin(from >= duration ? 0 : from);
      emit("state", state());
    },
    stop() {
      if (!playing) return;
      position = time();
      playing = false;
      emit("state", state());
      emit("time", position);
    },
    toggle: () => (playing ? clock.stop() : clock.play()),
    seek(seconds) {
      begin(seconds);
      emit("time", position);
    },
    setLoop(region) {
      loop = region && region[1] - region[0] > 0.05 ? [region[0], region[1]] : null;
      emit("state", state());
    },
    setLooping(on) {
      looping = Boolean(on);
      emit("state", state());
    },
    pose: () => ({ yaw: 0, pitch: 0, roll: 0 }),
    time,
    state,
  };
  return clock;
}
