/** The scene view's arithmetic, checked without a browser.
 *
 * Run by `tests/test_scene_view_js.py`, which hands in the page's `scene`
 * folder and a file of tracks the server sampled, with the kinematics' own
 * answers between the samples to compare against. Prints one JSON object.
 */
import { readFileSync } from "node:fs";

const folder = process.argv[2];
const given = JSON.parse(readFileSync(process.argv[3], "utf8"));
const { createTimeMap, zoomed, panned, ticks, formatTime, laneLayout, visible, posture, MIN_SPAN_S } =
  await import(`${folder}/timemap.js`);
const { bracket, sourceAt, listenerAt, activeAt, spanAt, trail, trailCapacity, viewYaw, heading } =
  await import(`${folder}/tracks.js`);
const { createTransport, createMix, SPEEDS } = await import(`${folder}/transport.js`);
const { labelOf, held } = await import(`${folder}/generator.js`);

const out = {};

// --- time mapping -------------------------------------------------------------
{
  const whole = createTimeMap({ duration: 1200, width: 900 });
  const near = zoomed(whole, 0.1, 300);
  const tiny = zoomed(whole, 1e-6, 300);
  const back = zoomed(near, 1e6, 300);
  const late = panned(near, 1e9);
  const marks = ticks(whole);
  out.map = {
    ends: [whole.toX(0), whole.toX(1200), whole.toT(450)],
    outside: [whole.toT(-50), whole.toT(5000)],
    roundTrip: Math.max(...[0, 17.3, 600, 1199.9].map((t) => Math.abs(near.toT(near.toX(t)) - t))),
    held: [whole.toT(300), near.toT(300)],
    nearSpan: near.span,
    tinySpan: tiny.span,
    minSpan: MIN_SPAN_S,
    back: [back.start, back.span],
    late: [late.start, late.end],
    step: marks.step,
    gaps: marks.marks.slice(1).map((mark, i) => mark.x - marks.marks[i].x),
    labels: marks.marks.slice(0, 3).map((mark) => mark.label),
    nearStep: ticks(near).step,
    times: [formatTime(0), formatTime(75.26, true), formatTime(1200)],
  };
}

// --- lanes --------------------------------------------------------------------
{
  const layout = laneLayout(["a", "b", "listener"], { laneHeight: 22, top: 4 });
  const [a, , listener] = layout.rows;
  out.lanes = {
    height: layout.height,
    ys: layout.rows.map((row) => row.y),
    at: [layout.at(3), layout.at(4), layout.at(25.9), layout.at(26), layout.at(69.9), layout.at(70)].map((row) => (row ? row.id : null)),
    bandsInside: a.activity.y >= a.y && a.movement.y + a.movement.height <= a.y + a.height,
    bandsApart: a.activity.y + a.activity.height <= a.movement.y,
    listenerActivity: listener.activity,
    listenerBand: listener.movement.height,
  };
  const spans = [0, 10, 20, 30, 40].map((start) => ({ start_s: start, end_s: start + 10 }));
  out.visible = [visible(spans, 12, 31), visible(spans, 50.5, 60), visible(spans, -5, 0)].map((list) => list.map((s) => s.start_s));
  out.posture = [
    posture({ type: "dwell", height: "seated" }),
    posture({ type: "dwell", height: "standing" }),
    posture({ type: "travel" }),
    posture({ type: "rise", to: "standing" }),
    posture({ type: "rise", to: "seated" }),
    posture({ type: "walk", height: "standing" }),
  ];
}

// --- tracks: the server's samples, read between ----------------------------------
{
  const tracks = given.tracks;
  const worst = { atSample: 0, between: 0, yawAtSample: 0, listener: 0, head: 0 };
  const near = (a, b) => Math.hypot(a.x - b[0], a.y - b[1], a.z - b[2]);
  tracks.sources.forEach((track, s) => {
    tracks.t.forEach((t, i) => {
      const at = sourceAt(tracks, track, t);
      worst.atSample = Math.max(worst.atSample, near(at, [track.x[i], track.y[i], track.z[i]]));
      worst.yawAtSample = Math.max(worst.yawAtSample, Math.abs(at.yaw_deg - track.yaw_deg[i]));
    });
    given.between.t.forEach((t, i) => {
      worst.between = Math.max(worst.between, near(sourceAt(tracks, track, t), given.between.sources[s][i]));
    });
  });
  given.between.t.forEach((t, i) => {
    const at = listenerAt(tracks, t);
    worst.listener = Math.max(worst.listener, near(at, given.between.listener[i]));
    worst.head = Math.max(worst.head, Math.abs(at.yaw_deg - given.between.listener_yaw[i]));
  });
  const last = tracks.t.length - 1;
  const voice = tracks.sources.find((track) => track.activity.length > 1);
  const [first, second] = voice.activity;
  const points = trail(tracks, tracks.listener, 31.34, 4);
  const buffer = new Float32Array(3 * trailCapacity(4, tracks.step_s));
  const count = trail(tracks, tracks.listener, 31.34, 4, buffer);
  const end = listenerAt(tracks, 31.34);
  const start = listenerAt(tracks, 27.34);
  out.tracks = {
    worst,
    brackets: [bracket(tracks.t, tracks.step_s, -1), bracket(tracks.t, tracks.step_s, 1e9), bracket(tracks.t, tracks.step_s, tracks.t[3])],
    lastIndex: last,
    lastInterval: bracket(tracks.t, tracks.step_s, (tracks.t[last - 1] + tracks.t[last]) / 2),
    active: [
      activeAt(voice.activity, first[0]),
      activeAt(voice.activity, (first[0] + first[1]) / 2),
      activeAt(voice.activity, (first[1] + second[0]) / 2),
      activeAt(voice.activity, -1),
      activeAt([], 3),
    ],
    span: [spanAt(voice.movement, 0).start_s, spanAt(voice.movement, tracks.duration_s).end_s === tracks.duration_s],
    trailPoints: points.length / 3,
    trailCount: count,
    trailFits: count <= trailCapacity(4, tracks.step_s),
    trailEnds: [
      Math.hypot(points[0] - start.x, points[2] - start.z),
      Math.hypot(points[points.length - 3] - end.x, points[points.length - 1] - end.z),
    ],
    trailSame: points.every((value, i) => Math.abs(value - buffer[i]) < 1e-5),
    trailAtStart: trail(tracks, tracks.listener, 0, 4).length / 3,
  };
  // Forward in the viewport's frame is (-sin yaw, -cos yaw) in (x, z).
  const forward = (yawDeg) => [-Math.sin(viewYaw(yawDeg)), -Math.cos(viewYaw(yawDeg))];
  out.yaw = { view: [0, 90, 215].map(forward), plan: [0, 90, 215].map(heading) };
}

// --- the transport and the mix: what sound will attach to ---------------------------
{
  let clock = 100;
  const transport = createTransport({ now: () => clock });
  const said = [];
  for (const event of ["load", "play", "pause", "seek", "speed", "end"]) {
    transport.on(event, (state) => said.push([event, Number(state.t.toFixed(6)), state.speed, state.playing]));
  }
  let ticked = 0;
  const stop = transport.on("tick", () => ticked++);
  transport.load(60);
  transport.play();
  clock += 2;
  const afterTwo = transport.time();
  transport.setSpeed(4);
  clock += 1;
  const afterFour = transport.time();
  transport.pause();
  clock += 50;
  const paused = transport.time();
  transport.seek(58);
  transport.play();
  clock += 0.25;
  transport.setClock(() => 7);
  const carried = transport.time();
  transport.setClock(() => clock);
  clock += 5;
  transport.tick();
  const ended = transport.state();
  transport.play();
  const again = transport.time();
  transport.seek(-3);
  transport.setSpeed(16);
  stop();
  const before = ticked;
  transport.tick();
  out.transport = { said, afterTwo, afterFour, paused, carried, ended, again, speeds: SPEEDS, unsubscribed: ticked === before };

  const mix = createMix();
  const changes = [];
  mix.on("change", (state) => changes.push(state.audible.join(",")));
  mix.setSources(["a", "b", "c"]);
  mix.toggleMute("b");
  mix.toggleSolo("b");
  mix.toggleSolo("c");
  mix.toggleSolo("b");
  mix.toggleSolo("c");
  mix.toggleMute("nobody");
  mix.setSources(["a"]);
  out.mix = { changes, state: mix.state() };
}

// --- the generator's panel: units and limits -----------------------------------------
out.panel = {
  labels: ["turn_rate_deg_s", "speed_m_s", "seated_share", "count", "pitch_deg", "dwell_s"].map((key) => labelOf(key)),
  held: [
    held({ type: "number", min: 0.1, max: 1.5 }, 9),
    held({ type: "number", min: 0.1, max: 1.5 }, -1),
    held({ type: "integer", min: 0, max: 16 }, 3.6),
  ],
};

console.log(JSON.stringify(out));
