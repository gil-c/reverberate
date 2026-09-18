/** A walk through a field, rendered offline: the scheduler of
 * `audio/spatial.js` and the convolutions of `audio/engine.js`, on a virtual
 * clock, reported as the filter the listener hears at every instant.
 *
 * The decode is the page's own (`audio/decode.js`) and is really run, so what
 * a response costs is measured rather than assumed. The rules that decide
 * which cell, when the late part moves, how much is cut from a response, are
 * imported from `spatial.js`, and the settings from `engine.js` and
 * `spatial.js`, so what is measured with no variant is what the page ships.
 * The one thing modelled rather than shared is the schedule of the workers:
 * a request is issued, the worker is busy for as long as a decode costs, and
 * the response takes over then.
 */
import { createDecoder } from "../../src/reverberate/viz/app/audio/decode.js";
import { EARLY_SLOTS, FADE_S, TAIL_FADE_S, TAIL_SLOTS } from "../../src/reverberate/viz/app/audio/engine.js";
import { headMatrix } from "../../src/reverberate/viz/app/audio/sh.js";
import {
  arrivalSamples,
  latencyOf,
  MIN_INTERVAL_S,
  settings as SPLIT,
  tailMoves,
} from "../../src/reverberate/viz/app/audio/spatial.js";
import { createPair } from "./graph.mjs";

const SOUND_SPEED_M_S = 343.2;
const DEGREES = 180 / Math.PI;

/** The page as it ships. A variant overrides any of these. */
export const SHIPPED = {
  ...SPLIT,
  minIntervalS: MIN_INTERVAL_S,
  engineFadeS: FADE_S,
  convolvers: EARLY_SLOTS,
  tailFadeS: TAIL_FADE_S,
  //: Cut each cell at its own direct arrival and carry the propagation in a
  //: delay line that follows the listener.
  alignDirect: true,
  //: The early part on the page's own convolver, whose responses have the
  //: input's whole past, rather than on `ConvolverNode`s.
  worklet: true,
  //: The late part fetched again at every cell, as it once was, rather than
  //: by `spatial.tailMoves`.
  tailEveryCell: false,
  //: How often the page looks at the pose: one animation frame.
  frameHz: 60,
  //: How many decodes are timed before the walk to fix what one costs, so
  //: two variants measured an hour apart still compare.
  calibrate: 7,
  //: How often the filter is looked at, and from when.
  traceHopS: 0.01,
  traceSkipS: 0.5,
};

/** A pose at every animation frame, from a described move. */
export function trajectory({ seconds, frameHz, from, walk = [0, 0], turnRate = 0, y }) {
  const poses = [];
  const count = Math.round(seconds * frameHz);
  for (let frame = 0; frame <= count; frame++) {
    const t = frame / frameHz;
    poses.push({ t, x: from.x + walk[0] * t, y, z: from.z + walk[1] * t, yaw: from.yaw + turnRate * t, pitch: 0 });
  }
  return poses;
}

const timed = (work) => {
  const started = performance.now();
  work();
  return performance.now() - started;
};

/** Run one variant over one trajectory: the filter at every hop, how far
 *  the head was ahead of it, how many responses took over, what one cost. */
export async function renderWalk({ field, decoderFilters, poses, sampleRate, variant = {} }) {
  const settings = { ...SHIPPED, ...variant };
  // Two workers in the page, so two decoders here: what one keeps, and what
  // it drops to stay under its cap, must not touch the other's.
  const decoder = createDecoder();
  const lateDecoder = createDecoder();
  decoder.setDecoder(decoderFilters);
  lateDecoder.setDecoder(decoderFilters);

  const earlySamples = Math.round((settings.earlyMs / 1000) * sampleRate);
  const fadeSamples = Math.round((settings.fadeMs / 1000) * sampleRate);
  const lateStart = earlySamples - fadeSamples;
  const lead = decoder.lead;
  const latency = settings.alignDirect ? await latencyOf(field, sampleRate) : 0;
  const shiftOf = (position) =>
    settings.alignDirect ? arrivalSamples(field.index.direct_path_m[position], sampleRate, latency) : 0;

  const early = createPair({ sampleRate, fade: settings.engineFadeS, count: settings.convolvers, hot: settings.worklet });
  const tail = createPair({ sampleRate, fade: settings.tailFadeS, count: TAIL_SLOTS });
  const propagation = []; // { at, seconds }, one per early response

  const kept = new Set();
  async function keepEarly(position) {
    const key = `e:${position}`;
    if (!kept.has(key)) {
      const channels = await field.cell(position);
      const shift = shiftOf(position);
      decoder.keep({
        key,
        channels: channels.map((c) => Float32Array.from(c.subarray(shift, shift + earlySamples))),
        fadeOutFrom: lateStart,
        fadeOut: fadeSamples,
      });
      kept.add(key);
    }
    return key;
  }
  async function keepLate(position) {
    const key = `l:${position}`;
    if (!kept.has(key)) {
      const channels = await field.cell(position);
      const start = lateStart + shiftOf(position);
      lateDecoder.keep({ key, channels: channels.map((c) => Float32Array.from(c.subarray(start))), fadeIn: fadeSamples, limit: 3 });
      kept.add(key);
    }
    return key;
  }

  // --- the scheduler, as `spatial.js` runs it ------------------------------
  let earlyCost = null;
  let lateCost = null;
  let lastIssue = -Infinity;
  let earlyFreeAt = 0;
  let lateFreeAt = 0;
  let lateKey = null;
  let lateHead = null;
  let cell = null;
  let stillFrom = 0;
  let lastPose = null;
  let tailState = { position: null, at: null, fetchedAt: -Infinity };

  const sameHead = (a, b) => Boolean(a && b) && a.every((v, i) => Math.abs(v - b[i]) < 1e-9);
  const moved = (a, b) => !a || ["x", "z", "yaw", "pitch"].some((key) => Math.abs(a[key] - b[key]) > 1e-9);

  function renderLate(now, key, head) {
    const plan = lateDecoder.lateSteps({ key, head });
    let ms = 0;
    for (const step of plan.steps) ms += timed(step);
    if (lateCost === null) lateCost = ms;
    const at = Math.max(now, lateFreeAt) + lateCost / 1000;
    lateFreeAt = at;
    tail.swap(at, plan.ears);
    lateHead = head;
  }

  for (const pose of poses) {
    const now = pose.t;
    if (moved(lastPose, pose)) stillFrom = now;
    lastPose = pose;
    // The page holds the pose and comes back; nothing is issued here.
    if (now - lastIssue < settings.minIntervalS || now < earlyFreeAt) continue;
    const position = field.cellAt(pose.x, pose.y, pose.z);
    if (position === null) continue;
    lastIssue = now;
    const key = await keepEarly(position);
    const head = headMatrix(pose.yaw, pose.pitch);
    if (earlyCost === null) {
      // Timed on the first cell over yaw and pitch alike: a pure yaw is three
      // times cheaper, and timing it alone would flatter the schedule.
      const times = [];
      for (let i = 0; i < settings.calibrate; i++) {
        times.push(timed(() => decoder.renderEarly({ key, head: headMatrix((i * Math.PI) / 3, i % 2 ? 0 : 0.2) })));
      }
      earlyCost = times.sort((a, b) => a - b)[times.length >> 1];
    }
    const at = Math.max(now, earlyFreeAt) + earlyCost / 1000;
    earlyFreeAt = at;
    early.swap(at, decoder.renderEarly({ key, head }), pose.yaw);
    if (settings.alignDirect) {
      const s = field.index.source_position;
      propagation.push({ at: now, seconds: Math.hypot(pose.x - s[0], pose.y - s[1], pose.z - s[2]) / SOUND_SPEED_M_S });
    }
    // The late part: by the distance walked, and again once the head is still.
    const moves = settings.tailEveryCell
      ? position !== cell
      : tailMoves(tailState, pose, position, field.index.rooms, now);
    cell = position;
    if (moves) {
      tailState = { position, at: { x: pose.x, y: pose.y, z: pose.z }, fetchedAt: now };
      lateKey = await keepLate(position);
      renderLate(at, lateKey, head);
    } else if (lateKey && now - stillFrom >= settings.stillMs / 1000 && !sameHead(head, lateHead)) {
      renderLate(at, lateKey, head);
    }
  }

  // --- the filter at every hop ----------------------------------------------
  // The early part and the tail where the tail sits, with the delay line in
  // front of them reported rather than applied; and half a second of the
  // tail as energy alone, which is where swapping it shows.
  const traceLength = Math.min(earlySamples + 2 * fadeSamples, 8192);
  const reverbLength = Math.round(0.5 * sampleRate);
  const delayAt = (t) => {
    let i = 0;
    while (i < propagation.length - 1 && propagation[i + 1].at <= t) i++;
    const a = propagation[i];
    const b = propagation[i + 1];
    const seconds = b && t > a.at ? a.seconds + ((b.seconds - a.seconds) * (t - a.at)) / (b.at - a.at) : a.seconds;
    return seconds * sampleRate;
  };
  const trace = [];
  const lastT = poses[poses.length - 1].t;
  for (let t = settings.traceSkipS; t < lastT; t += settings.traceHopS) {
    const ears = early.effective(t, traceLength, [new Float64Array(traceLength), new Float64Array(traceLength)]);
    const tailEars = tail.effective(t, reverbLength, [new Float64Array(reverbLength), new Float64Array(reverbLength)]);
    let reverb = 0;
    for (let ear = 0; ear < 2; ear++) {
      for (let i = 0; i < reverbLength; i++) reverb += tailEars[ear][i] ** 2;
      for (let i = 0; i + lateStart - lead < traceLength; i++) ears[ear][i + lateStart - lead] += tailEars[ear][i];
    }
    trace.push({
      ears,
      delaySamples: propagation.length ? delayAt(t) : 0,
      reverbDb: 10 * Math.log10(Math.max(reverb, 1e-30)),
    });
  }

  // How far the head has turned past the filter it is hearing: the price of
  // a long crossfade, and the one number a long one makes worse.
  const lagDegrees = [];
  for (const pose of poses) {
    if (pose.t < settings.traceSkipS) continue;
    const yaw = early.weighted(pose.t);
    if (yaw !== null) lagDegrees.push(Math.abs(pose.yaw - yaw) * DEGREES);
  }
  return { trace, lagDegrees, swaps: early.swaps(), costMs: earlyCost };
}

/** The lattice on its own: the nearest cell, decoded at the true pose at
 *  every frame, with no schedule and no crossfade. Turning, it is what an
 *  engine with no latency would give; walking, it is no floor, because a
 *  crossfade spanning two cells averages them and the page is smoother. */
export async function renderFloor({ field, decoderFilters, poses, sampleRate }) {
  const decoder = createDecoder();
  decoder.setDecoder(decoderFilters);
  const earlySamples = Math.round((SPLIT.earlyMs / 1000) * sampleRate);
  const latency = await latencyOf(field, sampleRate);
  const source = field.index.source_position;
  const trace = [];
  for (const pose of poses) {
    if (pose.t < SHIPPED.traceSkipS) continue;
    const position = field.cellAt(pose.x, pose.y, pose.z);
    const key = String(position);
    const shift = arrivalSamples(field.index.direct_path_m[position], sampleRate, latency);
    const channels = await field.cell(position);
    decoder.keep({ key, channels: channels.map((c) => Float32Array.from(c.subarray(shift, shift + earlySamples))), limit: 4 });
    const distance = Math.hypot(pose.x - source[0], pose.y - source[1], pose.z - source[2]);
    trace.push({
      ears: decoder.renderEarly({ key, head: headMatrix(pose.yaw, pose.pitch) }),
      delaySamples: (distance / SOUND_SPEED_M_S) * sampleRate,
    });
  }
  return { trace };
}
