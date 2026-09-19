/** Follows the listener: which cell, which head, and asks the workers for the
 * response, then hands it to the engine and the plots.
 *
 * **What a cell's response is, here, is the response with its own propagation
 * stripped off.** The lattice is forty centimetres wide, so the time the
 * direct sound takes to arrive steps by up to a millisecond and a half from
 * one cell to the next; left in the response, that step is a jump in the
 * waveform and, worse, a millisecond of comb filtering for the whole of the
 * crossfade that follows it. So every cell is cut at its own direct arrival,
 * `direct_path_m / c`, and the time itself goes to the engine's delay line,
 * which follows the listener's real distance to the source continuously.
 * What is left in the response is the room, which neighbouring cells agree
 * about far better than they agree about when the sound arrives.
 *
 * Two stages, so a head turn never waits on a long decode:
 *
 * 1. The first `earlyMs` of a cell's response are decoded on every head
 *    update, at most every `MIN_INTERVAL_S`, in the early worker.
 * 2. The rest is decoded in a worker of its own per source: once when the
 *    cell is entered, at the head of that moment,
 *    and again, exactly, once the head has been still for `stillMs`. A
 *    stillness decode is cancelled if the head moves before it finishes, so
 *    the late part is never an orientation the listener has already left.
 *    At rest the sum of the two parts is the exact decode of the whole
 *    response; while turning, only the part after `earlyMs` lags.
 *
 * One request in flight per source and stage; a pose that arrives while one
 * is pending replaces the pending pose, so the newest is always next.
 */
import { decoderLead, solverLatency } from "./decode.js";
import { headMatrix } from "./sh.js";

//: Fastest the early part is re-rendered while the listener turns, seconds.
//: The engine's crossfade is longer than this on purpose, so one response is
//: always fading into the next and the listener never hears a plateau.
export const MIN_INTERVAL_S = 0.04;
//: Metres per second in air, as the field's own producer uses.
export const SOUND_SPEED_M_S = 343.2;
//: How far a listener may be from a cell and still be given it, and how far
//: they must get before they are given nothing at all.
//:
//: The flat's air is holey -- the solver wrote no cell inside the furniture
//: or against the walls, and on `hssd_0076` there are more empty lattice
//: points than full ones -- while the picture lets the listener stand
//: anywhere a room contains them. Walking a room therefore crosses places
//: with no cell underneath, and cutting the sound at each of them is what a
//: walk was heard to crackle on: a fade out and a fade in every tenth of a
//: second as the listener grazes the edge of the solved air. Being given the
//: room from a metre away is wrong; being given silence is more wrong, and
//: unlike the first it is audible. The two distances differ so that grazing
//: the boundary cannot chatter across it.
export const REACH_M = 1.1;
export const HOLD_M = 2.2;
//: How far the listener walks before the late part is fetched from a nearer
//: cell.
//:
//: Giving a `ConvolverNode` a buffer clears it, so every swap of the late
//: part throws away a second of reverberation that was sounding and starts
//: building the next one from nothing. Swapping it at every cell means doing
//: that five times a second at walking pace: the reverberation never
//: finishes building, and what is heard is a flutter at the rate of the
//: crossing, which was measured as the loudest thing left in a walk. The
//: late field is diffuse -- it is the part ADR 0013 says has no direction to
//: lag by -- and within a room it hardly changes over a couple of metres, so
//: it is fetched every two instead, and again whenever the room changes. The
//: early part, which carries the direction and the direct sound, still
//: follows every cell.
const TAIL_STEP_M = 2.0;
//: And no sooner than this, whatever the distance. Two convolvers share the
//: late part and its crossfade is a quarter of a second, so one of them has
//: to be silent again before the next arrives.
const TAIL_INTERVAL_S = 0.7;

//: Where the cells about to be needed are fetched from ahead of time: half a
//: metre off in each direction, a step and a bit of the usual lattice.
const PREFETCH_AT = [
  [0.5, 0],
  [-0.5, 0],
  [0, 0.5],
  [0, -0.5],
];

/** Adjustable from the page. `earlyMs` is where a whole response is split;
 *  `stillMs` how long the head must be still before the late part is
 *  re-decoded for it; `fadeMs` the crossfade between the two parts. */
export const settings = { earlyMs: 150, stillMs: 200, fadeMs: 10 };

/** Seconds before the late part is to be fetched again, from `position`,
 *  for a listener at `pose` at `now` seconds: zero for now, Infinity for not
 *  until the listener moves on. `tail` is where and when it was last
 *  fetched, `{ position, at, fetchedAt }`, and `rooms` the field's own. */
export function tailWait(tail, pose, position, rooms, now) {
  if (tail.position === null) return 0;
  const room = (cell) => (rooms ? rooms[cell] : null);
  const far =
    Math.hypot(pose.x - tail.at.x, pose.y - tail.at.y, pose.z - tail.at.z) > TAIL_STEP_M ||
    // A doorway is narrower than a metre, and the reverberation either side
    // of one is not the same reverberation.
    room(position) !== room(tail.position);
  return far ? Math.max(0, tail.fetchedAt + TAIL_INTERVAL_S - now) : Infinity;
}

/** Samples stripped from the front of a cell's response: the time the sound
 *  takes to get there, which the engine's delay line carries instead, and
 *  the solver's own latency, which nothing needs. */
export const arrivalSamples = (directPathM, sampleRate, latency) =>
  Math.round((directPathM / SOUND_SPEED_M_S) * sampleRate) + latency;

/** The solver's own latency in a field, samples, measured once on the cell
 *  nearest the source, where the direct sound is clearest. */
export async function latencyOf(field, sampleRate) {
  const paths = field.index.direct_path_m;
  let nearest = 0;
  for (let position = 1; position < paths.length; position++) {
    if (paths[position] < paths[nearest]) nearest = position;
  }
  const channels = await field.cell(nearest);
  return solverLatency(channels[0], (paths[nearest] / SOUND_SPEED_M_S) * sampleRate);
}

export function createSpatial({ engine, workerUrl, onRendered, onStatus, onCell }) {
  const early = new Worker(workerUrl, { type: "module" });
  const fields = new Map();
  const pendingEarly = new Map(); // request id -> source id
  let nextId = 1;
  let decoderMessage = null;
  let lead = 0;

  const sampleRate = () => engine.sampleRate;
  const earlySamples = () => Math.round((settings.earlyMs / 1000) * sampleRate());
  const fadeSamples = () => Math.round((settings.fadeMs / 1000) * sampleRate());

  early.onmessage = (event) => {
    const message = event.data;
    if (message.type !== "brir") return;
    const id = pendingEarly.get(message.id);
    pendingEarly.delete(message.id);
    const state = fields.get(id);
    if (!state) return;
    state.pendingEarly = false;
    if (!message.missing) {
      state.lastEarly = message.early;
      engine.setEarly(id, message.early);
      state.earlyMs = message.ms;
      report(id, state);
    }
    if (state.wanted) {
      const pose = state.wanted;
      state.wanted = null;
      request(id, pose);
    }
  };

  function report(id, state) {
    if (!state.lastEarly) return;
    onRendered(id, { ms: state.earlyMs, lateMs: state.lateMs, early: state.lastEarly });
  }

  function lateWorker(state) {
    if (!state.late) {
      state.late = new Worker(workerUrl, { type: "module" });
      if (decoderMessage) state.late.postMessage(copyDecoder());
      state.late.onmessage = (event) => onLate(state, event.data);
    }
    return state.late;
  }

  const copyDecoder = () => ({ ...decoderMessage, filters: Float32Array.from(decoderMessage.filters) });

  function setDecoder(decoder) {
    // What the early worker will cut from the front of every early part, so
    // the late part can be placed that much earlier to meet it.
    lead = decoderLead(decoder.filters, decoder.channels, decoder.taps);
    decoderMessage = {
      type: "decoder",
      order: decoder.order,
      channels: decoder.channels,
      taps: decoder.taps,
      filters: decoder.filters,
    };
    early.postMessage(copyDecoder());
    for (const state of fields.values()) {
      if (state.late) state.late.postMessage(copyDecoder());
      state.keptEarly.clear();
      state.keptLate.clear();
      state.lateKey = null;
    }
  }

  function onLate(state, message) {
    if (message.type !== "late") return;
    if (message.cancelled) {
      if (state.lateInFlight === message.id) state.lateInFlight = null;
      return;
    }
    if (state.lateInFlight === message.id) state.lateInFlight = null;
    if (message.missing || message.key !== state.lateKey) return;
    state.lastLate = message.brir;
    state.lateMs = message.ms;
    state.lateHead = state.lateHeadRequested;
    engine.setTail(state.id, message.brir, state.tailStart - lead);
    report(state.id, state);
    onStatus(state.id, { late: "exact" });
  }

  const arrivalOf = (state, position) =>
    arrivalSamples(state.field.index.direct_path_m[position], sampleRate(), state.latency);

  async function keepEarly(id, state, position) {
    const arrival = arrivalOf(state, position);
    const key = `${id}:${position}:${earlySamples()}:${fadeSamples()}:${arrival}`;
    if (state.keptEarly.has(key)) return key;
    const channels = await state.field.cell(position);
    const cut = Math.min(channels[0].length - arrival, earlySamples());
    // Copies, so the chunk's own buffer is not detached by the transfer.
    // Faded out on the field where the late part fades in, when there is
    // one (`lateFor`); see `decode.js`.
    const split = state.field.index.samples > arrival + earlySamples() + fadeSamples();
    early.postMessage({
      type: "cell",
      key,
      channels: channels.map((c) => Float32Array.from(c.subarray(arrival, arrival + cut))),
      fadeOutFrom: earlySamples() - fadeSamples(),
      fadeOut: split ? fadeSamples() : 0,
    });
    state.keptEarly.add(key);
    if (state.keptEarly.size > 64) {
      const oldest = state.keptEarly.values().next().value;
      state.keptEarly.delete(oldest);
      early.postMessage({ type: "forget", key: oldest });
    }
    return key;
  }

  /** The rest of a cell's response after the split, kept in its worker. */
  async function lateFor(id, state, position) {
    const start = earlySamples() - fadeSamples();
    const arrival = arrivalOf(state, position);
    if (state.field.index.samples <= arrival + earlySamples() + fadeSamples()) return null;
    const key = `${id}:${position}:late:${start}:${arrival}`;
    if (!state.keptLate.has(key)) {
      const channels = await state.field.cell(position);
      lateWorker(state).postMessage({
        type: "keep",
        key,
        channels: channels.map((c) => Float32Array.from(c.subarray(start + arrival))),
        fadeIn: fadeSamples(),
        limit: 3,
      });
      state.keptLate.add(key);
      if (state.keptLate.size > 3) state.keptLate.delete(state.keptLate.values().next().value);
    }
    return { key, start, fade: fadeSamples() };
  }

  function renderLate(state, head) {
    if (!state.lateKey) return;
    if (state.lateInFlight !== null) {
      lateWorker(state).postMessage({ type: "cancel", id: state.lateInFlight });
      state.lateInFlight = null;
    }
    const id = nextId++;
    state.lateInFlight = id;
    state.lateHeadRequested = head;
    lateWorker(state).postMessage({ type: "render", id, key: state.lateKey, head });
  }

  /** The head moved: cancel a stillness decode in flight, and arm the timer. */
  function moved(state, pose) {
    state.lastPose = pose;
    if (state.lateInFlight !== null && state.lateStillPending) {
      lateWorker(state).postMessage({ type: "cancel", id: state.lateInFlight });
      state.lateInFlight = null;
      state.lateStillPending = false;
    }
    clearTimeout(state.stillTimer);
    state.stillTimer = setTimeout(() => {
      const head = headMatrix(state.lastPose.yaw, state.lastPose.pitch);
      if (!state.lateKey || sameHead(head, state.lateHead)) return;
      state.lateStillPending = true;
      renderLate(state, head);
    }, settings.stillMs);
  }

  const sameHead = (a, b) => Boolean(a && b) && a.every((v, i) => Math.abs(v - b[i]) < 1e-9);

  async function request(id, pose) {
    const state = fields.get(id);
    if (!state || !decoderMessage) return;
    const now = performance.now() / 1000;
    if (state.pendingEarly || now - state.lastAt < MIN_INTERVAL_S) {
      state.wanted = pose;
      if (!state.pendingEarly) {
        setTimeout(() => {
          if (state.wanted && !state.pendingEarly) {
            const next = state.wanted;
            state.wanted = null;
            request(id, next);
          }
        }, (MIN_INTERVAL_S - (now - state.lastAt)) * 1000 + 1);
      }
      return;
    }
    const position = state.field.cellAt(
      pose.x,
      pose.y,
      pose.z,
      state.cell === null ? REACH_M : HOLD_M
    );
    if (position === null) {
      if (state.cell !== null) {
        state.cell = null;
        state.lateKey = null;
        engine.silence(id);
        onStatus(id, { cell: null });
      }
      return;
    }
    state.pendingEarly = true;
    state.lastAt = now;
    const head = headMatrix(pose.yaw, pose.pitch);
    try {
      if (state.latency === null) state.latency = await state.latencyKnown;
      // The propagation the responses no longer carry, from where the
      // listener really is rather than from the cell they stand in: it is
      // the one part of the render that can follow them continuously.
      const to = state.field.index.source_position;
      const distance = Math.hypot(pose.x - to[0], pose.y - to[1], pose.z - to[2]);
      engine.setPropagation(id, distance / SOUND_SPEED_M_S, pose.at);
      const cellKey = await keepEarly(id, state, position);
      // The late part follows at its own pace: every two metres, or a new room.
      const wait = tailWait(state.tail, pose, position, state.field.index.rooms, now);
      clearTimeout(state.tailTimer);
      if (wait === 0) {
        state.tail = { position, at: { x: pose.x, y: pose.y, z: pose.z }, fetchedAt: now };
      } else if (Number.isFinite(wait)) {
        // Due, but too soon after the last: come back for it then, since a
        // listener who steps through a door and stops sends no more poses.
        state.tailTimer = setTimeout(() => request(id, { ...pose, at: performance.now() / 1000 }), wait * 1000 + 1);
      }
      const late = await lateFor(id, state, state.tail.position);
      if (late) {
        state.tailStart = late.start;
        if (late.key !== state.lateKey) {
          // A new late cell or room: its part at this head, now; never
          // cancelled by movement, only superseded by a newer one.
          state.lateKey = late.key;
          state.lastLate = null;
          state.lateHead = null;
          state.lateStillPending = false;
          renderLate(state, head);
        }
      } else {
        state.lateKey = null;
      }
      if (position !== state.cell) onCell(id, position);
      state.cell = position;
      const requestId = nextId++;
      pendingEarly.set(requestId, id);
      early.postMessage({ type: "render", id: requestId, cell: cellKey, head });
      moved(state, pose);
      onStatus(id, { cell: position, ...state.field.describe(position), late: "entry" });
      for (const [dx, dz] of PREFETCH_AT) {
        state.field.prefetch(state.field.cellAt(pose.x + dx, pose.y, pose.z + dz, REACH_M));
      }
    } catch (error) {
      state.pendingEarly = false;
      onStatus(id, { error: error.message });
    }
  }

  return {
    setDecoder,
    settings,
    /** Change a setting; the split moves, so every kept part is dropped. */
    setSettings(changes) {
      Object.assign(settings, changes);
      for (const state of fields.values()) {
        state.keptEarly.clear();
        state.keptLate.clear();
        state.lateKey = null;
        state.lateHead = null;
      }
    },
    setField(id, field) {
      const previous = fields.get(id);
      if (previous) {
        clearTimeout(previous.stillTimer);
        clearTimeout(previous.tailTimer);
        if (previous.late) previous.late.terminate();
      }
      if (!field) {
        fields.delete(id);
        engine.silence(id);
        return;
      }
      fields.set(id, {
        id,
        field,
        latency: null,
        latencyKnown: latencyOf(field, sampleRate()).catch(() => 0),
        cell: null,
        keptEarly: new Set(),
        keptLate: new Set(),
        late: null,
        lateKey: null,
        lateInFlight: null,
        lateStillPending: false,
        lateHead: null,
        lateHeadRequested: null,
        lastEarly: null,
        lastLate: null,
        tailStart: 0,
        tail: { position: null, at: null, fetchedAt: -Infinity },
        tailTimer: null,
        earlyMs: null,
        lateMs: null,
        pendingEarly: false,
        wanted: null,
        lastAt: -Infinity,
        lastPose: null,
        stillTimer: null,
      });
    },
    update(pose, ids) {
      // When the pose was taken, not when it is used: the scheduler holds a
      // pose while a decode is in flight, and the delay line needs to know
      // how old the one it is given is. See `engine.setPropagation`.
      const stamped = { ...pose, at: performance.now() / 1000 };
      for (const id of ids) request(id, stamped);
    },
    /** The field of a source, for the page's own use of its lattice. */
    fieldOf: (id) => (fields.get(id) || {}).field || null,
  };
}
