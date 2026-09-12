/** Follows the listener: which cell, which head, and asks the workers for the
 * response, then hands it to the engine and the plots.
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
import { headMatrix } from "./sh.js";
import { decay, joined, spectrogram } from "./analysis.js";

//: Fastest the early part is re-rendered while the listener turns, seconds.
const MIN_INTERVAL_S = 0.06;

/** Adjustable from the page. `earlyMs` is where a whole response is split;
 *  `stillMs` how long the head must be still before the late part is
 *  re-decoded for it; `fadeMs` the crossfade between the two parts. */
export const settings = { earlyMs: 150, stillMs: 200, fadeMs: 10 };

export function createSpatial({ engine, workerUrl, onRendered, onStatus }) {
  const early = new Worker(workerUrl, { type: "module" });
  const fields = new Map();
  const pendingEarly = new Map(); // request id -> source id
  let nextId = 1;
  let decoderMessage = null;
  const stats = { lateRenders: 0, lateCancelled: 0, lateMsTotal: 0, lateStill: 0 };

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
    const late = state.lastLate;
    const whole = late ? joined(state.lastEarly[0], late[0], state.tailStart) : state.lastEarly[0];
    onRendered(id, {
      ms: state.earlyMs,
      lateMs: state.lateMs,
      early: state.lastEarly,
      spectrogram: spectrogram(whole),
      decay: decay(whole),
      seconds: whole.length / sampleRate(),
    });
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
      stats.lateCancelled += 1;
      if (state.lateInFlight === message.id) state.lateInFlight = null;
      return;
    }
    if (state.lateInFlight === message.id) state.lateInFlight = null;
    if (message.missing || message.key !== state.lateKey) return;
    stats.lateRenders += 1;
    stats.lateMsTotal += message.ms;
    state.lastLate = message.brir;
    state.lateMs = message.ms;
    state.lateHead = state.lateHeadRequested;
    engine.setTail(state.id, message.brir, state.tailStart);
    report(state.id, state);
    onStatus(state.id, { late: "exact" });
  }

  async function keepEarly(id, state, position) {
    const key = `${id}:${position}:${earlySamples()}`;
    if (state.keptEarly.has(key)) return key;
    const channels = await state.field.cell(position);
    const cut = Math.min(channels[0].length, earlySamples());
    // Copies, so the chunk's own buffer is not detached by the transfer.
    early.postMessage({ type: "cell", key, channels: channels.map((c) => Float32Array.from(c.subarray(0, cut))) });
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
    if (state.field.index.samples <= earlySamples() + fadeSamples()) return null;
    const key = `${id}:${position}:late:${start}`;
    if (!state.keptLate.has(key)) {
      const channels = await state.field.cell(position);
      lateWorker(state).postMessage({
        type: "keep",
        key,
        channels: channels.map((c) => Float32Array.from(c.subarray(start))),
        crossfade: fadeSamples(),
        limit: 3,
      });
      state.keptLate.add(key);
      if (state.keptLate.size > 3) state.keptLate.delete(state.keptLate.values().next().value);
    }
    return { key, start, fade: fadeSamples() };
  }

  function renderLate(state, head, still) {
    if (!state.lateKey) return;
    if (state.lateInFlight !== null) {
      lateWorker(state).postMessage({ type: "cancel", id: state.lateInFlight });
      state.lateInFlight = null;
    }
    const id = nextId++;
    state.lateInFlight = id;
    state.lateHeadRequested = head;
    if (still) stats.lateStill += 1;
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
      renderLate(state, head, true);
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
    const position = state.field.cellAt(pose.x, pose.y, pose.z);
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
      const cellKey = await keepEarly(id, state, position);
      const late = await lateFor(id, state, position);
      let fadeStart = 0;
      let crossfade = 0;
      if (late) {
        fadeStart = late.start;
        crossfade = late.fade;
        state.tailStart = late.start;
        if (late.key !== state.lateKey) {
          // A new cell or room: its late part at this head, now; never
          // cancelled by movement, only superseded by a newer cell.
          state.lateKey = late.key;
          state.lastLate = null;
          state.lateHead = null;
          state.lateStillPending = false;
          renderLate(state, head, false);
        }
      } else {
        state.lateKey = null;
      }
      state.cell = position;
      const requestId = nextId++;
      pendingEarly.set(requestId, id);
      early.postMessage({ type: "render", id: requestId, cell: cellKey, head, fadeStart, crossfade });
      moved(state, pose);
      onStatus(id, { cell: position, ...state.field.describe(position), late: "entry" });
      for (const [dx, dz] of [[0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]]) {
        state.field.prefetch(state.field.cellAt(pose.x + dx, pose.y, pose.z + dz));
      }
    } catch (error) {
      state.pendingEarly = false;
      onStatus(id, { error: error.message });
    }
  }

  return {
    setDecoder,
    settings,
    stats,
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
      for (const id of ids) request(id, pose);
    },
    /** The field of a source, for the page's own use of its lattice. */
    fieldOf: (id) => (fields.get(id) || {}).field || null,
  };
}
