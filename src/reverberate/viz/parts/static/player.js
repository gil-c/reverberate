/** An ambisonic player: order 7 streams in, two ears out, the listener's own head.
 *
 * It plays the items of `api/items` (`reverberate.viz.parts.routes`): each a
 * signal in the scene's fixed frame, 64 channels, or two ears already
 * decoded. Several items of one length are **synchronised**: `select` goes
 * from one to another at the same instant, the two crossfaded over one block
 * of 10.7 ms under the same head, so that what changes is the render and
 * nothing else.
 *
 * Nothing is decoded here that the inspector does not decode: the audio
 * thread is its worklet (`viz/app/scene/sound.worklet.js`), the filters are
 * the measured head's (`viz/decoders.py`), turned by the head's matrix on
 * this thread (`sound-decode.js`, `rotateSpectra`). Two ears go through the
 * same thread with filters that change nothing, so that a switch, a start
 * and a stop are the same for both.
 *
 * The head is the scene's own, when one is given (`setSceneHead`), turned by
 * what the listener adds (`turn`): with nothing added, order 7 is heard as
 * the kit's files of two ears hold it.
 *
 *   const player = createPlayer();
 *   await player.load(items);         // rows of api/items, of one length
 *   player.play(); player.select(items[1].id); player.turn({ yaw: 30 });
 */
import { BLOCK, filterSpectra, headMatrix3, rotateSpectra } from "../reuse/scene/sound-decode.js";

//: Frames asked for at a time, and how far ahead of what is heard.
const CHUNK = 12000;
const AHEAD = 72000;
const RAD = Math.PI / 180;
//: The head is turned again when it has moved by this, degrees.
const HEAD_STEP_DEG = 0.25;

/** The head a step track holds at `seconds`, linear between steps. */
export function headAt(track, seconds) {
  if (!track || !track.yaw_deg.length) return { yaw: 0, pitch: 0, roll: 0 };
  const last = track.yaw_deg.length - 1;
  const at = Math.min(Math.max(seconds / track.step_s, 0), last);
  const k = Math.min(Math.floor(at), Math.max(last - 1, 0));
  const u = at - k;
  const mix = (values) => values[k] + (values[Math.min(k + 1, last)] - values[k]) * u;
  return { yaw: mix(track.yaw_deg), pitch: mix(track.pitch_deg), roll: mix(track.roll_deg || [0]) || 0 };
}

/** The gain of a level in dB. */
export const gainOf = (db) => 10 ** (db / 20);

export function createPlayer({ api = "api", decoders = "decoders", worklet = "reuse/scene/sound.worklet.js" } = {}) {
  const listeners = {};
  const emit = (name, detail) => (listeners[name] || []).forEach((fn) => fn(detail));

  let context = null;
  let gain = null;
  let node = null;
  let nodeChannels = 0;
  let decoder = null; // the measured head: { order, channels, taps, filters }
  let base = null; // the spectra of the filters the node listens with
  const spare = [];

  let items = [];
  let item = null;
  let balance = null;
  let rate = 48000;
  let total = 0;

  let playing = false;
  let gen = 0;
  let next = 0; // the next frame to ask of the stream of `gen`
  let flying = -1; // the generation a request is out for
  let awaited = 0; // the generation whose start the audio thread has not yet taken
  let position = 0; // the frame heard, as the audio thread last said
  let saidAt = 0;
  let sounding = false;
  let loop = null; // [first, stop) in frames, or null
  let looping = false;
  let loopHead = null; // the loop's first chunk, kept so that a turn of the loop does not wait
  let levelDb = 0;
  let sceneHead = null;
  let turned = { yaw: 0, pitch: 0 };
  let lastHead = [NaN, NaN, NaN];
  let failure = "";
  let peak = 0; // the largest sample the audio thread made in its last block, before the level
  let starved = 0; // how often the stream ran out of frames since it was loaded

  const framesUrl = (id, start, count) => {
    const query = new URLSearchParams({ item: id, start, count });
    if (balance !== null) query.set("balance", balance);
    return `${api}/frames?${query}`;
  };

  async function ensureAudio(channels) {
    if (!context) {
      context = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
      if (!context.audioWorklet) throw new Error("this page has no AudioWorklet: open it on localhost");
      await context.audioWorklet.addModule(worklet);
      gain = context.createGain();
      gain.connect(context.destination);
    }
    if (node && nodeChannels === channels) return;
    if (node) node.disconnect();
    if (channels === 2) {
      // Two ears as written: each its own channel through a filter of one tap.
      base = filterSpectra({ channels: 2, taps: 1, filters: new Float32Array([1, 0, 0, 1]) });
    } else {
      if (!decoder) {
        const records = await fetch(`${decoders}/decoders.json`).then((r) => r.json());
        if (!records.length) throw new Error("no measured head on this server: order 7 cannot be decoded");
        const record = records[0];
        const filters = new Float32Array(await fetch(`${decoders}/${record.url}`).then((r) => r.arrayBuffer()));
        decoder = { order: record.order, channels: record.channels, taps: record.taps, filters, head: record.head };
      }
      if (decoder.channels !== channels) throw new Error(`the head decodes ${decoder.channels} channels, the item has ${channels}`);
      base = filterSpectra(decoder);
    }
    node = new AudioWorkletNode(context, "scene-stream", {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      processorOptions: { channels },
    });
    nodeChannels = channels;
    node.port.onmessage = (event) => heard(event.data);
    node.onprocessorerror = () => fail("the audio thread stopped on an error");
    node.connect(gain);
    spare.length = 0;
    lastHead = [NaN, NaN, NaN];
    turnHead();
    setGain();
  }

  function fail(message) {
    failure = message;
    playing = false;
    emit("state", state());
  }

  function setGain() {
    if (gain && item) gain.gain.value = gainOf(levelDb + item.level_db);
  }

  /** The head now: the scene's at this instant, turned by what the listener added. */
  function pose() {
    const scene = headAt(sceneHead, time());
    return { yaw: scene.yaw + turned.yaw, pitch: Math.max(-89, Math.min(89, scene.pitch + turned.pitch)), roll: scene.roll };
  }

  function turnHead() {
    if (!node || !base) return;
    if (nodeChannels === 2) {
      if (Number.isNaN(lastHead[0])) {
        lastHead = [0, 0, 0];
        node.port.postMessage({ type: "filters", re: base.re.slice(), im: base.im.slice() });
      }
      return;
    }
    const { yaw, pitch, roll } = pose();
    const moved = Math.max(Math.abs(yaw - lastHead[0]), Math.abs(pitch - lastHead[1]), Math.abs(roll - lastHead[2]));
    if (moved < HEAD_STEP_DEG) return;
    lastHead = [yaw, pitch, roll];
    // The recipe's yaw is the ambisonic frame's; the inspector's function takes a camera's.
    const matrix = headMatrix3(yaw * RAD - Math.PI / 2, pitch * RAD, roll * RAD);
    const made = rotateSpectra(base, decoder.order, matrix, spare.pop() || null);
    node.port.postMessage({ type: "filters", re: made.re, im: made.im }, [made.re.buffer, made.im.buffer]);
  }

  function heard(message) {
    if (message.type === "spent") {
      spare.push({ re: message.re, im: message.im });
    } else if (message.type === "error") {
      fail(message.message);
    } else if (message.type === "started") {
      if (message.gen === awaited) awaited = 0;
    } else if (message.type === "state") {
      // Until the start is taken the thread still speaks of the stream it left.
      if (!playing || awaited) return;
      sounding = message.running && !message.waiting;
      position = Math.max(0, Math.min(message.position, total));
      saidAt = message.at;
      peak = Number.isFinite(message.peak) ? message.peak : 0;
      if (message.starved) starved += 1;
      if (Number.isFinite(message.peak) && item) {
        const over = message.peak * gainOf(levelDb + item.level_db);
        if (over > 1) emit("clip", { overDb: 20 * Math.log10(over) });
      }
      const end = looping && loop ? loop[1] : total;
      if (message.ended || position >= end) {
        if (looping) begin(loop ? loop[0] : 0);
        else {
          playing = false;
          position = 0;
          node.port.postMessage({ type: "stop" });
          emit("state", state());
        }
        return;
      }
      pump();
    }
  }

  /** Ask for what the stream lacks, a chunk at a time, in order. */
  function pump() {
    if (!playing || !item || flying === gen || next >= total || next - position > AHEAD) return;
    const mine = gen;
    const start = next;
    const count = Math.min(CHUNK, total - start);
    flying = mine;
    fetch(framesUrl(item.id, start, count))
      .then(async (response) => {
        if (!response.ok) throw new Error((await response.json()).error || `${response.status}`);
        return new Float32Array(await response.arrayBuffer());
      })
      .then((data) => {
        if (flying === mine) flying = -1;
        if (mine !== gen || !playing) return pump();
        if (looping && loop && start === loop[0]) loopHead = { key: headKey(), data: data.slice(), count };
        node.port.postMessage({ type: "chunk", gen: mine, start, frames: count, data, go: true }, [data.buffer]);
        next = start + count;
        pump();
      })
      .catch((error) => {
        if (flying === mine) flying = -1;
        if (mine === gen) fail(error.message);
      });
  }

  const headKey = () => `${item ? item.id : ""}|${balance}|${loop ? loop[0] : 0}`;

  /** A stream from `frame`: what sounded is faded out, what comes is faded in. */
  function begin(frame) {
    gen += 1;
    position = Math.max(0, Math.min(Math.round(frame), Math.max(total - 1, 0)));
    next = position;
    sounding = false;
    awaited = gen;
    node.port.postMessage({ type: "start", sample: position, gen, total });
    if (loopHead && loopHead.key === headKey() && loop && position === loop[0]) {
      const data = loopHead.data.slice();
      node.port.postMessage({ type: "chunk", gen, start: position, frames: loopHead.count, data, go: true }, [data.buffer]);
      next = position + loopHead.count;
    }
    pump();
  }

  /** The same instant of another stream: the audio thread crossfades to it when it has it. */
  function changeStream() {
    loopHead = null;
    if (!playing || !node) return;
    gen += 1;
    // From two blocks before what is heard: the first block's window reaches one back.
    next = Math.max(0, Math.round(position) - 2 * BLOCK);
    pump();
  }

  function time() {
    if (!playing || !context) return position / rate;
    const lead = sounding ? Math.min(Math.max(context.currentTime - saidAt, 0), 0.03) : 0;
    return Math.min(position / rate + lead, total / rate);
  }

  const state = () => ({
    playing,
    looping,
    loop: loop ? [loop[0] / rate, loop[1] / rate] : null,
    item: item ? item.id : null,
    duration: total / rate,
    levelDb,
    kind: item ? item.kind : null,
    turned: { ...turned },
    head: decoder ? decoder.head : null,
    failure,
  });

  function frame() {
    if (playing) {
      turnHead();
      emit("time", time());
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  const player = {
    on(name, fn) {
      (listeners[name] = listeners[name] || []).push(fn);
      return player;
    },
    /** The items switched between, rows of `api/items`: one length, one rate, one kind. */
    async load(rows, { keep = false } = {}) {
      if (!rows.length) throw new Error("nothing to play");
      const [first] = rows;
      for (const row of rows) {
        if (row.frames !== first.frames || row.rate !== first.rate || row.channels !== first.channels) {
          throw new Error(`${row.id} is not of the length of ${first.id}: they could not be switched`);
        }
      }
      const was = playing;
      if (playing && (!keep || first.channels !== nodeChannels)) player.stop();
      items = rows;
      rate = first.rate;
      total = first.frames;
      const same = keep && item ? rows.find((row) => row.label === item.label) : null;
      item = same || first;
      if (!keep) position = 0;
      if (loop) loop = [Math.min(loop[0], total), Math.min(loop[1], total)];
      failure = "";
      if (context) {
        try {
          await ensureAudio(item.channels);
        } catch (error) {
          return fail(error.message);
        }
      }
      setGain();
      if (was && keep && playing) changeStream();
      emit("state", state());
    },
    items: () => items,
    /** Hear another item of those loaded, from the same instant. */
    select(id) {
      const chosen = items.find((row) => row.id === id);
      if (!chosen || chosen === item) return;
      item = chosen;
      setGain();
      changeStream();
      emit("state", state());
    },
    /** Ask the item heard again from this instant: what its name stands for has changed. */
    refresh: () => changeStream(),
    /** The balance the frames are asked under: a version `api/balance` answered with. */
    setBalance(version) {
      if (version === balance) return;
      balance = version;
      changeStream();
    },
    async play() {
      if (playing || !item) return;
      failure = "";
      try {
        await ensureAudio(item.channels);
        await context.resume();
      } catch (error) {
        return fail(error.message);
      }
      playing = true;
      const from = looping && loop && (position < loop[0] || position >= loop[1]) ? loop[0] : position;
      begin(from >= total ? 0 : from);
      emit("state", state());
    },
    stop() {
      if (!playing) return;
      position = Math.round(time() * rate);
      playing = false;
      gen += 1;
      if (node) node.port.postMessage({ type: "stop" });
      emit("state", state());
      emit("time", time());
    },
    toggle: () => (playing ? player.stop() : player.play()),
    seek(seconds) {
      const frame = Math.max(0, Math.min(Math.round(seconds * rate), total));
      if (playing) begin(frame);
      else position = frame;
      emit("time", frame / rate);
    },
    /** The region a loop turns in, seconds; `null` for the whole item. */
    setLoop(region) {
      loop = region && region[1] - region[0] > 0.05 ? [Math.round(region[0] * rate), Math.round(region[1] * rate)] : null;
      loopHead = null;
      emit("state", state());
    },
    setLooping(on) {
      looping = Boolean(on);
      emit("state", state());
    },
    /** The level, dB over the page's default (at which a pack's full scale is 98 dB SPL). */
    setLevel(db) {
      levelDb = db;
      setGain();
      emit("state", state());
    },
    /** The scene's own head over the item: `{ step_s, yaw_deg, pitch_deg, roll_deg }`, or null. */
    setSceneHead(track) {
      sceneHead = track;
      turnHead();
    },
    /** What the listener adds to the scene's head, degrees; `yaw` is to his left. */
    turn({ yaw = turned.yaw, pitch = turned.pitch } = {}) {
      turned = { yaw: ((((yaw + 180) % 360) + 360) % 360) - 180, pitch: Math.max(-80, Math.min(80, pitch)) };
      turnHead();
      emit("head", { ...turned });
    },
    /** What the audio thread last made: its block's peak at the level heard, dB re full scale. */
    heard: () => ({ peakDb: 20 * Math.log10(Math.max(peak * (item ? gainOf(levelDb + item.level_db) : 1), 1e-9)), starved }),
    /** The node the two ears leave by, once something has played: for a meter or a recorder. */
    output: () => gain,
    time,
    state,
    pose,
  };
  return player;
}
