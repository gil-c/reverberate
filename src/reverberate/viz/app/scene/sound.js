/** A scene heard: the signal engine's own output, at the listener's place, through headphones.
 *
 * The scene view replays a recipe without sound. Once the recipe is traced
 * into a pack the server renders it with the engine training will use and
 * keeps the stems (`viz/audit_stems.py`); this stage plays them. It asks for
 * the mix of the sources that are heard, chunk by chunk, as raw order 7
 * frames, and hands them to the audio thread, which turns them by the head
 * and decodes them to two ears (`sound-decode.js`) with the decoder the page
 * already listens to a solver's field with. The page renders nothing: the
 * bytes it receives are the engine's, and `verify` proves it by their
 * SHA-256.
 *
 * **The sound drives the picture.** With a pack open the transport runs on
 * the audio clock (`transport.setClock`): scene time advances by the samples
 * that sounded, so the timeline stands still while a chunk is still being
 * rendered and the markers are where the sound is. At another speed than 1
 * the scene is silent and the page's own clock is handed back.
 *
 * **Whose head.** `scene's head` turns the field by the recipe's head over
 * time, which is what the scene's listener hears; `my head` by the view, so
 * dragging the view is turning the head, the listener's place staying the
 * scene's.
 */
import { headMatrix3, filterSpectra, rotateSpectra } from "./sound-decode.js";
import {
  DEFAULT_LEVEL_DB,
  LEVEL_RANGE_DB,
  FULL_SCALE_SPL_DB,
  chunkSpans,
  commonRanges,
  createClipMeter,
  createFetchPlan,
  createSoundClock,
  levelGain,
  meterFill,
} from "./sound-plan.js";
import { createArrivals } from "./arrivals.js";
import { listenerAt, viewYaw } from "./tracks.js";

const RAD = Math.PI / 180;
// How often the server is told where the listener is and asked what is rendered.
const STATUS_MS = 500;
// How long a chunk request waits on the server for its render.
const WAIT_MS = 20000;
// A seek restarts the stream once the pointer has stopped for this long.
const SEEK_MS = 60;
// Steps of arrivals fetched at once: two seconds.
const ARRIVAL_STEPS = 40;
const LISTENER = "listener";

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

async function api(path, body) {
  const options =
    body === undefined
      ? undefined
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(`api/audit/${path}`, options);
  const answer = await response.json();
  if (!response.ok) throw new Error(answer.error || `${response.status}`);
  return answer;
}

const hex = (buffer) => [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, "0")).join("");

export function createSceneSound({ THREE, viewport, scene, root }) {
  const { transport, mix } = scene;
  const wall = () => performance.now() / 1000;
  const arrivals = createArrivals(THREE, viewport);
  viewport.overlays.add(arrivals.group);

  // --- the bar ----------------------------------------------------------------
  const bar = make("div", "tl-bar tl-sound");
  const pick = make("select", "tl-pack");
  pick.title = "the scene pack whose sound is played";
  const seg = (name, options, title) => {
    const group = make("div", `seg sub tl-${name}`);
    group.title = title;
    for (const [value, label] of options) {
      const button = make("button", "", label);
      button.type = "button";
      button.dataset.value = value;
      group.append(button);
    }
    return group;
  };
  const heads = seg("whose", [["scene", "scene's head"], ["mine", "my head"]], "whose head turns the field");
  const voices = seg("voices", [["pack", "as traced"], ["on", "directive"], ["off", "omni"]], "voice directivity; changing it renders the stems it concerns again");
  const level = make("label", "tl-trail");
  const levelInput = make("input");
  Object.assign(levelInput, { type: "number", min: LEVEL_RANGE_DB[0], max: LEVEL_RANGE_DB[1], step: 1, value: DEFAULT_LEVEL_DB });
  level.title =
    `the page's level. At 0 dB a sample of 1 in the pack, ${FULL_SCALE_SPL_DB} dB SPL, is the output's full scale; ` +
    `at the default, ${DEFAULT_LEVEL_DB} dB, full scale stands for ${FULL_SCALE_SPL_DB - DEFAULT_LEVEL_DB} dB SPL, which keeps a whole scene under it`;
  level.append(make("span", "", "level"), levelInput, make("em", "", "dB"));
  // Lit while the output passes full scale. The browser clips there; nothing here limits
  // the sound, which would change what is heard without saying so.
  const clipBadge = make("span", "tl-clip");
  const ahead = make("label", "check");
  const aheadBox = make("input");
  aheadBox.type = "checkbox";
  aheadBox.checked = true;
  ahead.title = "render the whole scene while the page is open, not only what is near the cursor";
  ahead.append(aheadBox, make("span", "", "render ahead"));
  const verify = make("button", "tl-verify", "verify");
  verify.type = "button";
  verify.title = "hash the bytes the page receives for the chunk under the cursor and compare with the engine's";
  const badge = make("span", "tl-badge");
  const note = make("span", "note tl-note");
  bar.append(make("span", "tl-tag", "sound"), pick, heads, voices, level, clipBadge, ahead, verify, badge, note);
  root.querySelector(".tl-bar").after(bar);

  // --- state ------------------------------------------------------------------
  let packs = [];
  let pack = null; // the pack open, an entry of `packs`
  let directivity = "pack";
  let headMode = "scene";
  let status = null;
  let plan = null;
  let poll = null;
  let opened = 0; // bumped whenever the pack or its settings change

  let context = null;
  let node = null;
  let gain = null;
  let decoder = null;
  let base = null; // the decoder's spectra, unturned
  const spare = []; // filter buffers the audio thread gave back
  let audio = null; // the promise of the three above
  let failure = "";

  const clock = createSoundClock({ sampleRate: 48000, contextTime: () => (context ? context.currentTime : 0) });
  let streaming = false;
  let starved = false;
  let position = 0; // the sample being heard, as the audio thread last said
  let pressed = 0; // when play was pressed, for the time to first sound
  let firstSound = null;
  let seekTimer = null;
  let lastHead = [NaN, NaN, NaN];
  let turnMs = 0;
  let thread = null; // what the audio thread last said of itself
  let verdict = "";
  const clip = createClipMeter({ hold: 3 });

  const levels = new Map(); // chunk index -> { source id: [dB per step] }
  let top = -200;
  let arrivalCache = new Map(); // `${source}:${first step}` -> answer, or a promise of it
  let drawn = "";

  const query = (extra = {}) => new URLSearchParams({ pack: pack.id, directivity, ...extra }).toString();
  const audible = () => mix.state().audible;
  const rate = () => (status ? status.sample_rate_hz : 48000);
  const sampleAt = (t) => Math.round(t * rate());

  // --- the audio thread -------------------------------------------------------
  function ensureAudio() {
    if (audio) return audio;
    audio = (async () => {
      const records = await fetch("decoders/decoders.json").then((r) => r.json());
      const record = records[0];
      const filters = new Float32Array(await fetch(`decoders/${record.url}`).then((r) => r.arrayBuffer()));
      decoder = { order: record.order, channels: record.channels, taps: record.taps, filters };
      base = filterSpectra(decoder);
      context = context || new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
      if (!context.audioWorklet) throw new Error("this page has no AudioWorklet: open it on localhost");
      await context.audioWorklet.addModule(new URL("./sound.worklet.js", import.meta.url));
      node = new AudioWorkletNode(context, "scene-stream", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
        processorOptions: { channels: decoder.channels },
      });
      node.port.onmessage = (event) => heard(event.data);
      node.onprocessorerror = () => {
        failure = "the audio thread stopped on an error";
        say();
      };
      gain = context.createGain();
      setLevel();
      node.connect(gain).connect(context.destination);
      lastHead = [NaN, NaN, NaN];
      turnHead();
    })();
    audio.catch((error) => {
      failure = error.message;
      audio = null;
      // Without sound the picture still plays, on the page's clock.
      transport.setClock(wall);
      say();
    });
    return audio;
  }

  function setLevel() {
    if (gain) gain.gain.value = levelGain(levelInput.value);
    // A new level is a new question: what clipped at the old one is forgotten.
    clip.reset();
    showClip();
  }

  /** The badge: lit for three seconds after the output passed full scale, with by how much. */
  function showClip() {
    const state = clip.state(context ? context.currentTime : 0);
    clipBadge.textContent = state.clipping ? `clip +${state.overDb.toFixed(1)} dB` : "";
    clipBadge.title = state.events
      ? `the output passed full scale ${state.events} time(s) at this level, by ${state.overDb.toFixed(1)} dB at most: lower the level by that much`
      : "";
  }

  /** What the audio thread says: its count for the clock, and when it ran dry. */
  function heard(message) {
    if (message.type === "started") {
      if (plan && message.gen === plan.gen()) clock.begin(message.played);
    } else if (message.type === "state") {
      thread = message;
      if (Number.isFinite(message.peak)) {
        clip.report(message.peak, levelGain(levelInput.value), message.at);
        showClip();
      }
      const sounding = message.running && !message.waiting;
      clock.report(message.played, message.at, sounding);
      if (streaming) {
        position = Math.max(0, message.position);
        if (sounding && firstSound === null && pressed) firstSound = (performance.now() - pressed) / 1000;
        if (message.starved && !starved) {
          starved = true;
          say();
        }
        pump();
      }
    } else if (message.type === "spent") {
      spare.push({ re: message.re, im: message.im });
    } else if (message.type === "error") {
      failure = message.message;
      say();
    }
  }

  /** The head, by whichever mode, as filters for the audio thread. */
  function turnHead() {
    if (!node || !base) return;
    let yaw;
    let pitch;
    let roll = 0;
    const tracks = scene.tracks();
    if (headMode === "scene" && tracks) {
      const head = listenerAt(tracks, transport.time());
      yaw = viewYaw(head.yaw_deg);
      pitch = head.pitch_deg * RAD;
      roll = head.roll_deg * RAD;
    } else {
      ({ yaw, pitch } = viewport.pose());
    }
    if (Math.abs(yaw - lastHead[0]) < 1e-4 && Math.abs(pitch - lastHead[1]) < 1e-4 && Math.abs(roll - lastHead[2]) < 1e-4) return;
    lastHead = [yaw, pitch, roll];
    const started = performance.now();
    const turned = rotateSpectra(base, decoder.order, headMatrix3(yaw, pitch, roll), spare.pop() || null);
    turnMs += (performance.now() - started - turnMs) * 0.1;
    node.port.postMessage({ type: "filters", re: turned.re, im: turned.im }, [turned.re.buffer, turned.im.buffer]);
  }

  // --- the stream -------------------------------------------------------------
  function pump() {
    if (!streaming || !plan || !pack) return;
    for (const index of plan.next(position)) fetchChunk(plan.gen(), index, opened);
  }

  async function fetchChunk(gen, index, mine) {
    const sources = audible();
    try {
      const response = await fetch(`api/audit/chunk?${query({ index, sources: sources.join(","), wait_ms: WAIT_MS })}`);
      if (mine !== opened || !plan) return;
      if (!response.ok) {
        plan.failed(gen, index);
        if (response.status !== 503) failure = (await response.json()).error || `${response.status}`;
        if (gen === plan.gen()) setTimeout(pump, 250);
        say();
        return;
      }
      const data = new Float32Array(await response.arrayBuffer());
      if (mine !== opened || !plan || !plan.arrived(gen, index)) return;
      const heardLevels = JSON.parse(response.headers.get("X-Audit-Levels") || "{}");
      levels.set(index, { ...(levels.get(index) || {}), ...heardLevels });
      for (const values of Object.values(heardLevels)) for (const db of values) top = Math.max(top, db);
      const go = plan.mayGo(position, { starved });
      if (go) starved = false;
      node.port.postMessage(
        {
          type: "chunk",
          gen,
          start: Number(response.headers.get("X-Audit-Start")),
          frames: Number(response.headers.get("X-Audit-Frames")),
          data,
          go,
        },
        [data.buffer]
      );
      pump();
      say();
    } catch (error) {
      if (plan) plan.failed(gen, index);
      failure = error.message;
      say();
    }
  }

  async function startStream() {
    if (!pack || transport.state().speed !== 1) return;
    clock.hold();
    pressed = performance.now();
    const mine = opened;
    try {
      await ensureAudio();
      await context.resume();
    } catch {
      return;
    }
    if (mine !== opened || !plan || !transport.state().playing) return;
    streaming = true;
    starved = false;
    firstSound = null;
    position = sampleAt(transport.time());
    turnHead();
    node.port.postMessage({ type: "start", sample: position, gen: plan.restart(), total: status.samples });
    pump();
    tell();
  }

  function stopStream() {
    clearTimeout(seekTimer);
    if (!streaming) return;
    streaming = false;
    clock.hold();
    if (node) node.port.postMessage({ type: "stop" });
    if (plan) plan.restart();
  }

  /** A solo, a mute, a switch: the same instant under another mix, faded to. */
  function remix() {
    if (!streaming || !plan) return;
    plan.restart();
    pump();
    tell();
  }

  // --- the server -------------------------------------------------------------
  async function tell() {
    if (!pack) return;
    const mine = opened;
    try {
      const cursor = plan ? plan.chunkAt(streaming ? position : sampleAt(transport.time())) : 0;
      const answer = await api("status", { pack: pack.id, directivity, cursor, audible: audible(), background: aheadBox.checked });
      if (mine !== opened) return;
      status = answer;
      if (!plan) plan = createFetchPlan({ chunkSamples: status.chunk_samples, chunks: status.chunks });
      failure = "";
      for (const [id, source] of Object.entries(status.sources)) {
        const broken = Object.values(source.failed)[0];
        if (broken) failure = `${id}: ${broken}`;
      }
      scene.refreshTimeline();
    } catch (error) {
      if (mine !== opened) return;
      failure = error.message;
    }
    say();
  }

  function say() {
    for (const group of [heads, voices]) {
      const value = group === heads ? headMode : directivity;
      for (const button of group.querySelectorAll("button")) button.classList.toggle("on", button.dataset.value === value);
    }
    voices.hidden = !(pack && pack.directivity_switch);
    verify.disabled = !pack;
    const placeholders = status ? Object.values(status.sources).filter((source) => source.placeholder).length : 0;
    badge.textContent = placeholders ? "placeholder audio" : "";
    badge.title = status
      ? Object.entries(status.sources)
          .map(([id, source]) => `${id}: ${source.dry}`)
          .join("\n")
      : "";
    if (!pack) {
      note.textContent = packs.length ? "" : "no pack for this scene";
      note.classList.remove("bad");
      return;
    }
    const parts = [];
    if (failure) parts.push(failure);
    else if (!status) parts.push("opening the pack");
    else {
      const playing = transport.state().playing;
      if (playing && transport.state().speed !== 1) parts.push("silent at this speed");
      else if (playing && context && context.state !== "running") parts.push("the browser holds the sound: press play");
      else if (playing && (starved || clock.held())) parts.push("waiting for the render");
      if (firstSound !== null) parts.push(`first sound after ${firstSound.toFixed(2)} s`);
      if (status.rate !== null) parts.push(`rendering ${status.rate.toFixed(2)} s of stem a second`);
      if (status.blocked) parts.push(status.blocked);
      const bytes = Object.values(status.sources).reduce((sum, source) => sum + source.disk_bytes, 0);
      parts.push(`stems ${(bytes / 1e9).toFixed(2)} GB`);
      if (verdict) parts.push(verdict);
    }
    note.textContent = parts.join(" · ");
    note.title = status ? `${status.cache}\nturning the head: ${turnMs.toFixed(1)} ms` : "";
    note.classList.toggle("bad", Boolean(failure));
  }

  function close() {
    stopStream();
    opened += 1;
    clearInterval(poll);
    poll = null;
    pack = null;
    status = null;
    plan = null;
    levels.clear();
    top = -200;
    arrivalCache = new Map();
    drawn = "";
    verdict = "";
    failure = "";
    firstSound = null;
    arrivals.clear();
    transport.setClock(wall);
    scene.timelineSound(null);
  }

  async function open(id) {
    close();
    const found = packs.find((entry) => entry.id === id);
    pick.value = found ? id : "";
    if (!found) {
      say();
      return;
    }
    pack = found;
    const mine = opened;
    say();
    const tracks = scene.tracks();
    if (!tracks || tracks.recipe_sha256 !== pack.recipe_sha256) {
      // A pack whose recipe is not on show: the pack itself says where everything is.
      try {
        const own = await api(`tracks?${query()}`);
        if (mine !== opened) return;
        keep = pack.id;
        scene.showTracks(own);
      } catch (error) {
        failure = error.message;
        say();
        return;
      }
    }
    if (mine !== opened) return;
    scene.timelineSound({
      ready: (source) =>
        status && status.sources[source]
          ? chunkSpans(status.sources[source].ready, status.chunk_samples, rate(), transport.state().duration)
          : [],
      mixReady: () =>
        status
          ? chunkSpans(
              commonRanges(audible().map((source) => status.sources[source].ready)),
              status.chunk_samples,
              rate(),
              transport.state().duration
            )
          : [],
      level(source, t) {
        if (!status) return 0;
        const sample = t * rate();
        const index = Math.floor(sample / status.chunk_samples);
        const values = (levels.get(index) || {})[source];
        if (!values || !mix.audible(source)) return 0;
        return meterFill(values[Math.floor((sample - index * status.chunk_samples) / status.step_samples)], top);
      },
      label: (source) => (status && status.sources[source] ? status.sources[source].dry : ""),
    });
    if (transport.state().speed === 1) {
      clock.hold();
      transport.setClock(clock.now.bind(clock));
    }
    poll = setInterval(tell, STATUS_MS);
    await tell();
    if (mine === opened && transport.state().playing) startStream();
  }

  // The pack the stage opened by itself showing its own tracks: kept when the scene changes to them.
  let keep = null;

  async function list(recipe) {
    try {
      packs = await api(`packs${recipe ? `?recipe_sha256=${recipe}` : ""}`);
    } catch (error) {
      packs = [];
      failure = error.message;
    }
    const options = [new Option(packs.length ? "no sound" : "no pack", "")];
    for (const entry of [...packs].sort((a, b) => Number(b.matches) - Number(a.matches))) {
      const minutes = (entry.duration_s / 60).toFixed(1);
      options.push(new Option(`${entry.matches ? "" : "other · "}${entry.name} · ${entry.sources.length} sources · ${minutes} min`, entry.id));
    }
    pick.replaceChildren(...options);
    pick.value = pack ? pack.id : "";
    say();
  }

  // --- what is drawn ------------------------------------------------------------
  async function draw() {
    const selected = scene.selected();
    if (!pack || !status || !selected || selected === LISTENER) {
      if (drawn) arrivals.clear();
      drawn = "";
      return;
    }
    const steps = Math.round(status.samples / status.step_samples);
    const step = Math.min(steps - 1, Math.max(0, Math.floor((transport.time() * rate()) / status.step_samples)));
    const name = `${selected}:${step}`;
    if (name === drawn) return;
    const first = Math.floor(step / ARRIVAL_STEPS) * ARRIVAL_STEPS;
    const key = `${selected}:${first}`;
    if (!arrivalCache.has(key)) {
      const mine = opened;
      const asked = api(`arrivals?${query({ source: selected, from: first, to: first + ARRIVAL_STEPS })}`)
        .then((answer) => {
          if (mine === opened) arrivalCache.set(key, answer);
          draw();
        })
        .catch(() => arrivalCache.delete(key));
      arrivalCache.set(key, asked);
      if (arrivalCache.size > 64) arrivalCache.delete(arrivalCache.keys().next().value);
    }
    const answer = arrivalCache.get(key);
    if (!answer || answer instanceof Promise || !answer.steps[step - first]) return;
    drawn = name;
    arrivals.show(answer.steps[step - first], { soundSpeed: answer.sound_speed_m_s, grid: answer.tail_grid });
  }

  // --- the proof ----------------------------------------------------------------
  async function check() {
    if (!pack || !plan) return;
    const index = plan.chunkAt(sampleAt(transport.time()));
    const sources = audible();
    verdict = `verifying chunk ${index}`;
    say();
    try {
      const sets = [...sources.map((source) => [source]), ...(sources.length > 1 ? [sources] : [])];
      for (const set of sets) {
        const named = { index, sources: set.join(",") };
        const response = await fetch(`api/audit/chunk?${query({ ...named, wait_ms: WAIT_MS })}`);
        if (!response.ok) throw new Error((await response.json()).error);
        const mine = hex(await crypto.subtle.digest("SHA-256", await response.arrayBuffer()));
        const theirs = await api(`checksum?${query(named)}`);
        if (mine !== theirs.sha256) {
          verdict = `chunk ${index}, ${set.join("+")}: the page's bytes are NOT the engine's`;
          failure = verdict;
          say();
          return;
        }
      }
      verdict = `chunk ${index}: ${sources.length} stems${sources.length > 1 ? " and their mix" : ""} are the engine's bytes`;
    } catch (error) {
      verdict = `not verified: ${error.message}`;
    }
    say();
  }

  // --- wiring -------------------------------------------------------------------
  transport.on("play", () => {
    if (!pack) return;
    startStream();
    say();
  });
  transport.on("pause", () => {
    stopStream();
    say();
  });
  transport.on("load", stopStream);
  transport.on("seek", () => {
    if (!pack) return;
    drawn = "";
    if (!transport.state().playing || transport.state().speed !== 1) {
      tell();
      return;
    }
    // The clock stops where the transport anchored it, and the stream follows
    // once the pointer rests.
    stopStream();
    seekTimer = setTimeout(startStream, SEEK_MS);
  });
  transport.on("speed", ({ speed, playing }) => {
    if (!pack) return;
    if (speed !== 1) {
      stopStream();
      transport.setClock(wall);
    } else {
      clock.hold();
      transport.setClock(clock.now.bind(clock));
      if (playing) startStream();
    }
    say();
  });
  transport.on("tick", () => {
    if (!pack) return;
    if (headMode === "scene") turnHead();
    draw();
  });
  mix.on("change", () => {
    if (!pack) return;
    remix();
    tell();
  });
  viewport.onMove(() => {
    if (pack && headMode === "mine") turnHead();
  });
  scene.onSelect(() => {
    drawn = "";
    draw();
  });
  scene.onScene(async (shown) => {
    const recipe = shown ? shown.tracks.recipe_sha256 : null;
    if (pack && keep === pack.id && recipe === pack.recipe_sha256) {
      // The stage's own doing: the pack's tracks were put on show.
      await list(recipe);
      return;
    }
    keep = null;
    close();
    await list(recipe);
    // A pack dropped beside its recipe is found and opened.
    const match = shown ? packs.find((entry) => entry.matches) : null;
    if (match) open(match.id);
  });

  pick.addEventListener("pointerdown", () => {
    const tracks = scene.tracks();
    list(tracks ? tracks.recipe_sha256 : null);
  });
  pick.addEventListener("change", () => open(pick.value));
  heads.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    headMode = button.dataset.value;
    lastHead = [NaN, NaN, NaN];
    turnHead();
    say();
  });
  voices.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button || button.dataset.value === directivity || !pack) return;
    directivity = button.dataset.value;
    // Other stems: what is rendered, the levels and the stream all follow.
    levels.clear();
    verdict = "";
    remix();
    tell();
  });
  levelInput.addEventListener("change", setLevel);
  aheadBox.addEventListener("change", tell);
  verify.addEventListener("click", check);
  list(null);

  return {
    open,
    close,
    list,
    check,
    setHead(mode) {
      headMode = mode;
      lastHead = [NaN, NaN, NaN];
      turnHead();
      say();
    },
    /** For a check that cannot listen: what the stage is doing. */
    state: () => ({
      pack: pack ? pack.id : null,
      packs: packs.map((entry) => ({ id: entry.id, name: entry.name, matches: entry.matches })),
      streaming,
      starved,
      held: clock.held(),
      clock: clock.now(),
      position,
      firstSound,
      turnMs,
      headMode,
      directivity,
      verdict,
      failure,
      thread,
      context: context ? { state: context.state, base: context.baseLatency, output: context.outputLatency } : null,
      status,
    }),
  };
}
