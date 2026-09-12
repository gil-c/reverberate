/** Boot: wire the panels, the viewport, the plan, the sound and the run. */
import { createViewport, roomAt } from "./viewport.js";
import { createMeshViews } from "./grid.js";
import { createSourceGlyphs } from "./sources.js";
import { createMinimap } from "./minimap.js";
import { createListenerTab } from "./listener.js";
import { createPlayers, createSourceList } from "./players.js";
import { setupPanels } from "./panels.js";
import { createPlots } from "./plots.js";
import { createPoints } from "./points.js";
import { bindSettings, loadSettings } from "./settings.js";
import { setupFolds } from "./folds.js";
import { createEngine } from "./audio/engine.js";
import { loadField } from "./audio/field.js";
import { createSpatial } from "./audio/spatial.js";
import { state } from "./state.js";

const $ = (selector) => document.querySelector(selector);
const busy = (text) => {
  $("#hud-busy").textContent = text || "";
};
const sourceById = (id) => state.sources.find((s) => s.id === id);
const audibleIds = () => state.sources.filter((s) => s.on && s.field).map((s) => s.id);

// --- picture ------------------------------------------------------------------
const viewport = createViewport($("#scene"), $("#view-pane"));
const { THREE } = viewport;
setupPanels($("#app"), () => viewport.resize());
setupFolds($("#left"));
const glyphs = createSourceGlyphs(THREE, viewport);
viewport.overlays.add(glyphs.group);
const points = createPoints(viewport);
const minimap = createMinimap($("#map"), {
  onMove: (x, z) => viewport.moveTo({ x, z }),
  onSelectSource: (id) => selectSource(id),
});
const listenerTab = createListenerTab($("#pose-fields"), { onEdit: (pose) => viewport.moveTo(pose) });

// --- sound --------------------------------------------------------------------
const engine = createEngine();
const plots = createPlots($("#spectrogram"), $("#decay"), {
  spectrogram: $("#spectrogram-cap"),
  decay: $("#decay-cap"),
});
let lastRender = null;
const spatial = createSpatial({
  engine,
  workerUrl: new URL("./audio/brir.worker.js", import.meta.url),
  onRendered: (id, message) => {
    const energy = (ear) => ear.reduce((sum, v) => sum + v * v, 0);
    lastRender = { id, left: energy(message.early[0]), right: energy(message.early[1]), ms: message.ms };
    if (id === state.selected) {
      plots.drawSpectrogram(message.spectrogram, engine.sampleRate, message.seconds);
      plots.drawDecay(message.decay, message.seconds);
    }
    audioStatus();
  },
  onStatus: (id, status) => {
    const source = sourceById(id);
    if (source) Object.assign(source, status);
    if (status.error) busy(`${id}: ${status.error}`);
    audioStatus();
  },
});
let heads = [];

function audioStatus() {
  const source = sourceById(state.selected);
  const parts = [];
  parts.push(source && source.cell !== null && source.cell !== undefined ? `cell ${source.cell}` : "no cell");
  if (source && source.solvedToHz) parts.push(`solved to ${(source.solvedToHz / 1000).toFixed(1)} kHz`);
  if (lastRender) parts.push(`update <b>${lastRender.ms.toFixed(1)} ms</b>`);
  if (source && source.late) parts.push(source.late === "exact" ? "late exact" : "late at entry");
  $("#audio-update").innerHTML = parts.join(" · ");
}

async function loadHead(name) {
  const record = heads.find((h) => h.name === name) || heads[0];
  if (!record) return;
  const bytes = await fetch(`decoders/${record.url}`).then((r) => r.arrayBuffer());
  spatial.setDecoder({ order: record.order, channels: record.channels, taps: record.taps, filters: new Float32Array(bytes) });
  $("#audio-format").textContent = `binaural · order ${record.order} · ${engine.sampleRate / 1000} kHz · ${record.name}`;
  spatial.update(viewport.pose(), audibleIds());
}

const players = createPlayers($("#players"), {
  onSelect: (id) => selectSource(id),
  isPlaying: (id) => engine.isPlaying(id),
  onPlay: async (id, play) => {
    const source = sourceById(id);
    if (!source) return;
    if (play) {
      try {
        if (source.voice) await engine.setVoice(id, source.voice);
        await engine.play(id);
      } catch (error) {
        busy(`${id}: ${error.message}`);
      }
      spatial.update(viewport.pose(), [id]);
    } else {
      engine.stop(id);
    }
    renderSources();
  },
  onVolume: (id, value) => {
    sourceById(id).volume = value;
    engine.setVolume(id, value);
  },
  onLoop: (id, loop) => {
    sourceById(id).loop = loop;
    engine.setLoop(id, loop);
    renderSources();
  },
  onVoice: (id, url) => {
    sourceById(id).voice = url;
    engine.setVoice(id, url);
  },
});
const sourceList = createSourceList($("#sources"), {
  onSelect: (id) => selectSource(id),
  onToggle: (id, enabled) => {
    sourceById(id).on = enabled;
    if (enabled) spatial.update(viewport.pose(), [id]);
    else {
      engine.stop(id);
      engine.silence(id);
    }
    renderSources();
  },
});

/** Every view of the sources: list, players, plan marks, glyphs, panel state. */
function renderSources() {
  // Nothing to hear: the audio panel is dimmed and inert rather than empty,
  // so a panel with no sound in it is never taken for a broken one.
  $("#right").classList.toggle("idle", !audibleIds().length);
  sourceList.render(state.sources, state.selected);
  players.render(state.sources, state.selected, state.listener);
  minimap.setSources(state.sources, state.selected);
  glyphs.update(state.sources, state.selected);
  viewport.invalidate();
  $("#plot-source").textContent = state.selected || "";
}

function selectSource(id) {
  state.selected = id;
  renderSources();
  plots.clear();
  spatial.update(viewport.pose(), audibleIds());
}

// --- settings and exact mode -----------------------------------------------------
const settings = loadSettings();
spatial.setSettings({ earlyMs: settings.earlyMs, stillMs: settings.stillMs, fadeMs: settings.fadeMs });
points.show(settings.showPoints);
minimap.showPoints(settings.showPoints);
viewport.setGrab(settings.grabDrag);
bindSettings($("#tab-set"), settings, (key, value) => {
  if (["earlyMs", "stillMs", "fadeMs"].includes(key)) {
    spatial.setSettings({ [key]: value });
    spatial.update(viewport.pose(), audibleIds());
  }
  if (key === "showPoints") {
    points.show(value);
    minimap.showPoints(value);
    viewport.invalidate();
  }
  if (key === "exact") armSettle();
  if (key === "grabDrag") viewport.setGrab(value);
});

/** Exact mode: once the listener has stood still for `settleMs`, glide to
 *  the nearest measurement point of the selected source, or of any source. */
let settleTimer = null;
let settledAt = null;
function nearestPoint(pose) {
  for (const id of [state.selected, ...audibleIds()]) {
    const field = spatial.fieldOf(id);
    const cell = field ? field.cellAt(pose.x, pose.y, pose.z) : null;
    if (cell !== null) return field.index.positions[cell];
  }
  return null;
}
function armSettle() {
  clearTimeout(settleTimer);
  if (!settings.exact || viewport.isGliding()) return;
  settleTimer = setTimeout(() => {
    const pose = viewport.pose();
    const target = nearestPoint(pose);
    if (!target) return;
    const [x, y, z] = target;
    if (Math.hypot(x - pose.x, y - pose.y, z - pose.z) < 1e-3) return;
    settledAt = target;
    viewport.glideTo({ x, y, z }, settings.glideMs);
  }, settings.settleMs);
}

// --- view and mesh selectors ----------------------------------------------------
let meshViews = null;
let grid = null;
// Bumped on every run change, so a grid or a field still loading for the
// old run is dropped rather than attached to the new one.
let generation = 0;

const tierText = (status) => {
  if (!status || !status.room) return "";
  const tiles = status.tiles > 1 ? ` · ${status.tiles_drawn}/${status.tiles} tiles` : "";
  return `fine ${status.fine_mm.toFixed(2)} mm${tiles} · far ${status.coarse_mm.toFixed(2)} mm`;
};

function renderViewButtons() {
  for (const button of $("#view").querySelectorAll("button")) {
    button.classList.toggle("on", button.dataset.view === state.view);
  }
  $("#fmax").replaceChildren(
    ...(meshViews ? meshViews.bands : []).map((band) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `${Number(band) / 1000} kHz`;
      button.disabled = Boolean(state.room) && !meshViews.availableFor(state.room).includes(band);
      button.classList.toggle("on", state.fmax === band);
      button.addEventListener("click", () => setFmax(band));
      return button;
    })
  );
  $("#fmax").style.opacity = state.view === "acoustic" ? 1 : 0.45;
}
$("#view").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (button) setView(button.dataset.view);
});

function setView(view) {
  if (view === "acoustic" && !meshViews) return;
  state.view = view;
  if (view === "acoustic") reconcileMesh();
  else $("#hud-tier").textContent = "";
  viewport.show(state.view);
  renderViewButtons();
}

async function setFmax(band) {
  if (!meshViews || state.fmax === band) return;
  state.fmax = band;
  const mine = generation;
  renderViewButtons();
  busy(`loading ${Number(band) / 1000} kHz grid`);
  try {
    grid = await meshViews.view(band);
  } catch (error) {
    busy(`${Number(band) / 1000} kHz grid failed: ${error.message}`);
    return;
  }
  if (mine !== generation || state.fmax !== band) return;
  busy("");
  viewport.setAcoustic(grid.group);
  followGrid(viewport.pose());
}

function followGrid(pose) {
  if (state.view !== "acoustic" || !grid) return;
  grid.follow(new THREE.Vector3(pose.x, pose.y, pose.z), meshViews.roomIn(state.fmax, state.room));
}

/** Keep the band limit honest for the room the listener stands in.
 *
 * A band with no payload for this room is disabled. If the one in force is
 * such a band, the finest available takes over; with none, the view drops
 * to colour. Nothing is drawn coarser than its label says.
 */
function reconcileMesh() {
  if (!meshViews) return;
  const note = $("#fmax-note");
  const available = state.room ? meshViews.availableFor(state.room) : meshViews.bands;
  if (state.view === "acoustic") {
    if (!available.length) {
      note.textContent = state.room ? `no grid in ${state.room.name}` : "no grid";
      state.fmax = null;
      setView("colour");
      return;
    }
    if (!available.includes(state.fmax)) {
      note.textContent = state.fmax ? `${Number(state.fmax) / 1000} kHz not built for ${state.room.name}` : "";
      setFmax(available[available.length - 1]);
    } else {
      note.textContent = "";
    }
  }
  renderViewButtons();
}

// --- following the listener -----------------------------------------------------
viewport.onMove((pose) => {
  const moved = pose.x !== state.listener.x || pose.y !== state.listener.y || pose.z !== state.listener.z;
  state.listener = pose;
  const rooms = state.manifest ? state.manifest.rooms : [];
  const name = roomAt(rooms, pose.x, pose.z);
  if (name !== (state.room && state.room.name)) {
    state.room = rooms.find((room) => room.name === name) || null;
    $("#room").textContent = name || "";
    reconcileMesh();
  }
  followGrid(pose);
  const degrees = ((((pose.yaw * 180) / Math.PI) % 360) + 360) % 360;
  $("#hud-pos").textContent = `${pose.x.toFixed(2)} ${pose.y.toFixed(2)} ${pose.z.toFixed(2)} · ${degrees | 0}°`;
  minimap.update(pose, name);
  listenerTab.update(pose);
  players.updateDistances(state.sources, pose);
  points.follow(pose.x, pose.y, pose.z);
  spatial.update(pose, audibleIds());
  // A translation arms the settle timer; a turn on the spot, the glide's own
  // motion and arriving on the point do not.
  if (moved) {
    const atRest = settledAt && Math.hypot(pose.x - settledAt[0], pose.y - settledAt[1], pose.z - settledAt[2]) < 1e-3;
    if (viewport.isGliding() || atRest) clearTimeout(settleTimer);
    else {
      settledAt = null;
      armSettle();
    }
  }
});

// --- apartments and runs ----------------------------------------------------------
const runsByScene = new Map();
const apartmentsById = new Map();
let voices = [];

/** Open a run of the apartment on screen, or none: its sources, fields and grids. */
async function openRun(run) {
  generation += 1;
  state.run = run;
  grid = null;
  meshViews = null;
  state.fmax = null;
  for (const source of state.sources) {
    engine.stop(source.id);
    spatial.setField(source.id, null);
  }
  points.set([]);
  minimap.setPoints([]);
  viewport.setAcoustic(null);
  plots.clear();
  lastRender = null;
  $("#hud-tier").textContent = "";
  if (state.view === "acoustic") setView("colour");
  state.sources = [];
  state.selected = null;
  if (run) {
    const data = await fetch(`${run.url}/run.json`).then((r) => r.json());
    if (state.run !== run) return;
    data.baseUrl = run.url;
    state.sources = data.sources.map((source, i) => ({
      ...source,
      on: true,
      volume: 0.8,
      loop: true,
      voice: voices.length ? voices[i % voices.length].url : null,
      cell: null,
    }));
    state.selected = state.sources.length ? state.sources[0].id : null;
    meshViews = createMeshViews(THREE, data, (status) => {
      $("#hud-tier").textContent = state.view === "acoustic" ? tierText(status) : "";
    });
    const mine = generation;
    for (const source of state.sources.filter((s) => s.field)) {
      engine.setLoop(source.id, true);
      engine.setVolume(source.id, source.volume);
      loadField(`${run.url}/${source.field.url}`)
        .then((field) => {
          if (mine !== generation) return;
          spatial.setField(source.id, field);
          points.set(audibleIds().map((id) => spatial.fieldOf(id)).filter(Boolean));
          minimap.setPoints(points.positions());
          spatial.update(viewport.pose(), [source.id]);
        })
        .catch((error) => busy(`${source.id} field: ${error.message}`));
    }
    // Opening a run is a choice of what to listen to: stand a metre from its
    // first source, facing it, rather than wherever the apartment left us.
    if (state.sources.length) {
      const [sx, sy, sz] = state.sources[0].position;
      viewport.moveTo({ x: sx + 1.0, y: sy, z: sz, yaw: Math.PI / 2, pitch: 0 });
    }
  }
  glyphs.set(state.sources);
  renderSources();
  audioStatus();
  renderViewButtons();
  if (meshViews && meshViews.bands.length) setView("acoustic");
}

/** The walkable outline's extent on each axis, and its area by the shoelace formula. */
function outlineExtent(outline) {
  const xs = [];
  const zs = [];
  let area = 0;
  const ringArea = (ring) => {
    let sum = 0;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      sum += ring[j][0] * ring[i][1] - ring[i][0] * ring[j][1];
    }
    return Math.abs(sum) / 2;
  };
  for (const part of outline || []) {
    for (const [x, z] of part.exterior) xs.push(x), zs.push(z);
    area += ringArea(part.exterior);
    for (const hole of part.holes || []) area -= ringArea(hole);
  }
  if (!xs.length) return { x: [-10, 10], z: [-10, 10], area: 0 };
  return { x: [Math.min(...xs), Math.max(...xs)], z: [Math.min(...zs), Math.max(...zs)], area };
}

/** A few numbers about the apartment, from what the manifest already holds. */
function renderFacts(manifest, area) {
  const doorways = /(\d+) doorways/.exec(`${manifest.title || ""} ${manifest.hint || ""}`);
  const facts = [
    ["floor", `${area.toFixed(0)} m²`],
    ["rooms", `${(manifest.rooms || []).length}`],
    ["ceiling", `${(manifest.ceilingHeight - manifest.floorHeight).toFixed(2)} m`],
    ["doorways", doorways ? doorways[1] : "—"],
    ["pieces", `${(manifest.instances || []).length}`],
  ];
  if (manifest.storeys > 1) facts.push(["storey", `${(manifest.storey_index || 0) + 1} of ${manifest.storeys}`]);
  $("#facts").replaceChildren(
    ...facts.flatMap(([name, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = name;
      const dd = document.createElement("dd");
      dd.textContent = value;
      return [dt, dd];
    })
  );
}

/** Open an apartment by short name, then the run asked for or the lead one. */
async function openApartment(id, runName) {
  state.apartment = id;
  state.manifest = null;
  state.room = null;
  $("#facts").replaceChildren();
  const runs = runsByScene.get(apartmentsById.get(id).scene_id) || [];
  const runSelect = $("#run");
  runSelect.hidden = runs.length < 2;
  runSelect.replaceChildren(
    ...runs.map((run) => {
      const option = document.createElement("option");
      option.value = run.name;
      option.textContent = run.name;
      return option;
    })
  );
  const run = runs.find((r) => r.name === runName) || runs.find((r) => r.lead) || runs[0] || null;
  if (run) runSelect.value = run.name;

  busy("opening apartment");
  const base = `scenes/${id}/`;
  let manifest;
  try {
    manifest = await fetch(`${base}manifest.json`).then((r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    });
  } catch (error) {
    busy(`apartment ${id} failed: ${error.message}`);
    return;
  }
  if (state.apartment !== id) return;
  state.manifest = manifest;
  const extent = outlineExtent(manifest.outline);
  minimap.setPlan({ rooms: manifest.rooms, outline: manifest.outline });
  listenerTab.setBounds({
    x: extent.x,
    z: extent.z,
    height: [0.3, manifest.ceilingHeight - manifest.floorHeight - 0.3],
    floorHeight: manifest.floorHeight,
  });
  renderFacts(manifest, extent.area);
  busy("loading furniture");
  const furnished = viewport.setApartment(manifest, base);
  await openRun(run);
  await furnished;
  if (state.apartment === id) busy("");
}

async function boot() {
  voices = await fetch("voices/voices.json").then((r) => (r.ok ? r.json() : [])).catch(() => []);
  voices = voices.map((voice) => ({ ...voice, url: `voices/${voice.url}` }));
  players.setVoices(voices);

  heads = await fetch("decoders/decoders.json").then((r) => (r.ok ? r.json() : [])).catch(() => []);
  const headSelect = $("#head");
  headSelect.replaceChildren(
    ...heads.map((record) => {
      const option = document.createElement("option");
      option.value = record.name;
      option.textContent = record.name;
      return option;
    })
  );
  headSelect.addEventListener("change", () => loadHead(headSelect.value));
  const level = $("#level");
  level.addEventListener("input", () => {
    engine.setMasterDb(Number(level.value));
    $("#level-read").textContent = `${level.value} dB`;
  });
  await loadHead(heads.length ? heads[0].name : null);

  const runs = await fetch("runs.json").then((r) => (r.ok ? r.json() : [])).catch(() => []);
  for (const run of runs) {
    if (!runsByScene.has(run.scene_id)) runsByScene.set(run.scene_id, []);
    runsByScene.get(run.scene_id).push(run);
  }
  const lead = runs.find((r) => r.lead) || runs[0] || null;
  // Only what opens without assembling is offered at all: minutes of work
  // behind a click. Apartments with a run first, the lead run's own on top.
  const apartments = (await fetch("apartments.json").then((r) => r.json())).filter((a) => a.ready);
  for (const apartment of apartments) apartmentsById.set(apartment.local, apartment);
  apartments.sort(
    (a, b) =>
      (b.scene_id === (lead && lead.scene_id)) - (a.scene_id === (lead && lead.scene_id)) ||
      (b.runs.length > 0) - (a.runs.length > 0)
  );
  const select = $("#apartment");
  const withAudio = $("#with-audio");
  const fill = () => {
    const shown = apartments.filter((a) => !withAudio.checked || a.runs.length);
    const current = select.value;
    select.replaceChildren(
      ...shown.map((apartment) => {
        const option = document.createElement("option");
        option.value = apartment.local;
        option.textContent = apartment.runs.length ? `${apartment.local} · ${apartment.runs.join(", ")}` : apartment.local;
        return option;
      })
    );
    if (shown.some((a) => a.local === current)) select.value = current;
  };
  fill();
  withAudio.addEventListener("change", fill);
  select.addEventListener("change", () => openApartment(select.value));
  $("#run").addEventListener("change", () => openApartment(select.value, $("#run").value));
  if (apartments.length) await openApartment(select.value, lead ? lead.name : undefined);
}

renderViewButtons();
boot().catch((error) => busy(`failed to start: ${error.message}`));

// Test hook: lets an automated check move the listener and read the state,
// which is otherwise only verifiable by eye.
window.reverberate = {
  state,
  settings,
  engine,
  spatial,
  points,
  moveTo: (pose) => viewport.moveTo(pose),
  glideTo: (target, ms) => viewport.glideTo(target, ms),
  pose: () => viewport.pose(),
  setView,
  setFmax,
  lastRender: () => lastRender,
};
