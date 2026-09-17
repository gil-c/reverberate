/** Boot: wire the panels, the viewport, the plan, the sound and the run. */
import { createViewport, roomAt } from "./viewport.js";
import { clippedAt, createMeshViews } from "./grid.js";
import { createSourceGlyphs } from "./sources.js";
import { createMinimap } from "./minimap.js";
import { createListenerTab } from "./listener.js";
import { createPlayers, createSourceList } from "./players.js";
import { setupPanels } from "./panels.js";
import { createPlots } from "./plots.js";
import { createDashboard } from "./dashboard.js";
import { createMirrorLayers } from "./mirror.js";
import { createAuditPanel } from "./audit.js";
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
// The mirror's audit is a view of its own, beside colour and wave.
const mirrorLayers = createMirrorLayers(THREE, { onLoaded: () => viewport.invalidate() });
mirrorLayers.onRedraw(() => viewport.invalidate());
viewport.setMirror(mirrorLayers.group);
viewport.onResize((width, height) => mirrorLayers.setSize(width, height));
viewport.resize();
const audit = createAuditPanel($("#audit"), THREE, {
  onHighlight: () => viewport.invalidate(),
  onMirrorSwitch: (name, on) => {
    mirrorLayers.setWanted(name, on);
    if (on && (name === "reflectors" || name === "occluders") && !mirrorLayers.loaded(name)) {
      busy(`loading mirror ${name}`);
      mirrorLayers.ensure().then(() => busy(""));
    }
  },
  onColourBy: (mode) => mirrorLayers.setColourBy(mode),
});
// A click without a drag on an audit view names the face under the cursor.
const pickRay = new THREE.Raycaster();
let downAt = null;
viewport.canvas.addEventListener("pointerdown", (event) => {
  downAt = [event.clientX, event.clientY];
});
viewport.canvas.addEventListener("pointerup", (event) => {
  const start = downAt;
  downAt = null;
  if (!start || Math.hypot(event.clientX - start[0], event.clientY - start[1]) > 4) return;
  if (state.view !== "acoustic" && state.view !== "mirror") return;
  const rect = viewport.canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );
  pickRay.setFromCamera(ndc, viewport.camera);
  const meshes =
    state.view === "mirror" ? mirrorLayers.pickable() : grid ? grid.group.children.filter((m) => m.visible) : [];
  const hit = pickRay
    .intersectObjects(meshes, false)
    .find((h) => !clippedAt(h.object.userData.clip, h.point));
  if (!hit) {
    audit.pick(null);
    return;
  }
  const value = hit.object.geometry.attributes.aLabel.getX(hit.face.a);
  const label = value >= 0 ? audit.labels()[value] : null;
  if (label === null || label === undefined) {
    busy(value === -2 ? "a sealed inside" : "a rigid face, no material");
    audit.pick(null);
    return;
  }
  busy("");
  if (state.view === "mirror" && mirrorLayers.isReflectors(hit.object)) {
    audit.pick({ label, layer: "reflectors", facet: mirrorLayers.facetOf(Math.floor(hit.faceIndex / 2)) });
  } else {
    audit.pick({ label, layer: state.view === "mirror" ? "occluders" : "grid" });
  }
});

/** The mirror's paths at the selected source's cell, drawn and counted. */
function showMirrorCell(id, position) {
  mirrorLayers.showCell(id, position);
  audit.setCell(id, position, mirrorLayers.pathsAt(id, position));
}
const minimap = createMinimap($("#map"), {
  onMove: (x, z) => viewport.moveTo({ x, z }),
  onSelectSource: (id) => selectSource(id),
});
const listenerTab = createListenerTab($("#pose-fields"), { onEdit: (pose) => viewport.moveTo(pose) });

// --- sound --------------------------------------------------------------------
const engine = createEngine();
const plots = createPlots({
  spectrogram: $("#spectrogram"),
  decay: $("#decay"),
  direction: $("#direction"),
  captions: { spectrogram: $("#spectrogram-cap"), decay: $("#decay-cap"), direction: $("#direction-cap") },
  workerUrl: new URL("./audio/plots.worker.js", import.meta.url),
});
const dashboard = createDashboard($("#mirror"), {
  table: $("#mirror-table"),
  summary: $("#mirror-summary"),
  caption: $("#mirror-cap"),
});
let lastRender = null;
const spatial = createSpatial({
  engine,
  workerUrl: new URL("./audio/brir.worker.js", import.meta.url),
  onRendered: (id, message) => {
    const energy = (ear) => ear.reduce((sum, v) => sum + v * v, 0);
    lastRender = { id, left: energy(message.early[0]), right: energy(message.early[1]), ms: message.ms };
    audioStatus();
  },
  onCell: (id, position) => {
    if (id === state.selected) {
      showPlots(id, position);
      dashboard.show(id, position);
      showMirrorCell(id, position);
    }
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
  if (source && (source.mirrorField || source.mirrorFieldC)) {
    parts.push(state.ab === "mirror" ? "<b>B mirror</b>" : state.ab === "mirror_c" ? "<b>C new</b>" : "A wave");
  }
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
  dashboard.clear();
  const source = sourceById(id);
  if (source && source.cell !== null && source.cell !== undefined) {
    showPlots(id, source.cell);
    dashboard.show(id, source.cell);
    showMirrorCell(id, source.cell);
  } else {
    showMirrorCell(null, null);
  }
  spatial.update(viewport.pose(), audibleIds());
}

// --- A against B: the wave solver's field, or the geometric mirror of it -----------
/** The field a source is heard through under the A-B choice. */
function activeField(source) {
  if (state.ab === "mirror" && source.mirrorField) return source.mirrorField;
  if (state.ab === "mirror_c" && source.mirrorFieldC) return source.mirrorFieldC;
  return source.waveField;
}

function refreshAb() {
  const anyMirror = state.sources.some((s) => s.mirrorField);
  const anyC = state.sources.some((s) => s.mirrorFieldC);
  for (const button of $("#ab").querySelectorAll("button")) {
    button.classList.toggle("on", button.dataset.ab === state.ab);
    if (button.dataset.ab === "mirror") button.disabled = !anyMirror;
    if (button.dataset.ab === "mirror_c") button.disabled = !anyC;
  }
  $("#ab").style.opacity = anyMirror || anyC ? 1 : 0.45;
}

function setAb(mode) {
  if (mode === state.ab) return;
  state.ab = mode;
  refreshAb();
  for (const source of state.sources) {
    const field = activeField(source);
    if (field) spatial.setField(source.id, field);
  }
  plots.clear();
  spatial.update(viewport.pose(), audibleIds());
  audioStatus();
  dashboard.select(mode === "mirror_c" ? "c" : "");
  const source = sourceById(state.selected);
  if (source && source.cell !== null && source.cell !== undefined) dashboard.show(source.id, source.cell);
}
$("#ab").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (button && !button.disabled) setAb(button.dataset.ab);
});

/** The plots of one source's response at a cell, from the field itself. */
function showPlots(id, position) {
  const field = spatial.fieldOf(id);
  if (!field) return;
  const earlySamples = Math.round((settings.earlyMs / 1000) * field.index.sample_rate_hz);
  plots.show(field, position, earlySamples).catch((error) => busy(`${id} plots: ${error.message}`));
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
  if (key === "nearM") followGrid(viewport.pose(), true);
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
let runData = null; // run.json of the run open
// Bumped on every run change, so a grid or a field still loading for the
// old run is dropped rather than attached to the new one.
let generation = 0;

const tierText = (status) => {
  if (!status) return "";
  const tiles = `${status.tiles_drawn}/${status.tiles} tiles · ${(status.quads / 1e6).toFixed(1)} M quads`;
  return `fine ${status.fine_mm.toFixed(2)} mm · ${tiles} · far ${status.coarse_mm.toFixed(2)} mm`;
};

function renderViewButtons() {
  for (const button of $("#view").querySelectorAll("button")) {
    button.classList.toggle("on", button.dataset.view === state.view);
    if (button.dataset.view === "acoustic") button.disabled = !meshViews || !meshViews.bands.length;
    if (button.dataset.view === "mirror") button.disabled = !mirrorLayers.record();
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
  // The band limits belong to the wave view alone.
  $("#fmax").hidden = state.view !== "acoustic";
  $("#fmax-note").hidden = state.view !== "acoustic";
}
$("#view").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (button) setView(button.dataset.view);
});

function setView(view) {
  if (view === "acoustic" && !meshViews) return;
  if (view === "mirror" && !mirrorLayers.record()) return;
  state.view = view;
  if (view === "acoustic") reconcileMesh();
  else $("#hud-tier").textContent = "";
  if (view === "mirror") {
    if (["reflectors", "occluders"].some((n) => mirrorLayers.wanted[n] && !mirrorLayers.loaded(n))) {
      busy("loading the mirror's scene");
      mirrorLayers.ensure().then(() => busy(""), (error) => busy(`mirror audit: ${error.message}`));
    }
  }
  viewport.show(state.view);
  audit.setMode(view);
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
  audit.setWave(runData ? runData.meshes[band] : null, band);
  followGrid(viewport.pose(), true);
}

function followGrid(pose, force = false) {
  if (state.view !== "acoustic" || !grid) return;
  grid.follow(new THREE.Vector3(pose.x, pose.y, pose.z), force);
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
  plots.setHead(pose.yaw);
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
  const previous = state.run;
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
  mirrorLayers.clear();
  plots.clear();
  lastRender = null;
  $("#hud-tier").textContent = "";
  if (state.view !== "colour") setView("colour");
  runData = null;
  audit.setWave(null);
  audit.setMirror(null);
  state.sources = [];
  state.selected = null;
  if (run) {
    const data = await fetch(`${run.url}/run.json`).then((r) => r.json());
    if (state.run !== run) return;
    data.baseUrl = run.url;
    runData = data;
    state.sources = data.sources.map((source, i) => ({
      ...source,
      on: true,
      volume: 0.8,
      loop: true,
      voice: voices.length ? voices[i % voices.length].url : null,
      cell: null,
      waveField: null,
      mirrorField: null,
      mirrorFieldC: null,
    }));
    dashboard.clear();
    state.selected = state.sources.length ? state.sources[0].id : null;
    const mine = generation;
    mirrorLayers
      .load(data)
      .then((check) => {
        if (mine !== generation) return;
        audit.setMirror(check);
        renderViewButtons();
      })
      .catch((error) => busy(`mirror audit: ${error.message}`));
    meshViews = createMeshViews(THREE, data, () => settings.nearM, (status) => {
      $("#hud-tier").textContent = state.view === "acoustic" ? tierText(status) : "";
    });
    for (const source of state.sources.filter((s) => s.field)) {
      engine.setLoop(source.id, true);
      engine.setVolume(source.id, source.volume);
      loadField(`${run.url}/${source.field.url}`)
        .then((field) => {
          if (mine !== generation) return;
          source.waveField = field;
          // The mirror was handed the reference field's lattice as its receivers.
          if (source.mirror || source.mirror_c) mirrorLayers.setReceivers(field.index.positions);
          for (const mirror of [source.mirrorField, source.mirrorFieldC]) {
            if (mirror) plots.shareReference(mirror, field);
          }
          spatial.setField(source.id, activeField(source));
          plots.setReference(field).catch((error) => busy(`${source.id} reference: ${error.message}`));
          points.set(audibleIds().map((id) => spatial.fieldOf(id)).filter(Boolean));
          minimap.setPoints(points.positions());
          spatial.update(viewport.pose(), [source.id]);
        })
        .catch((error) => busy(`${source.id} field: ${error.message}`));
      if (source.mirror) {
        loadField(`${run.url}/${source.mirror.url}`)
          .then((field) => {
            if (mine !== generation) return;
            source.mirrorField = field;
            if (source.waveField) plots.shareReference(field, source.waveField);
            refreshAb();
            if (state.ab === "mirror") {
              spatial.setField(source.id, field);
              spatial.update(viewport.pose(), [source.id]);
            }
          })
          .catch((error) => busy(`${source.id} mirror: ${error.message}`));
      }
      if (source.mirror_c) {
        loadField(`${run.url}/${source.mirror_c.url}`)
          .then((field) => {
            if (mine !== generation) return;
            source.mirrorFieldC = field;
            if (source.waveField) plots.shareReference(field, source.waveField);
            refreshAb();
            if (state.ab === "mirror_c") {
              spatial.setField(source.id, field);
              spatial.update(viewport.pose(), [source.id]);
            }
          })
          .catch((error) => busy(`${source.id} mirror C: ${error.message}`));
      }
      dashboard.load(source.id, source.metrics ? `${run.url}/${source.metrics.url}` : null);
      dashboard.load(source.id, source.metrics_c ? `${run.url}/${source.metrics_c.url}` : null, "c");
    }
    refreshAb();
    // Opening a run is a choice of what to listen to: stand a metre from its
    // first source, facing it, rather than wherever the apartment left us.
    // Not between two runs of one apartment: those are compared from where
    // one stands.
    const sameScene = previous && previous.scene_id === run.scene_id;
    if (state.sources.length && !sameScene) {
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
  engine.setMasterDb(Number(level.value));
  await loadHead(heads.length ? heads[0].name : null);

  const runs = await fetch("runs.json").then((r) => (r.ok ? r.json() : [])).catch(() => []);
  for (const run of runs) {
    if (!runsByScene.has(run.scene_id)) runsByScene.set(run.scene_id, []);
    runsByScene.get(run.scene_id).push(run);
  }
  const lead = runs.find((r) => r.lead) || runs[0] || null;
  // Only what opens without assembling is offered at all: minutes of work
  // behind a click. The server puts the one to open first at the head.
  const apartments = (await fetch("apartments.json").then((r) => r.json())).filter((a) => a.ready);
  for (const apartment of apartments) apartmentsById.set(apartment.local, apartment);
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
  grid: () => grid,
};
