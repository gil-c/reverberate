/** Scene mode: a recipe drawn and replayed; `sound.js` makes it heard.
 *
 * A recipe says where every source and the listener are over twenty minutes.
 * This mode generates one from the generator's ranges and a seed, or loads a
 * saved one, and replays it: markers in the 3D view and on the plan, a
 * timeline with a lane each. The page computes nothing about a scene: the
 * recipe, its summary, its violations and the sampled tracks all come from
 * the server (`viz/scene_api.py`), which asks `reverberate.scenes`. A recipe
 * goes back to the server as the text it came in (`report.canonical`), never
 * as the tree drawn from: a browser would write its `12.0` as `12`.
 *
 * **The seam for sound.** `transport` is the scene's clock and `mix` says
 * which sources are heard; both are exported on the mode and neither knows
 * the page (`transport.js`). An audio stage listens to `play`, `pause`, `seek`
 * and `speed`, may hand in its own clock with `transport.setClock`, and reads
 * `mix.audible(id)`; `onScene` tells it when another recipe is on show and
 * `onSelect` which source is picked. `sound.js` is that stage. It may also put
 * a scene on show without its recipe (`showTracks`), from a pack's own
 * tables, and draws its state on the timeline (`timelineSound`).
 */
import { createGeneratorPanel, createSummary } from "./generator.js";
import { createSceneMarkers } from "./markers.js";
import { createTimeline } from "./timeline.js";
import { activeAt, heading, listenerAt, sourceAt, trail, trailCapacity, viewYaw } from "./tracks.js";
import { createMix, createTransport } from "./transport.js";

const RAD = Math.PI / 180;
// The step the tracks are asked at, seconds: a straight line between two
// samples is then within 0.075 m of a source walking at the format's 1.5 m/s.
const STEP_S = 0.1;
const LISTENER = "listener";

async function api(path, body) {
  const options =
    body === undefined
      ? undefined
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(`api/scene/${path}`, options);
  const answer = await response.json();
  if (!response.ok) {
    const error = new Error(answer.error || `${response.status}`);
    error.status = response.status;
    throw error;
  }
  return answer;
}

export function createSceneMode({ THREE, viewport, minimap, elements, busy }) {
  const transport = createTransport();
  const mix = createMix();
  const markers = createSceneMarkers(THREE, viewport);
  markers.group.visible = false;
  viewport.overlays.add(markers.group);
  minimap.showScene(false);

  let enabled = false;
  let dwelling = null;
  let schema = null;
  let layout = null;
  let report = null;
  let tracks = null;
  let selected = null;
  let follow = false;
  let trailSeconds = 10;
  let buffer = new Float32Array(3);
  // Bumped whenever the dwelling or the recipe changes, so an answer that
  // comes back late is dropped and not drawn over its successor.
  let generation = 0;
  const sceneHandlers = [];
  const selectHandlers = [];

  const panel = createGeneratorPanel(elements.panel, {
    onGenerate: () => generate(),
    onSave: (name) => save(name),
    onLoad: (name) => load(name),
  });
  const summary = createSummary(elements.summary);
  const timeline = createTimeline(elements.timeline, {
    transport,
    mix,
    onSelect: (id) => select(id),
    onFollow: (on) => setFollow(on),
    onTrail: (seconds) => {
      trailSeconds = seconds;
      sizeTrails();
      frame();
    },
  });

  // --- drawing one instant ------------------------------------------------
  function sizeTrails() {
    if (!tracks) return;
    const capacity = trailCapacity(trailSeconds, tracks.step_s);
    buffer = new Float32Array(3 * capacity);
    markers.setActors([...tracks.sources, { id: LISTENER, kind: LISTENER }], capacity);
  }

  function frame() {
    if (!enabled || !tracks) return;
    const { t } = transport.state();
    const states = [];
    for (const track of tracks.sources) {
      const at = sourceAt(tracks, track, t);
      const points = trail(tracks, track, t, trailSeconds, buffer);
      const emitting = activeAt(track.activity, t);
      const audible = mix.audible(track.id);
      markers.place(track.id, at, buffer, points);
      minimap.moveActor(track.id, {
        x: at.x,
        z: at.z,
        facing: heading(at.yaw_deg),
        active: emitting,
        dim: !audible,
        selected: track.id === selected,
        trail: buffer,
        points,
      });
      // A source that is not heard is not drawn in the room; one that is
      // heard but silent now is drawn dark.
      states.push({ id: track.id, on: audible, dim: !emitting });
    }
    const head = listenerAt(tracks, t);
    const points = trail(tracks, tracks.listener, t, trailSeconds, buffer);
    markers.place(LISTENER, head, buffer, points);
    minimap.moveActor(LISTENER, {
      x: head.x,
      z: head.z,
      facing: heading(head.yaw_deg),
      active: true,
      selected: selected === LISTENER,
      trail: buffer,
      points,
    });
    // Seen through the listener's eyes, its own head is not in the picture.
    states.push({ id: LISTENER, on: !follow, dim: false });
    markers.update(states, selected);
    if (follow) {
      viewport.moveTo({
        x: head.x,
        y: head.y,
        z: head.z,
        yaw: viewYaw(head.yaw_deg),
        pitch: head.pitch_deg * RAD,
        roll: head.roll_deg * RAD,
        placed: true,
      });
    }
    timeline.draw();
    viewport.invalidate();
  }

  let raf = null;
  const loop = () => {
    raf = null;
    transport.tick();
    if (transport.state().playing) raf = requestAnimationFrame(loop);
  };
  transport.on("play", () => {
    if (raf === null) raf = requestAnimationFrame(loop);
  });
  transport.on("tick", frame);
  mix.on("change", frame);

  // --- what is on show ------------------------------------------------------
  function drawGround() {
    if (!layout) {
      minimap.setScene(null);
      markers.clearGround();
      return;
    }
    const recipe = report ? report.recipe : null;
    const used = {
      stations: new Set(recipe ? recipe.stations.map((station) => station.id) : []),
      rails: new Set(recipe ? recipe.rails.map((rail) => rail.id) : []),
    };
    // The recipe's own stations and rails are drawn as it wrote them; the
    // dwelling's others, which it leaves alone, are drawn under them.
    const rest = {
      stations: layout.stations.filter((station) => !used.stations.has(station.id)),
      rails: layout.rails.filter((rail) => !used.rails.has(rail.id)),
    };
    const taken = { stations: recipe ? recipe.stations : [], rails: recipe ? recipe.rails : [] };
    minimap.setScene({
      free: layout.free,
      stations: [...rest.stations, ...taken.stations],
      rails: [...rest.rails, ...taken.rails],
      used,
    });
    markers.setGround({ floor_y_m: layout.dwelling.floor_y_m, rest, used: taken });
  }

  function clear() {
    generation += 1;
    transport.load(0);
    report = null;
    tracks = null;
    selected = null;
    mix.setSources([]);
    timeline.setTracks(null);
    timeline.setSelected(null);
    minimap.setActors([]);
    markers.setActors([], 1);
    panel.canSave(false);
    for (const handler of sceneHandlers) handler(null);
  }

  /** Put a recipe on show: fetch its tracks, lay out the lanes, rest at zero. */
  async function show(next) {
    const mine = ++generation;
    const sampled = await api("tracks", { canonical: next.canonical, step_s: STEP_S });
    if (mine !== generation) return;
    put(sampled, next);
  }

  /** Tracks on show, with the recipe's report when there is one: a pack's own
   *  tracks have none, and nothing of them can be saved. */
  function put(sampled, next) {
    report = next;
    tracks = sampled;
    selected = null;
    // The marks first: the mix announces itself, and that draws a frame.
    minimap.setActors([...tracks.sources, { id: LISTENER, kind: LISTENER }]);
    sizeTrails();
    mix.setSources(tracks.sources.map((track) => track.id));
    drawGround();
    timeline.setTracks(tracks);
    timeline.setSelected(null);
    if (report) summary.show(report);
    panel.canSave(Boolean(report));
    transport.load(tracks.duration_s);
    for (const handler of sceneHandlers) handler({ report, tracks });
    frame();
  }

  async function generate(options = {}) {
    if (!dwelling) return;
    const seed = options.seed === undefined ? panel.seed() : options.seed;
    const parameters = options.parameters || panel.values();
    panel.setBusy(true);
    panel.setStatus("generating");
    try {
      const started = performance.now();
      const answer = await api("generate", { dwelling, seed, parameters });
      await show(answer);
      panel.setSeed(seed);
      if (answer.parameters) panel.setValues(answer.parameters);
      panel.setName(`seed_${seed}`);
      panel.setStatus(`generated in ${((performance.now() - started) / 1000).toFixed(1)} s, not saved`);
    } catch (error) {
      panel.setStatus(error.message, true);
    } finally {
      panel.setBusy(false);
    }
  }

  async function load(name) {
    panel.setStatus(`loading ${name}`);
    try {
      const answer = await api(`recipes/${dwelling}/${name}`);
      await show(answer);
      // The panel goes back to what the recipe was drawn from.
      panel.setSeed(answer.recipe.seed);
      if (answer.parameters) panel.setValues(answer.parameters);
      panel.setName(name);
      panel.setStatus(answer.parameters ? `${name}: the panel is the recipe's` : `${name}: no generator block to restore`);
    } catch (error) {
      panel.setStatus(error.message, true);
    }
  }

  async function save(name, overwrite = false) {
    if (!report) return;
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) {
      panel.setStatus("a name is letters, digits, _ and -", true);
      return;
    }
    try {
      const answer = await api(`recipes/${dwelling}/${name}`, { canonical: report.canonical, overwrite });
      panel.setSaved(await api(`recipes/${dwelling}`), name);
      panel.setStatus(`saved ${answer.path}`);
    } catch (error) {
      if (error.status === 409 && !overwrite && confirm(`${name} exists. Replace it?`)) {
        await save(name, true);
        return;
      }
      panel.setStatus(error.message, true);
    }
  }

  /** Read what a dwelling offers: its layout and its saved recipes. */
  async function open() {
    const mine = generation;
    const name = dwelling;
    busy("reading the dwelling's layout");
    try {
      if (!schema) {
        schema = await api("schema");
        panel.setSchema(schema);
      }
      const [found, saved] = await Promise.all([api(`layout/${name}`), api(`recipes/${name}`)]);
      if (mine !== generation || name !== dwelling) return;
      layout = found;
      panel.setSaved(saved);
      panel.setStatus(layout.summary);
      summary.showLayout(layout);
      drawGround();
      busy("");
    } catch (error) {
      busy("");
      panel.setStatus(error.message, true);
      summary.showLayout(null);
    }
    viewport.invalidate();
  }

  function select(id) {
    selected = selected === id ? null : id;
    timeline.setSelected(selected);
    frame();
    for (const handler of selectHandlers) handler(selected);
  }

  function setFollow(on) {
    follow = Boolean(on);
    timeline.setFollow(follow);
    // Leaving the listener's eyes leaves its tilt behind too.
    if (!follow) viewport.moveTo({ roll: 0 });
    frame();
  }

  function setEnabled(on) {
    enabled = Boolean(on);
    elements.toggle.checked = enabled;
    elements.panel.hidden = !enabled;
    elements.timeline.hidden = !enabled;
    elements.summary.hidden = !enabled;
    elements.audio.hidden = enabled;
    markers.group.visible = enabled;
    minimap.showScene(enabled);
    if (!enabled) {
      transport.pause();
      if (follow) setFollow(false);
    }
    viewport.resize();
    if (enabled && dwelling && !layout) open();
    frame();
    viewport.invalidate();
  }

  elements.toggle.addEventListener("change", () => setEnabled(elements.toggle.checked));
  addEventListener("keydown", (event) => {
    if (!enabled || event.code !== "Space") return;
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)) return;
    event.preventDefault();
    transport.toggle();
  });

  return {
    transport,
    mix,
    /** Another apartment is open: its scene, if any, is gone. */
    setDwelling(name) {
      if (name === dwelling && layout) {
        // The same apartment reopened, for another run: the plan was redrawn
        // to its scale and the marks follow it.
        frame();
        return;
      }
      dwelling = name;
      layout = null;
      clear();
      drawGround();
      summary.showLayout(null);
      if (enabled) open();
    },
    setEnabled,
    generate,
    load,
    save,
    select,
    setFollow,
    /** `handler({ report, tracks })` when a recipe is put on show, `handler(null)` when it goes. */
    onScene: (handler) => sceneHandlers.push(handler),
    /** `handler(id)` when a source or the listener is picked, `handler(null)` when none is. */
    onSelect: (handler) => selectHandlers.push(handler),
    /** Put on show tracks that came from elsewhere than a recipe: a pack's own. */
    showTracks(sampled) {
      generation += 1;
      put(sampled, null);
    },
    /** What the sound's stage draws on the timeline (`timeline.setSound`), or null. */
    timelineSound: (provider) => timeline.setSound(provider),
    refreshTimeline: () => timeline.refresh(),
    enabled: () => enabled,
    selected: () => selected,
    report: () => report,
    tracks: () => tracks,
    layout: () => layout,
    values: () => panel.values(),
    timeline: () => timeline.window(),
  };
}
