/** A pack's geometry as the computation used it, drawn over the scene or instead of it.
 *
 * The scene view shows the furnished model. No engine computed on it. This
 * panel shows, layer by layer and each under the key of the asset it is,
 * what they did compute on: the wave solver's staircase (its walls about the
 * head and a whole slice of nodes at a chosen height), the nodes each
 * listening array and each source stood on, the mirror's facets, occluders
 * and edges, a sample of the tail's rays, the image paths through the
 * facets they bounce on, and which cells serve the head now. With a second
 * pack of the same recipe it shows what their two grids disagree on, and
 * switches the sound between them at the same instant.
 *
 * Everything comes from `viz/computed_api.py`; the numbers are turned into
 * geometry by `computed-data.js`. Each answer says whether it is the thing
 * itself, a derived view or a sample, and the panel repeats it.
 */
import { sameScene, variantName } from "./variants.js";
import { orderColour } from "../mirror.js";
import {
  COLOURS,
  decayPoints,
  diffLines,
  energyColour,
  facetColourer,
  facetsHit,
  materialColour,
  mm,
  openingLine,
  pathSegments,
  quadMesh,
  raySegments,
  shortKey,
  sliceRect,
  sliceRgba,
  triangleColours,
  unpack,
} from "./computed-data.js";
import { listenerAt } from "./tracks.js";

const LISTENER = "listener";
const POLL_MS = 500;

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

async function ask(path, query) {
  const response = await fetch(`api/computed/${path}?${new URLSearchParams(query)}`);
  if (response.headers.get("X-Computed")) {
    const meta = JSON.parse(response.headers.get("X-Computed"));
    return { meta, arrays: unpack(await response.arrayBuffer(), meta) };
  }
  const answer = await response.json();
  if (!response.ok) throw new Error(answer.error || `${response.status}`);
  return answer;
}

export function createComputed({ THREE, viewport, minimap, scene, sound, root }) {
  const group = new THREE.Group();
  group.name = "as computed";
  viewport.overlays.add(group);

  // --- the panel ----------------------------------------------------------------
  const select = (title) => Object.assign(make("select"), { title });
  const number = (value, min, max, step, title) => Object.assign(make("input"), { type: "number", value, min, max, step, title });
  const field = (label, control, unit = "") => {
    const row = make("label", "field");
    row.append(make("span", "", label), control, make("em", "", unit));
    return row;
  };
  const packA = make("div", "note");
  const pickB = select("a second pack of the same scene, to compare with");
  const listen = make("div", "seg sub");
  for (const which of ["A", "B"]) listen.append(Object.assign(make("button", "", `hear ${which}`), { type: "button", value: which }));
  const gridA = select("the grid drawn for A");
  const gridB = select("the grid drawn for B");
  const height = number(1.7, -5, 50, 0.01, "the height of the slice, m; the layer of nodes nearest it is shown");
  const radius = number(3, 0.5, 8, 0.5, "how far about the head the walls are drawn, m");
  const sliceOf = select("which slice is on the plan and in the room");
  for (const [value, label] of [["a", "slice of A"], ["b", "slice of B"], ["diff", "A against B"]]) sliceOf.append(new Option(label, value));
  const colourBy = select("how the mirror's facets are coloured");
  for (const [value, label] of [["material", "by material"], ["absorption", "by absorption"], ["sheets", "two-sided and sheets"]]) colourBy.append(new Option(label, value));
  const band = select("the octave band of absorption and of a ray's energy");
  const source = select("the source whose paths, rays and low band are shown");
  const hist = select("the histogram: a site of the source and a cell of the listener");
  const rayCount = number(100, 10, 600, 10, "rays traced again; about 3 s a hundred");
  const here = Object.assign(make("button", "", "walls here"), { type: "button", title: "draw the walls about where the head is now" });
  const trace = Object.assign(make("button", "", "trace sample"), { type: "button" });
  const switches = make("div", "fields");
  const LAYERS = [
    ["walls", "wave grid: walls"],
    ["slice", "wave grid: slice"],
    ["snapped", "arrays and sources on nodes"],
    ["difference", "A against B: walls and air"],
    ["facets", "mirror: reflecting facets"],
    ["occluders", "mirror: occluders (rays)"],
    ["edges", "mirror: diffracting edges"],
    ["paths", "image paths through facets"],
    ["tail", "tail: sites, cells, histogram"],
    ["rays", "tail: a sample of rays"],
    ["low", "low band at this instant"],
    ["bare", "hide the furnished model"],
  ];
  const on = {};
  const boxes = {};
  for (const [name, label] of LAYERS) {
    const row = make("label", "field switch");
    const box = Object.assign(make("input"), { type: "checkbox" });
    box.addEventListener("change", () => {
      on[name] = box.checked;
      refresh(name);
    });
    boxes[name] = box;
    row.append(make("span", "", label), box, make("em"));
    switches.append(row);
  }
  const plot = Object.assign(make("canvas", "plot"), { width: 600, height: 200 });
  plot.hidden = true;
  const notes = {};
  const noteBox = make("div", "stack computed-notes");
  const say = (name, lines, bad = false) => {
    if (!notes[name]) {
      notes[name] = make("div", "note");
      notes[name].dataset.about = name;
      noteBox.append(notes[name]);
    }
    notes[name].textContent = [].concat(lines || []).join("\n");
    notes[name].classList.toggle("bad", bad);
    notes[name].hidden = !notes[name].textContent;
  };
  const picks = make("div", "fields");
  picks.append(
    field("B", pickB), field("grid A", gridA), field("grid B", gridB), field("height", height, "m"),
    field("about", radius, "m"), field("slice", sliceOf), field("facets", colourBy), field("band", band, "Hz"),
    field("source", source), field("tail", hist), field("rays", rayCount)
  );
  const buttons = make("div", "seg sub");
  buttons.append(here, trace);
  root.append(packA, listen, picks, switches, buttons, plot, noteBox);

  // --- state --------------------------------------------------------------------
  let about = null; // of pack A, the pack that is heard unless B is
  let aboutB = null;
  let idA = null;
  let heard = "A";
  let catalogue = null;
  let stamp = 0; // bumped when the pack changes: late answers are dropped
  const drawn = {}; // layer -> three objects
  const cache = { paths: new Map(), low: new Map() };
  let drawnStep = "";
  const failed = (name) => (error) => say(name, `${name}: ${error.message}`, true);

  function drop(name) {
    for (const object of drawn[name] || []) {
      group.remove(object);
      object.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (child.material.map) child.material.map.dispose();
          child.material.dispose();
        }
      });
    }
    drawn[name] = [];
    viewport.invalidate();
  }
  function put(name, ...objects) {
    for (const object of objects) {
      object.frustumCulled = false;
      group.add(object);
      drawn[name].push(object);
    }
    viewport.invalidate();
  }

  const head = () => {
    const tracks = scene.tracks();
    return tracks ? listenerAt(tracks, scene.transport.time()) : viewport.pose();
  };
  const step = () => (about ? Math.min(about.steps - 1, Math.max(0, Math.floor(scene.transport.time() / about.step_s + 1e-9))) : 0);
  const sourceId = () => {
    const chosen = scene.selected();
    return chosen && chosen !== LISTENER ? chosen : source.value;
  };

  // --- meshes -------------------------------------------------------------------
  function surface(mesh, opacity = 0.85) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(mesh.colours, 3));
    if (mesh.index) geometry.setIndex(new THREE.BufferAttribute(mesh.index, 1));
    geometry.computeVertexNormals();
    return new THREE.Mesh(
      geometry,
      new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: opacity < 1, opacity })
    );
  }
  function lines(positions, colours, hex) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (colours) geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
    const material = new THREE.LineBasicMaterial(colours ? { vertexColors: true } : { color: hex });
    material.depthTest = false;
    const object = new THREE.LineSegments(geometry, material);
    object.renderOrder = 5;
    return object;
  }
  function dots(positions, hex, size = 0.02) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
    const object = new THREE.Points(geometry, new THREE.PointsMaterial({ color: hex, size, depthTest: false }));
    object.renderOrder = 6;
    return object;
  }
  function ball(at, r, hex, wire = false) {
    const object = new THREE.Mesh(
      new THREE.SphereGeometry(r, wire ? 16 : 10, wire ? 10 : 8),
      new THREE.MeshBasicMaterial({ color: hex, wireframe: wire, depthTest: false, transparent: wire, opacity: wire ? 0.5 : 1 })
    );
    object.position.set(at[0], at[1], at[2]);
    object.renderOrder = 6;
    return object;
  }

  // --- the wave grid ------------------------------------------------------------
  const gridOf = (which) => (which === "a" ? gridA.value : gridB.value);
  const gridLabel = (told, which) => {
    const offered = [...(told ? told.grid.offered : []), ...(aboutB ? aboutB.grid.offered : [])];
    const entry = offered.find((one) => one.id === gridOf(which));
    return entry ? `${shortKey(entry.key)}${entry.own ? "" : " (NOT this pack's grid)"}, ${entry.from}` : "none";
  };

  async function drawWalls() {
    drop("walls");
    if (!on.walls || !about || !gridA.value) return say("walls", on.walls && about ? `wave grid of A: ${about.grid.why_none}` : "");
    const mine = stamp;
    const { x, y, z } = head();
    const [answer, facts] = await Promise.all([
      ask("grid/walls", { pack: idA, grid: gridA.value, x, y, z, r: radius.value }),
      ask("grid/facts", { pack: idA, grid: gridA.value }),
    ]);
    if (mine !== stamp || !on.walls) return;
    put("walls", surface(quadMesh(answer.arrays.corners, answer.arrays.material, materialColour)));
    const [low, high] = facts.absorbing_layer_m;
    const box = new THREE.Box3(new THREE.Vector3(...low), new THREE.Vector3(...high));
    put("walls", new THREE.Box3Helper(box, 0x8a8f98));
    say("walls", [
      `wave grid of A: ${gridLabel(about, "a")}`,
      `step ${mm(facts.step_m)}, ${facts.shape.join(" x ")} = ${facts.nodes.toLocaleString()} nodes, ${facts.reached.toLocaleString()} reached, ${facts.boundary_nodes.toLocaleString()} boundary (${facts.rigid_boundary_nodes.toLocaleString()} rigid)`,
      `walls: ${answer.meta.faces.toLocaleString()} cut links as ${answer.meta.quads.toLocaleString()} rectangles, within ${radius.value} m of the head. EXACT where drawn: ${answer.meta.exact}`,
      "grey box: the absorbing layer, one node inside the grid's box",
    ]);
  }

  let sliceCanvas = null;
  async function drawSlice() {
    drop("slice");
    minimap.setUnderlay(null);
    if (!on.slice || !about) return say("slice", "");
    const which = sliceOf.value;
    const mine = stamp;
    let answer;
    if (which === "diff") {
      if (!gridA.value || !gridB.value) return say("slice", "A against B needs a grid for each", true);
      answer = await ask("diff/slice", { pack: idA, a: gridA.value, b: gridB.value, y: height.value });
    } else {
      if (!gridOf(which)) return say("slice", `no grid for ${which.toUpperCase()}`, true);
      answer = await ask("grid/slice", { pack: idA, grid: gridOf(which), y: height.value });
    }
    if (mine !== stamp || !on.slice) return;
    const [rows, columns] = answer.meta.arrays[0].shape;
    sliceCanvas = sliceCanvas || document.createElement("canvas");
    sliceCanvas.width = columns;
    sliceCanvas.height = rows;
    sliceCanvas.getContext("2d").putImageData(new ImageData(sliceRgba(answer.arrays.classes, which === "diff" ? "diff" : "grid"), columns, rows), 0, 0);
    const rect = sliceRect(answer.meta, [rows, columns]);
    minimap.setUnderlay({ url: sliceCanvas.toDataURL(), ...rect });
    const texture = new THREE.CanvasTexture(sliceCanvas);
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.colorSpace = THREE.SRGBColorSpace;
    // The canvas's first row is the lowest z: the plane is laid flat with z down the picture.
    texture.flipY = false;
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(rect.x1 - rect.x0, rect.z1 - rect.z0),
      new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide, transparent: true, opacity: 0.8 })
    );
    plane.rotation.x = Math.PI / 2;
    plane.position.set((rect.x0 + rect.x1) / 2, answer.meta.y_m, (rect.z0 + rect.z1) / 2);
    put("slice", plane);
    say("slice", [
      which === "diff"
        ? `slice A against B at ${answer.meta.y_m.toFixed(3)} m. DERIVED: ${answer.meta.derived}. Orange: air only in A; blue: air only in B`
        : `slice of ${which.toUpperCase()} (${gridLabel(about, which)}) at ${answer.meta.y_m.toFixed(3)} m, layer ${answer.meta.layer}: ${columns} x ${rows} nodes. EXACT: ${answer.meta.exact}. Dark: not reached; light: air; colours: boundary nodes by material`,
    ]);
  }

  async function drawDifference() {
    drop("difference");
    if (!on.difference || !about) return say("difference", "");
    if (!gridA.value || !gridB.value || gridA.value === gridB.value) return say("difference", "A against B needs two different grids: choose a pack B, or a grid B", true);
    const mine = stamp;
    say("difference", "comparing the two grids");
    const { x, y, z } = head();
    const where = { x, y, z, r: radius.value };
    const [numbers, skin, wallsB, openings] = await Promise.all([
      ask("diff", { pack: idA, a: gridA.value, b: gridB.value }),
      ask("diff/walls", { pack: idA, a: gridA.value, b: gridB.value, ...where }),
      ask("grid/walls", { pack: idA, grid: gridB.value, ...where }),
      ask("diff/openings", { pack: idA, a: gridA.value, b: gridB.value, y: height.value }),
    ]);
    if (mine !== stamp || !on.difference) return;
    put("difference", surface(quadMesh(skin.arrays.corners, skin.arrays.value, (value) => (value ? COLOURS.onlyB : COLOURS.onlyA)), 0.9));
    put("difference", surface(quadMesh(wallsB.arrays.corners, wallsB.arrays.material, () => COLOURS.wallB), 0.45));
    for (const row of openings.changed.slice(0, 40)) put("difference", ball([row.centre_m[0], openings.layer_a_m, row.centre_m[1]], 0.05, 0xff3ca0));
    say("difference", [
      `A ${gridLabel(about, "a")}`,
      `B ${gridLabel(about, "b")}`,
      ...diffLines(numbers),
      `orange: air only in A; blue: air only in B; pale blue: B's walls (switch A's walls on to see both), within ${radius.value} m of the head`,
      `openings at ${openings.y_m} m: ${openings.openings_a} in A, ${openings.openings_b} in B, ${openings.changed.length} changed (pink). DERIVED: ${openings.rule}`,
      ...openings.changed.slice(0, 12).map(openingLine),
    ]);
  }

  async function drawSnapped() {
    drop("snapped");
    if (!on.snapped || !about) return say("snapped", "");
    const mine = stamp;
    const segments = [];
    for (const cell of about.cells) {
      put("snapped", ball(cell.centre_m, 0.012, 0x39d98a));
      if (cell.asked_m) {
        put("snapped", ball(cell.asked_m, 0.008, 0xffffff));
        segments.push(...cell.asked_m, ...cell.centre_m);
      }
    }
    for (const solved of about.solved_sources) {
      put("snapped", ball(solved.position_m, 0.008, 0xffffff), dots(solved.nodes_m.flat(), 0xffd24a, 0.015));
      const heaviest = solved.nodes_m[solved.weights.indexOf(Math.max(...solved.weights))];
      segments.push(...solved.position_m, ...heaviest);
    }
    if (segments.length) put("snapped", lines(new Float32Array(segments), null, 0xff3ca0));
    // The nodes of the arrays that serve the head now, or of all when they are few.
    const now = cache.low.get(`${sourceId()}:${step()}`);
    const wanted = about.cells.length <= 6 ? about.cells.map((cell) => cell.cell) : now ? now.cells.map((cell) => cell.cell) : [];
    const moved = about.cells.filter((cell) => cell.moved_mm !== null).map((cell) => cell.moved_mm);
    say("snapped", [
      about.snapped.from ? `arrays and sources: ${about.snapped.from}` : `arrays: centres from the pack; ${about.snapped.why_none}`,
      `${about.cells.length} cells; green: the node an array is about; white: the place asked for; pink: from one to the other` +
        (moved.length ? `, ${Math.min(...moved).toFixed(1)} to ${Math.max(...moved).toFixed(1)} mm` : ""),
      about.solved_sources.length
        ? `${about.solved_sources.length} solved source positions, each spread on 8 nodes (yellow); nearest node ${Math.max(...about.solved_sources.map((s) => s.nearest_mm)).toFixed(1)} mm away at most`
        : "",
      ...about.cells.slice(0, 8).map((cell) => `cell ${cell.cell}: ${cell.moved_mm === null ? "asked place not known" : `${cell.moved_mm.toFixed(1)} mm from asked`}${cell.nodes ? `, ${cell.nodes} nodes` : ""}, clearance ${cell.clearance_m} m`),
    ].filter(Boolean));
    if (about.snapped.step_m === null) return;
    for (const cell of wanted) {
      const answer = await ask("grid/array", { pack: idA, cell }).catch(() => null);
      if (mine !== stamp || !on.snapped) return;
      if (answer) put("snapped", dots(answer.arrays.nodes, 0x39d98a, 0.012));
    }
  }

  // --- the mirror ---------------------------------------------------------------
  async function ensureCatalogue() {
    if (!catalogue) {
      catalogue = await ask("mirror/facets", { pack: idA });
      band.replaceChildren(...catalogue.bands_hz.map((hz, index) => new Option(hz, index)));
      band.value = String(Math.max(0, catalogue.bands_hz.indexOf(1000)));
    }
    return catalogue;
  }
  const mirrorLine = (told) => `mirror scene ${shortKey(told.key)}${told.is_the_packs ? "" : ` (NOT the key the pack names, ${shortKey(told.pack_names)})`}: ${told.summary}`;

  async function drawFacets() {
    drop("facets");
    if (!on.facets || !about) return say("facets", "");
    const mine = stamp;
    const told = await ensureCatalogue();
    const answer = await ask("mirror/triangles", { pack: idA, layer: "reflectors" });
    if (mine !== stamp || !on.facets) return;
    const colourer = facetColourer(told, colourBy.value, Number(band.value));
    put("facets", surface({ positions: answer.arrays.vertices, colours: triangleColours(answer.arrays.facet, colourer) }, 0.8));
    const twoSided = told.facets.filter((facet) => facet.two_sided).length;
    say("facets", [
      mirrorLine(told),
      `EXACT: ${told.exact}. ${told.facets.length} facets, ${twoSided} two-sided, ${told.sheets} of them layers of a sheet the model holds twice (read ${told.coincident_facets})`,
      colourBy.value === "sheets" ? "pink: a layer of a doubled sheet; yellow: two-sided; grey: one-sided" : colourBy.value === "absorption" ? `absorption at ${told.bands_hz[Number(band.value)]} Hz: light and cold is hard, dark and warm is soft` : "one colour a material",
    ]);
  }

  async function drawOccluders() {
    drop("occluders");
    if (!on.occluders || !about) return say("occluders", "");
    const mine = stamp;
    const told = await ensureCatalogue();
    say("occluders", `loading ${told.occluder_triangles.toLocaleString()} occluder triangles`);
    const answer = await ask("mirror/triangles", { pack: idA, layer: "occluders" });
    if (mine !== stamp || !on.occluders) return;
    const colourer = colourBy.value === "absorption" ? (label) => facetColourer({ facets: [{ label }], absorption: told.absorption }, "absorption", Number(band.value))(0) : materialColour;
    put("occluders", surface({ positions: answer.arrays.vertices, colours: triangleColours(answer.arrays.label, colourer) }, 1));
    say("occluders", `occluders: ${told.occluder_triangles.toLocaleString()} triangles. EXACT: ${answer.meta.exact}`);
  }

  async function drawEdges() {
    drop("edges");
    if (!on.edges || !about) return say("edges", "");
    const mine = stamp;
    const answer = await ask("mirror/edges", { pack: idA });
    if (mine !== stamp || !on.edges) return;
    const positions = new Float32Array(6 * answer.meta.edges);
    for (let k = 0; k < answer.meta.edges; k++) {
      positions.set(answer.arrays.a.subarray(3 * k, 3 * k + 3), 6 * k);
      positions.set(answer.arrays.b.subarray(3 * k, 3 * k + 3), 6 * k + 3);
    }
    put("edges", lines(positions, null, 0xfff04a));
    say("edges", `diffracting edges: ${answer.meta.edges}, ${answer.meta.length_m_total} m. EXACT: ${answer.meta.exact}`);
  }

  async function drawStep() {
    if (!about) return;
    const id = sourceId();
    const at = step();
    const name = `${id}:${at}:${on.paths}:${on.low}`;
    if (name === drawnStep || !id) return;
    drawnStep = name;
    const mine = stamp;
    for (const [layer, path] of [["paths", "paths"], ["low", "low"]]) {
      if (!on[layer]) {
        drop(layer);
        say(layer, "");
        continue;
      }
      const key = `${id}:${at}`;
      if (!cache[layer].has(key)) {
        cache[layer].set(key, await ask(path, { pack: idA, source: id, step: at }).catch((error) => ({ error: error.message })));
        if (cache[layer].size > 200) cache[layer].delete(cache[layer].keys().next().value);
      }
      if (mine !== stamp || drawnStep !== name) return;
      const answer = cache[layer].get(key);
      drop(layer);
      if (answer.error) {
        say(layer, `${layer}: ${answer.error}`, true);
      } else if (layer === "paths") {
        const held = pathSegments(answer.rows);
        const colours = new Float32Array(2 * held.positions.length);
        const colour = new THREE.Color();
        held.orders.forEach((order, k) => {
          colour.setHex(orderColour(order));
          colours.set([colour.r, colour.g, colour.b, colour.r, colour.g, colour.b], 6 * k);
        });
        if (held.orders.length) put("paths", lines(held.positions, colours), dots(held.corners.flatMap((corner) => corner.at), 0xffffff, 0.04));
        const hit = facetsHit(answer.rows);
        say("paths", [
          `image paths of ${id} at step ${at}: ${answer.rows.length} rows in the pack, ${answer.found} drawn through their facets, ${answer.not_found} not found again` +
            (answer.not_found ? ` (pack traced by code ${shortKey(answer.pack_code_version)})` : "") +
            (answer.not_in_the_pack ? `, ${answer.not_in_the_pack} this code finds that the pack does not hold` : ""),
          `RECOMPUTED: ${answer.recomputed}. Worst delay difference ${answer.worst_delay_difference_s.toExponential(1)} s`,
          hit.length ? `facets hit: ${hit.slice(0, 10).map(([facet, count]) => `${facet} (${catalogue ? catalogue.labels[catalogue.facets[facet].label] : "?"}) x${count}`).join(", ")}` : "",
        ].filter(Boolean));
      } else {
        const segments = [];
        for (const cell of answer.cells) {
          segments.push(...answer.listener_m, ...cell.centre_m);
          put("low", ball(cell.centre_m, 0.03, 0x39d98a));
          for (const pair of cell.pairs) put("low", ball(pair.position_m, 0.03, 0xffd24a));
        }
        if (segments.length) put("low", lines(new Float32Array(segments), null, 0x39d98a));
        say("low", [
          `low band of ${id} at step ${at}: ${answer.mode_name}. EXACT: ${answer.exact}`,
          ...answer.cells.map((cell) => `cell ${cell.cell}, head ${cell.translation_mm} mm from its centre; ${cell.pairs.map((pair) => `source position ${pair.row} (weight ${pair.weight}, ${pair.from_source_mm} mm from the source)`).join(", ")}`),
        ]);
      }
    }
    if (on.snapped && about.cells.length > 6) drawSnapped();
  }

  // --- the tail -----------------------------------------------------------------
  function fillHist() {
    const rows = about ? about.tails[source.value] || [] : [];
    hist.replaceChildren(...rows.map((row) => new Option(`site ${row.row} at ${row.site_m.map((v) => v.toFixed(2)).join(", ")} · cell ${row.cell}`, row.row)));
  }

  async function drawTail() {
    drop("tail");
    plot.hidden = true;
    if (!on.tail || !about) return say("tail", "");
    const rows = about.tails[source.value] || [];
    if (!rows.length) return say("tail", `${source.value || "no source"} has no tail`, true);
    const mine = stamp;
    const cells = new Set();
    for (const row of rows) {
      put("tail", ball(row.site_m, 0.05, 0xff7a1a));
      cells.add(row.cell);
    }
    for (const cell of cells) put("tail", ball(about.cells[cell].centre_m, about.receiver_radius_m, 0x4fd1e8, true));
    const answer = await ask("tail", { pack: idA, source: source.value, row: hist.value || 0 });
    if (mine !== stamp || !on.tail) return;
    plot.hidden = false;
    const context = plot.getContext("2d");
    context.clearRect(0, 0, plot.width, plot.height);
    answer.energy.forEach((values, index) => {
      const [r, g, b] = materialColour(index);
      context.strokeStyle = `rgb(${r},${g},${b})`;
      context.lineWidth = index === Number(band.value) ? 3 : 1;
      context.beginPath();
      decayPoints(values, plot.width, plot.height).forEach(([px, py], k) => (k ? context.lineTo(px, py) : context.moveTo(px, py)));
      context.stroke();
    });
    // The shape the moments give in the chosen band, as a shell about the cell.
    const shape = answer.shape.energy ? answer.shape.energy[Math.min(Number(band.value) || 0, answer.shape.energy.length - 1)] : null;
    if (shape) {
      const centre = about.cells[answer.cell].centre_m;
      const { elevations, azimuths, directions } = answer.shape;
      const positions = new Float32Array(3 * shape.length);
      for (let k = 0; k < shape.length; k++) for (let axis = 0; axis < 3; axis++) positions[3 * k + axis] = centre[axis] + (0.2 + 0.6 * shape[k]) * directions[3 * k + axis];
      const index = [];
      for (let e = 0; e + 1 < elevations; e++) for (let a = 0; a < azimuths; a++) {
        const b = (a + 1) % azimuths;
        index.push(e * azimuths + a, (e + 1) * azimuths + a, e * azimuths + b, e * azimuths + b, (e + 1) * azimuths + a, (e + 1) * azimuths + b);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setIndex(index);
      put("tail", new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0x4fd1e8, wireframe: true, transparent: true, opacity: 0.5, depthTest: false })));
    }
    say("tail", [
      `tail of ${source.value}: ${rows.length} histograms. Orange: the sites rays leave from; blue spheres: the receiver spheres of ${about.receiver_radius_m} m on the tail's cells`,
      `histogram ${answer.row} (site at ${answer.site_m.join(", ")}, cell ${answer.cell}): energy over ${(answer.energy[0].length * answer.bin_s).toFixed(1)} s, 80 dB down the plot, a line a band (${answer.bands_hz.join(", ")} Hz). EXACT: ${answer.exact}`,
    ]);
  }

  async function drawRays() {
    drop("rays");
    if (!on.rays || !about) return say("rays", "");
    if (!(about.tails[source.value] || []).length) return say("rays", `${source.value || "no source"} has no tail`, true);
    const mine = stamp;
    say("rays", `tracing ${rayCount.value} rays again`);
    const started = performance.now();
    const answer = await ask("rays", { pack: idA, source: source.value, row: hist.value || 0, count: rayCount.value });
    if (mine !== stamp || !on.rays) return;
    const bands = answer.meta.bands_hz.length;
    const held = raySegments(answer.arrays, bands, Math.min(Number(band.value) || 0, bands - 1));
    put("rays", lines(held.positions, held.colours), ball(answer.meta.site_m, 0.06, 0xff7a1a), ball(answer.meta.cell_m, answer.meta.receiver_radius_m, 0x4fd1e8, true));
    const crossed = [...answer.arrays.crossing_vertex].flatMap((vertex) => [...answer.arrays.points.subarray(3 * vertex, 3 * vertex + 3)]);
    if (crossed.length) put("rays", dots(crossed, 0xffffff, 0.06));
    const [r, g, b] = energyColour(1e-3);
    say("rays", [
      `A SAMPLE: ${answer.meta.sample}. Traced in ${((performance.now() - started) / 1000).toFixed(1)} s`,
      `${held.legs.toLocaleString()} legs; colour: the energy left at ${answer.meta.bands_hz[Math.min(Number(band.value) || 0, bands - 1)]} Hz over 60 dB (30 dB down is rgb ${r},${g},${b}); white: the leg before a counted crossing of cell ${answer.meta.cell}'s sphere (${answer.arrays.crossing_vertex.length})`,
      answer.meta.truncated ? `${answer.meta.truncated} rays were still alive when they were left` : "",
    ].filter(Boolean));
  }

  // --- which layer redraws what -------------------------------------------------
  const DRAW = { walls: drawWalls, slice: drawSlice, snapped: drawSnapped, difference: drawDifference, facets: drawFacets, occluders: drawOccluders, edges: drawEdges, tail: drawTail, rays: drawRays };
  function refresh(name) {
    if (name === "bare") return viewport.setModelVisible(!on.bare);
    if (name === "paths" || name === "low") {
      drawnStep = "";
      if (on.paths && about) ensureCatalogue().catch(() => {});
      return drawStep().catch(failed(name));
    }
    return DRAW[name]().catch(failed(name));
  }
  const again = (...names) => names.forEach((name) => on[name] && refresh(name));

  function fillGrids() {
    const options = (told, own) => [
      new Option("no grid", ""),
      ...(told ? told.grid.offered : []).map((entry) => new Option(`${entry.own ? "" : "other · "}${shortKey(entry.key)}`, entry.id)),
      ...(own || []),
    ];
    const keepA = gridA.value;
    const keepB = gridB.value;
    gridA.replaceChildren(...options(about));
    gridA.value = [...gridA.options].some((o) => o.value === keepA) && keepA ? keepA : (about && about.grid.own) || (about && about.grid.offered[0] ? about.grid.offered[0].id : "");
    // B may be drawn on its own pack's grid, or on any grid offered to A.
    const ownB = aboutB && aboutB.grid.own ? [new Option(`B · ${shortKey(aboutB.grid.key)}`, aboutB.grid.own)] : [];
    gridB.replaceChildren(...options(about, ownB));
    gridB.value = [...gridB.options].some((o) => o.value === keepB) && keepB ? keepB : ownB.length ? ownB[0].value : "";
  }

  function header() {
    const key = (told) => (told ? `grid ${shortKey(told.assets.voxel_low_key)} · mirror ${shortKey(told.assets.mirror_scene_key)} · ${told.solver}` : "");
    packA.textContent = about
      ? `A: ${about.name} · ${key(about)}${aboutB ? `\nB: ${aboutB.name} · ${key(aboutB)}` : ""}\nheard: ${heard}`
      : "open a scene pack (the sound bar's pack) to see what it was computed on";
    for (const button of listen.querySelectorAll("button")) {
      button.classList.toggle("on", button.value === heard);
      button.disabled = !aboutB;
    }
  }

  async function open(id) {
    stamp += 1;
    const mine = stamp;
    idA = id;
    about = null;
    catalogue = null;
    cache.paths.clear();
    cache.low.clear();
    drawnStep = "";
    for (const [name] of LAYERS) if (name !== "bare") drop(name);
    for (const name of Object.keys(notes)) say(name, "");
    minimap.setUnderlay(null);
    if (!id) {
      aboutB = null;
      header();
      return;
    }
    try {
      const told = await ask("about", { pack: id });
      if (mine !== stamp) return;
      about = told;
    } catch (error) {
      say("pack", error.message, true);
      return;
    }
    if (aboutB && !oneScene(about, aboutB)) aboutB = null;
    const names = Object.keys(about.tails);
    source.replaceChildren(...names.map((name) => new Option(name, name)));
    band.replaceChildren(...about.bands_hz.map((hz, index) => new Option(hz, index)));
    band.value = String(Math.max(0, about.bands_hz.indexOf(1000)));
    height.value = head().y.toFixed(2);
    fillHist();
    fillGrids();
    header();
    for (const [name] of LAYERS) if (on[name] && name !== "bare") refresh(name);
  }

  // Two packs are one scene when the sound stage's list says so (the same movements,
  // whatever their recipes); packs it does not list are compared by their recipe.
  function oneScene(one, other) {
    const listed = sound.state().packs;
    const [a, b] = [one, other].map((told) => listed.find((entry) => entry.id === told.pack));
    return a && b ? sameScene(a, b) : one.recipe_sha256 === other.recipe_sha256;
  }

  async function chooseB(id) {
    aboutB = null;
    if (id) aboutB = await ask("about", { pack: id }).catch(() => null);
    if (aboutB && about && !oneScene(about, aboutB)) {
      say("pack", "B is not a pack of the same scene as A", true);
      aboutB = null;
    } else if (aboutB && about && aboutB.recipe_sha256 !== about.recipe_sha256) {
      say("pack", "B is the same scene under another recipe: its rails are solved at another pitch");
    }
    fillGrids();
    header();
    again("slice", "difference");
  }

  // The pack that is heard is the sound stage's; this panel follows it.
  let packsSeen = "";
  setInterval(() => {
    const state = sound.state();
    const listed = state.packs.map((entry) => entry.id).join(",");
    if (listed !== packsSeen) {
      packsSeen = listed;
      const keep = pickB.value;
      // Two packs brought home under one name are told apart by their ids.
      pickB.replaceChildren(new Option("none", ""), ...state.packs.map((entry) => new Option(`${variantName(entry)} · ${entry.id.slice(0, 6)}`, entry.id)));
      pickB.value = keep;
    }
    const expected = heard === "B" && aboutB ? aboutB.pack : idA;
    if (state.pack !== expected) {
      // Another pack was opened in the sound bar: it is A.
      heard = "A";
      open(state.pack);
    }
  }, POLL_MS);

  listen.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button || !aboutB || button.value === heard) return;
    heard = button.value;
    // The same recipe: the transport keeps its instant and the stream starts again there.
    sound.open(heard === "B" ? aboutB.pack : idA);
    header();
  });
  pickB.addEventListener("change", () => chooseB(pickB.value));
  gridA.addEventListener("change", () => again("walls", "slice", "difference"));
  gridB.addEventListener("change", () => again("slice", "difference"));
  height.addEventListener("change", () => again("slice", "difference"));
  radius.addEventListener("change", () => again("walls", "difference"));
  sliceOf.addEventListener("change", () => again("slice"));
  colourBy.addEventListener("change", () => again("facets", "occluders"));
  band.addEventListener("change", () => again("facets", "occluders", "tail", "rays"));
  source.addEventListener("change", () => {
    fillHist();
    drawnStep = "";
    again("tail", "paths", "low");
  });
  hist.addEventListener("change", () => again("tail"));
  here.addEventListener("click", () => again("walls", "difference"));
  trace.addEventListener("click", () => {
    boxes.rays.checked = on.rays = true;
    refresh("rays");
  });
  scene.transport.on("tick", () => {
    if (about && (on.paths || on.low)) drawStep().catch(failed("paths"));
  });
  scene.onSelect(() => {
    drawnStep = "";
    if (about && (on.paths || on.low)) drawStep().catch(failed("paths"));
  });
  header();

  return {
    open,
    chooseB,
    /** Switch a layer as its checkbox would. */
    set(name, value) {
      boxes[name].checked = on[name] = Boolean(value);
      return refresh(name);
    },
    controls: { height, radius, sliceOf, colourBy, band, source, hist, rayCount, gridA, gridB },
    again,
    /** For a check that cannot look: what is on show. */
    state: () => ({
      pack: idA,
      packB: aboutB ? aboutB.pack : null,
      heard,
      on: { ...on },
      drawn: Object.fromEntries(Object.entries(drawn).map(([name, objects]) => [name, objects.length])),
      notes: Object.fromEntries(Object.entries(notes).map(([name, node]) => [name, node.textContent])),
      grids: { pack: idA, a: gridA.value, b: gridB.value },
    }),
  };
}
