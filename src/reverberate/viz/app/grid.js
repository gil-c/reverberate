/** The solver's own grid, tiered: coarse everywhere, fine where you stand.
 *
 * A mesh payload is what `reverberate.experiments.audit_view` writes: every
 * room in two tiers of the *same* voxelisation, the solver's own step and
 * that step aggregated, cut into tiles where a room is too heavy for one file.
 * This draws every room's coarse tier at once and the fine tier of one room,
 * tiles nearest the listener first, under a quad budget.
 *
 * One `GridView` per band limit. The band limits a run offers are its
 * `run.meshes` keys, and a room absent from a payload is a room that limit
 * was never built for, which the page must refuse rather than draw coarser
 * under a finer name.
 */

//: The sealed inside of a solid object and the far side of a boundary the room
//: cannot hear. Both reserved, far from every material hue; see the Python
//: side, which chose them by distance in RGB.
const SEALED_RGB = [0.48, 0.03, 0.14];
const RIGID_RGB = [0.45, 0.45, 0.48];

//: The most quads the fine tier may hold at once, across every tile drawn.
//: Sixteen million holds the largest room measured so far whole on an
//: M-series laptop; below that a room draws with holes, and on an audit view
//: a hole is indistinguishable from geometry the voxeliser lost.
const FINE_QUAD_BUDGET = 16_000_000;

/** A material's colour: seventeen hues over three lightnesses, by index. */
function materialColour(rgb, index) {
  const hue = (index % 17) / 17;
  const lightness = [0.78, 0.62, 0.46][Math.floor(index / 17) % 3];
  return rgb.setHSL(hue, 0.6, lightness);
}

/** One payload of merged quads, as a mesh: fetch the three arrays and shade them. */
export async function fetchQuadMesh(THREE, base, meta) {
  const [corners, index, label] = await Promise.all([
    fetch(`${base}/${meta.corners_url}`).then((r) => r.arrayBuffer()),
    fetch(`${base}/${meta.index_url}`).then((r) => r.arrayBuffer()),
    fetch(`${base}/${meta.label_url}`).then((r) => r.arrayBuffer()),
  ]);
  const position = new Float32Array(corners);
  const labels = new Int16Array(label);
  const colour = new Float32Array(position.length);
  const rgb = new THREE.Color();
  for (let v = 0; v < labels.length; v++) {
    const kind = labels[v];
    if (kind === -2) rgb.setRGB(...SEALED_RGB);
    else if (kind < 0) rgb.setRGB(...RIGID_RGB);
    else materialColour(rgb, kind);
    colour[v * 3] = rgb.r;
    colour[v * 3 + 1] = rgb.g;
    colour[v * 3 + 2] = rgb.b;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colour, 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(index), 1));
  geometry.computeVertexNormals();
  // Unlit, shaded from the face's own normal, so a face keeps its colour and
  // only its orientation changes how bright it is: the colour is the datum.
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader.replace(
      "#include <color_vertex>",
      `#include <color_vertex>
       vec3 n = normalize(normalMatrix * normal);
       vColor.rgb *= 0.55 + 0.45 * abs(n.y) + 0.15 * abs(n.x);`
    );
  };
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "voxels";
  return mesh;
}

/** Load one band limit's payload and return the tiered view of it. */
export async function loadGridView(THREE, base, onStatus) {
  const index = await fetch(`${base}/rooms.json`).then((r) => {
    if (!r.ok) throw new Error(`${base}/rooms.json: ${r.status}`);
    return r.json();
  });
  const group = new THREE.Group();
  group.name = "grid";
  const state = index.rooms.map((room) => ({
    room,
    coarse: null,
    tiles: room.fine.tiles.map((file) => ({ file, mesh: null, loading: null, drawn: false })),
  }));
  let selected = null;
  let drawnQuads = 0;

  const say = () => {
    if (!onStatus) return;
    const entry = selected === null ? null : state[selected];
    onStatus({
      room: entry ? entry.room.name : null,
      fine_mm: entry ? entry.room.fine.cell_m * 1000 : null,
      coarse_mm: index.rooms.length ? index.rooms[0].coarse.cell_m * 1000 : null,
      tiles_drawn: entry ? entry.tiles.filter((tile) => tile.drawn).length : 0,
      tiles: entry ? entry.tiles.length : 0,
    });
  };

  // Every room's coarse tier up front: this is the picture of the whole flat.
  for (const entry of state) {
    const holder = new THREE.Group();
    holder.name = `${entry.room.name}-coarse`;
    for (const file of entry.room.coarse.tiles) {
      holder.add(await fetchQuadMesh(THREE, `${base}/${entry.room.dir}`, file));
    }
    entry.coarse = holder;
    group.add(holder);
  }

  /** The squared distance from a point to a tile's own box, zero inside it. */
  const reach = (file, at) => {
    const [lo, hi] = file.bounds;
    let sum = 0;
    for (let axis = 0; axis < 3; axis++) {
      const value = at.getComponent(axis);
      const gap = Math.max(lo[axis] - value, 0, value - hi[axis]);
      sum += gap * gap;
    }
    return sum;
  };

  /** Draw as much of the selected room's fine tier as the budget allows. */
  const refresh = async (at) => {
    if (selected === null) return;
    const entry = state[selected];
    const order = entry.tiles
      .map((tile, i) => ({ i, far: reach(tile.file, at) }))
      .sort((a, b) => a.far - b.far);
    let spent = 0;
    const wanted = new Set();
    for (const { i } of order) {
      const quads = Number(entry.tiles[i].file.quads);
      if (spent && spent + quads > FINE_QUAD_BUDGET) break;
      spent += quads;
      wanted.add(i);
    }
    for (const [i, tile] of entry.tiles.entries()) {
      if (!wanted.has(i)) {
        if (tile.mesh) tile.mesh.visible = false;
        tile.drawn = false;
        continue;
      }
      if (!tile.mesh) {
        tile.loading =
          tile.loading || fetchQuadMesh(THREE, `${base}/${entry.room.dir}`, tile.file);
        const mesh = await tile.loading;
        // The listener may have left the room while this was in flight; the
        // tile is kept, but it is not put on screen.
        if (selected === null || state[selected] !== entry) return;
        tile.mesh = mesh;
        group.add(mesh);
      }
      tile.mesh.visible = true;
      tile.drawn = true;
    }
    drawnQuads = spent;
    entry.coarse.visible = false;
    say();
  };

  /** Draw one room at the grid's own step, and everything else coarse. */
  const select = async (name, at) => {
    const next = state.findIndex((entry) => entry.room.name === name);
    if (next === selected) return;
    if (selected !== null) {
      const previous = state[selected];
      previous.coarse.visible = true;
      for (const tile of previous.tiles) {
        if (tile.mesh) tile.mesh.visible = false;
        tile.drawn = false;
      }
    }
    selected = next < 0 ? null : next;
    drawnQuads = 0;
    say();
    if (selected !== null) await refresh(at);
  };

  const bounds = new THREE.Box3();
  const corner = new THREE.Vector3();
  for (const entry of state) {
    for (const tier of [entry.room.fine, entry.room.coarse]) {
      for (const file of tier.tiles) {
        for (const point of file.bounds) bounds.expandByPoint(corner.fromArray(point));
      }
    }
  }

  const rooms = index.rooms.map((room) => room.name);
  let lastAt = null;
  let lastRoom = null;
  return {
    group,
    rooms,
    bounds,
    index,
    hasRoom: (name) => rooms.includes(name),
    say,
    /** Called every frame with the listener's position and room. */
    follow(at, room) {
      if (room !== lastRoom) {
        lastRoom = room;
        lastAt = at.clone();
        select(room, at);
      } else if (!lastAt || lastAt.distanceTo(at) > 0.5) {
        lastAt = at.clone();
        refresh(at);
      }
    },
  };
}

/** The payload room standing for a room of the plan, by name or by regions.
 *
 * The plan's rooms and a payload's rooms come from the same rules, but not
 * always from the same day: a payload built before a room was renamed still
 * holds the same regions, and those are what decide.
 */
export function matchRoom(payloadRooms, room) {
  if (!room) return null;
  const regions = new Set(room.regions || [room.name]);
  for (const candidate of payloadRooms) {
    if (candidate.name === room.name) return candidate.name;
    if ((candidate.regions || []).some((region) => regions.has(region))) return candidate.name;
  }
  return null;
}

/** The mesh views of one run, one per band limit, loaded when first drawn. */
export function createMeshViews(THREE, run, onStatus) {
  const views = new Map();
  const bands = Object.keys(run.meshes || {}).sort((a, b) => Number(a) - Number(b));
  return {
    bands,
    /** Band limits whose payload holds a room of the plan, finest last. */
    availableFor(room) {
      return bands.filter((band) => matchRoom(run.meshes[band].rooms, room) !== null);
    },
    /** The payload room's own name for a room of the plan, in one band. */
    roomIn(band, room) {
      return matchRoom(run.meshes[band].rooms, room);
    },
    async view(band) {
      if (!views.has(band)) {
        views.set(band, loadGridView(THREE, `${run.baseUrl}/${run.meshes[band].url}`, onStatus));
      }
      return views.get(band);
    },
  };
}
