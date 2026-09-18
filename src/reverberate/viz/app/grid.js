/** The solver's own grid, tiered: coarse everywhere, fine around the listener.
 *
 * A mesh payload is what `reverberate.experiments.audit_view` writes: every
 * room in two tiers of the *same* voxelisation, the solver's own step and
 * that step decimated, cut into tiles where a room is too heavy for one file.
 * The coarse tier of every room is resident and always drawn. Fine tiles are
 * admitted by distance to the listener, nearest first, under a quad budget,
 * and evicted once they fall behind; where a fine tile is on screen its own
 * room's coarse tier is cut away by the shader, so the two do not overlap.
 * A tile's box is the reach of its merged quads, so boxes overlap: the cut
 * spares what lies in the box of a tile not drawn, since those faces have no
 * fine stand-in yet. Only its own room's: a box reaches past a doorway into
 * the next room, whose coarse faces there are the only ones drawn.
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
//: M-series laptop; beyond that the picture is lost to swapping.
export const FINE_QUAD_BUDGET = 16_000_000;

//: A fine tile is evicted this much farther out than it is admitted, so a
//: listener pacing on the admission radius does not fetch it again and again.
const HYSTERESIS_M = 1;

//: How many fine tile boxes of one room the coarse shader can hold per list.
const CLIP_MAX = 64;

//: The label the audit views single out, shared by every quad mesh: -1 is
//: none, and then every face keeps its own colour exactly.
export const HIGHLIGHT = { value: -1 };
const HIGHLIGHT_OTHERS = "vec3(0.26, 0.28, 0.31)";

/** A material's colour: seventeen hues over three lightnesses, by index. */
export function materialColour(rgb, index) {
  const hue = (index % 17) / 17;
  const lightness = [0.78, 0.62, 0.46][Math.floor(index / 17) % 3];
  return rgb.setHSL(hue, 0.6, lightness);
}

/** The shader uniforms that carve the drawn fine tiles out of the coarse tier:
 *  `cut` boxes are dropped, except where a `keep` box says otherwise. */
function createClip(THREE) {
  const boxes = () => ({
    count: { value: 0 },
    lo: { value: Array.from({ length: CLIP_MAX }, () => new THREE.Vector3()) },
    hi: { value: Array.from({ length: CLIP_MAX }, () => new THREE.Vector3()) },
  });
  return { cut: boxes(), keep: boxes() };
}

/** Unlit, shaded from the face's own normal, so a face keeps its colour and
 *  only its orientation changes how bright it is: the colour is the datum.
 *  With `clip`, fragments inside any listed box are dropped. */
function quadMaterial(THREE, clip) {
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  if (clip) {
    // Pushed back a little, so where both tiers remain the fine one shows.
    material.polygonOffset = true;
    material.polygonOffsetFactor = 1;
    material.polygonOffsetUnits = 1;
  }
  material.customProgramCacheKey = () => (clip ? "grid-clip" : "grid");
  material.onBeforeCompile = (shader) => {
    shader.uniforms.highlight = HIGHLIGHT;
    shader.vertexShader = `attribute float aLabel;\nuniform float highlight;\n${shader.vertexShader}`.replace(
      "#include <color_vertex>",
      `#include <color_vertex>
       vec3 n = normalize(normalMatrix * normal);
       float shade = 0.55 + 0.45 * abs(n.y) + 0.15 * abs(n.x);
       vColor.rgb *= shade;
       if (highlight > -0.5 && abs(aLabel - highlight) > 0.5) vColor.rgb = ${HIGHLIGHT_OTHERS} * shade;`
    );
    if (!clip) return;
    Object.assign(shader.uniforms, {
      cutCount: clip.cut.count,
      cutLo: clip.cut.lo,
      cutHi: clip.cut.hi,
      keepCount: clip.keep.count,
      keepLo: clip.keep.lo,
      keepHi: clip.keep.hi,
    });
    shader.vertexShader = `varying vec3 vClip;\n${shader.vertexShader}`.replace(
      "#include <begin_vertex>",
      `#include <begin_vertex>
       vClip = (modelMatrix * vec4(transformed, 1.0)).xyz;`
    );
    shader.fragmentShader = `varying vec3 vClip;
       uniform int cutCount;
       uniform vec3 cutLo[${CLIP_MAX}];
       uniform vec3 cutHi[${CLIP_MAX}];
       uniform int keepCount;
       uniform vec3 keepLo[${CLIP_MAX}];
       uniform vec3 keepHi[${CLIP_MAX}];
       bool inBox(vec3 lo, vec3 hi) { return all(greaterThan(vClip, lo)) && all(lessThan(vClip, hi)); }
       ${shader.fragmentShader}`.replace(
      "void main() {",
      `void main() {
       bool kept = false;
       for (int i = 0; i < ${CLIP_MAX}; i++) {
         if (i >= keepCount) break;
         if (inBox(keepLo[i], keepHi[i])) { kept = true; break; }
       }
       if (!kept) {
         for (int i = 0; i < ${CLIP_MAX}; i++) {
           if (i >= cutCount) break;
           if (inBox(cutLo[i], cutHi[i])) discard;
         }
       }`
    );
  };
  return material;
}

/** One payload of merged quads, as a mesh: fetch the three arrays and shade them. */
export async function fetchQuadMesh(THREE, base, meta, clip = null) {
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
  // The label per corner, for the highlight and for picking a face.
  geometry.setAttribute("aLabel", new THREE.BufferAttribute(labels, 1));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(index), 1));
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, quadMaterial(THREE, clip));
  mesh.name = "voxels";
  // Picking must skip what the shader cuts away.
  mesh.userData.clip = clip;
  return mesh;
}

/** Whether the shader of a mesh with `clip` drops the fragment at `p`. */
export function clippedAt(clip, p) {
  if (!clip) return false;
  const inside = (list) => {
    for (let i = 0; i < list.count.value; i++) {
      const lo = list.lo.value[i];
      const hi = list.hi.value[i];
      if (p.x > lo.x && p.y > lo.y && p.z > lo.z && p.x < hi.x && p.y < hi.y && p.z < hi.z) return true;
    }
    return false;
  };
  return !inside(clip.keep) && inside(clip.cut);
}

/** The squared distance from a point to a tile's own box, zero inside it. */
function reach(file, at) {
  const [lo, hi] = file.bounds;
  let sum = 0;
  for (let axis = 0; axis < 3; axis++) {
    const value = at.getComponent(axis);
    const gap = Math.max(lo[axis] - value, 0, value - hi[axis]);
    sum += gap * gap;
  }
  return sum;
}

/** Which fine tiles to draw around `at`: within `nearM`, nearest first, under the budget. */
export function admit(tiles, at, nearM, budget = FINE_QUAD_BUDGET) {
  const order = tiles
    .map((tile, i) => ({ i, far: reach(tile.file, at) }))
    .filter(({ far }) => far <= nearM * nearM)
    .sort((a, b) => a.far - b.far);
  let spent = 0;
  const wanted = new Set();
  for (const { i } of order) {
    const quads = Number(tiles[i].file.quads);
    if (spent && spent + quads > budget) break;
    spent += quads;
    wanted.add(i);
  }
  return { wanted, quads: spent };
}

/** Load one band limit's payload and return the tiered view of it.
 *
 * `near()` gives the admission radius in metres at the time of each refresh.
 */
export async function loadGridView(THREE, base, near, onStatus) {
  const index = await fetch(`${base}/rooms.json`).then((r) => {
    if (!r.ok) throw new Error(`${base}/rooms.json: ${r.status}`);
    return r.json();
  });
  const group = new THREE.Group();
  group.name = "grid";
  const clips = index.rooms.map(() => createClip(THREE));
  const tiles = index.rooms.flatMap((room, r) =>
    room.fine.tiles.map((file) => ({ file, dir: room.dir, room: r, mesh: null, loading: null, drawn: false }))
  );
  const step = (tier) => Math.max(...index.rooms.map((room) => room[tier].cell_m)) * 1000;
  let drawnQuads = 0;

  const say = () => {
    if (!onStatus || !index.rooms.length) return;
    onStatus({
      fine_mm: step("fine"),
      coarse_mm: step("coarse"),
      tiles_drawn: tiles.filter((tile) => tile.drawn).length,
      tiles: tiles.length,
      quads: drawnQuads,
    });
  };

  // Every room's coarse tier up front: this is the picture of the whole flat.
  for (const [r, room] of index.rooms.entries()) {
    for (const file of room.coarse.tiles) {
      group.add(await fetchQuadMesh(THREE, `${base}/${room.dir}`, file, clips[r]));
    }
  }

  const dispose = (tile) => {
    if (!tile.mesh) return;
    group.remove(tile.mesh);
    tile.mesh.geometry.dispose();
    tile.mesh.material.dispose();
    tile.mesh = null;
  };

  /** Tell each room's coarse shader which fine boxes are on screen and which are not. */
  const carve = () => {
    for (const clip of clips) clip.cut.count.value = clip.keep.count.value = 0;
    for (const tile of tiles) {
      const list = tile.drawn ? clips[tile.room].cut : clips[tile.room].keep;
      const n = list.count.value;
      if (n === CLIP_MAX) continue;
      list.lo.value[n].fromArray(tile.file.bounds[0]);
      list.hi.value[n].fromArray(tile.file.bounds[1]);
      list.count.value = n + 1;
    }
  };

  let pass = 0;
  /** Draw the fine tiles around `at`, drop those left behind. */
  const refresh = async (at) => {
    const mine = ++pass;
    const nearM = near();
    const { wanted, quads } = admit(tiles, at, nearM);
    const hold = (nearM + HYSTERESIS_M) ** 2;
    for (const [i, tile] of tiles.entries()) {
      if (wanted.has(i)) continue;
      tile.drawn = false;
      if (tile.mesh) tile.mesh.visible = false;
      if (reach(tile.file, at) > hold) dispose(tile);
    }
    // The carve follows what is on screen at every step, or a hidden tile
    // would leave a hole in the coarse tier while the next one downloads.
    carve();
    for (const i of wanted) {
      const tile = tiles[i];
      if (!tile.mesh) {
        tile.loading = tile.loading || fetchQuadMesh(THREE, `${base}/${tile.dir}`, tile.file);
        const mesh = await tile.loading;
        tile.loading = null;
        if (!tile.mesh) {
          tile.mesh = mesh;
          mesh.visible = false;
          group.add(mesh);
        }
        // The listener may have moved on while this was in flight: the tile
        // is kept, and the latest pass decides whether it goes on screen.
        if (mine !== pass) return;
      }
      tile.mesh.visible = true;
      tile.drawn = true;
      carve();
    }
    drawnQuads = quads;
    say();
  };

  let lastAt = null;
  return {
    group,
    say,
    /** Called on every move with the listener's position; `force` redraws in place. */
    follow(at, force = false) {
      if (force || !lastAt || lastAt.distanceTo(at) > 0.5) {
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
export function createMeshViews(THREE, run, near, onStatus) {
  const views = new Map();
  const bands = Object.keys(run.meshes || {}).sort((a, b) => Number(a) - Number(b));
  return {
    bands,
    /** Band limits whose payload holds a room of the plan, finest last. */
    availableFor(room) {
      return bands.filter((band) => matchRoom(run.meshes[band].rooms, room) !== null);
    },
    async view(band) {
      if (!views.has(band)) {
        views.set(band, loadGridView(THREE, `${run.baseUrl}/${run.meshes[band].url}`, near, onStatus));
      }
      return views.get(band);
    },
  };
}
