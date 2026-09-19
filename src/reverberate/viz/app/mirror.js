/** The mirror solver's view: the derived scene it read, and the paths it found.
 *
 * The reflecting facets and the decimated occluders, each drawn as the
 * scene's own triangles (the server writes them from the scene the solver
 * read), the receivers it was given, and at the listener's cell the paths the
 * image sources validated. Facets are coloured by material label with the
 * grid's palette, or by facet kind (shell floor, wall, ceiling, furniture).
 */
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { fetchQuadMesh } from "./grid.js";

const ORDER_COLOURS = [0xf5f0e6, 0xf2a541, 0xe8783d, 0xd94f3a, 0xb03a5c, 0x7a3a8c, 0x4a4a9c];

/** A path's colour by its reflection order. */
export const orderColour = (order) => ORDER_COLOURS[Math.min(order, ORDER_COLOURS.length - 1)];

//: Facet kinds and their colours when the facets are coloured by kind.
export const KIND_COLOURS = {
  shell_floor: 0xb08850,
  shell_wall: 0xd8d2c4,
  shell_ceiling: 0x6f93c0,
  furniture: 0x5fbf72,
};

//: Path width in CSS pixels: a one pixel hairline is lost against the walls.
const PATH_PX = 2;

export function createMirrorLayers(THREE, { onLoaded, onPaths } = {}) {
  const group = new THREE.Group();
  group.name = "mirror";
  const layers = { reflectors: null, occluders: null };
  const loading = { reflectors: null, occluders: null };
  const wanted = { reflectors: true, occluders: false, paths: true, receivers: false };
  let colourBy = "label";
  let base = null;
  let scene = null; // layers.json: key, labels, facets, census, materials
  let quadMeta = null;
  const pathFiles = new Map(); // source id -> parsed paths json
  let pathUrls = {}; // source id -> url, fetched when first shown
  let cell = { id: null, position: null }; // where the listener stands
  let lines = null;
  let receivers = null;
  let redraw = () => {};
  const resolution = new THREE.Vector2(innerWidth, innerHeight);

  function apply() {
    for (const name of ["reflectors", "occluders"]) {
      if (layers[name]) layers[name].visible = wanted[name];
    }
    // Occluders overlap the reflectors they were decimated from: with both
    // on, the occluders turn translucent and sit behind.
    if (layers.occluders) {
      const both = wanted.reflectors && Boolean(layers.reflectors);
      const material = layers.occluders.material;
      if (material.transparent !== both) material.needsUpdate = true;
      material.transparent = both;
      material.opacity = both ? 0.35 : 1;
      material.depthWrite = !both;
    }
    if (lines) lines.visible = wanted.paths;
    if (receivers) receivers.visible = wanted.receivers;
    redraw();
  }

  /** The kind colours of the reflector triangles, made once. */
  function kindColours(mesh) {
    if (mesh.userData.kindColour) return mesh.userData.kindColour;
    const colour = new Float32Array(mesh.geometry.attributes.position.count * 3);
    const rgb = new THREE.Color();
    let quad = 0;
    for (const facet of scene.facets) {
      rgb.setHex(KIND_COLOURS[facet.kind] ?? 0xff00ff);
      for (let t = 0; t < facet.triangles; t++, quad++) {
        for (let corner = 0; corner < 4; corner++) {
          const at = (quad * 4 + corner) * 3;
          colour[at] = rgb.r;
          colour[at + 1] = rgb.g;
          colour[at + 2] = rgb.b;
        }
      }
    }
    mesh.userData.kindColour = new THREE.BufferAttribute(colour, 3);
    return mesh.userData.kindColour;
  }

  /** Hatch the back of a one-sided facet: the engine reflects nothing there. */
  function markBacks(mesh) {
    const sides = new Float32Array(mesh.geometry.attributes.position.count);
    let quad = 0;
    for (const facet of scene.facets) {
      sides.fill(facet.sides, quad * 4, (quad + facet.triangles) * 4);
      quad += facet.triangles;
    }
    mesh.geometry.setAttribute("aSides", new THREE.BufferAttribute(sides, 1));
    const material = mesh.material;
    const base = material.onBeforeCompile;
    material.onBeforeCompile = (shader) => {
      base(shader);
      shader.vertexShader = `attribute float aSides;\nvarying float vSides;\n${shader.vertexShader}`.replace(
        "#include <begin_vertex>",
        "#include <begin_vertex>\n vSides = aSides;"
      );
      shader.fragmentShader = `varying float vSides;\n${shader.fragmentShader}`.replace(
        "#include <dithering_fragment>",
        `#include <dithering_fragment>
         if (!gl_FrontFacing && vSides < 2.5) {
           float stripe = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 8.0));
           gl_FragColor.rgb = mix(gl_FragColor.rgb * 0.35, vec3(0.55, 0.1, 0.12), stripe);
         }`
      );
    };
    material.customProgramCacheKey = () => "mirror-reflectors";
    material.needsUpdate = true;
  }

  function paint() {
    const mesh = layers.reflectors;
    if (!mesh) return;
    if (!mesh.userData.labelColour) mesh.userData.labelColour = mesh.geometry.attributes.color;
    mesh.geometry.setAttribute("color", colourBy === "kind" ? kindColours(mesh) : mesh.userData.labelColour);
    redraw();
  }

  async function ensureLayer(name) {
    if (!base || layers[name] || !quadMeta || !quadMeta[name]) return;
    const mine = base;
    loading[name] = loading[name] || fetchQuadMesh(THREE, base, quadMeta[name]);
    const mesh = await loading[name];
    loading[name] = null;
    if (mine !== base || layers[name]) {
      if (layers[name] !== mesh) {
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
      return;
    }
    mesh.name = `mirror-${name}`;
    if (name === "occluders") {
      mesh.material.polygonOffset = true;
      mesh.material.polygonOffsetFactor = 2;
      mesh.material.polygonOffsetUnits = 2;
      mesh.renderOrder = 1;
    }
    layers[name] = mesh;
    group.add(mesh);
    if (name === "reflectors") {
      markBacks(mesh);
      paint();
    }
    apply();
    if (onLoaded) onLoaded(name);
  }

  function drawPaths(id, position) {
    if (lines) {
      dropMesh(lines);
      lines = null;
    }
    const file = pathFiles.get(id);
    if (!file || position === null || position === undefined) return;
    const rows = file.points[position] || [];
    const vertices = [];
    const colours = [];
    const colour = new THREE.Color();
    for (const [order, flat] of rows) {
      colour.setHex(orderColour(order));
      for (let i = 0; i + 5 < flat.length; i += 3) {
        vertices.push(flat[i], flat[i + 1], flat[i + 2], flat[i + 3], flat[i + 4], flat[i + 5]);
        colours.push(colour.r, colour.g, colour.b, colour.r, colour.g, colour.b);
      }
    }
    if (!vertices.length) return;
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(vertices);
    geometry.setColors(colours);
    const material = new LineMaterial({ vertexColors: true, linewidth: PATH_PX, depthTest: false });
    material.resolution.copy(resolution);
    lines = new LineSegments2(geometry, material);
    lines.name = "mirror-paths";
    lines.renderOrder = 5;
    group.add(lines);
    apply();
  }

  function dropMesh(object) {
    group.remove(object);
    object.geometry.dispose();
    object.material.dispose();
  }

  /** The paths of source `id`, fetched once per run; the cell redrawn when they land. */
  function ensurePaths(id) {
    if (!id || pathFiles.has(id) || !pathUrls[id]) return;
    const mine = base;
    pathFiles.set(id, null);
    fetch(pathUrls[id])
      .then((r) => (r.ok ? r.json() : null))
      .then((file) => {
        if (base !== mine) return;
        pathFiles.set(id, file);
        if (cell.id === id) drawPaths(cell.id, cell.position);
        if (onPaths) onPaths(id);
      })
      .catch(() => pathFiles.delete(id));
  }

  return {
    group,
    /** The drawing size, for the paths' pixel width. */
    setSize(width, height) {
      resolution.set(width, height);
      if (lines) lines.material.resolution.copy(resolution);
    },
    /** Called whenever the picture changed. */
    onRedraw(handler) {
      redraw = handler;
    },
    /** Forget the previous run's layers. */
    clear() {
      for (const name of ["reflectors", "occluders"]) {
        if (layers[name]) dropMesh(layers[name]);
        layers[name] = null;
        loading[name] = null;
      }
      if (receivers) dropMesh(receivers);
      receivers = null;
      drawPaths(null, null);
      pathFiles.clear();
      pathUrls = {};
      cell = { id: null, position: null };
      scene = null;
      base = null;
      quadMeta = null;
    },
    /** The mirror record of run.json; resolves with the scene once its index is read. */
    async load(run) {
      this.clear();
      if (!run || !run.mirror) return null;
      const mine = `${run.baseUrl}/${run.mirror.audit}`;
      base = mine;
      pathUrls = Object.fromEntries(
        Object.entries(run.mirror.paths || {}).map(([id, url]) => [id, `${run.baseUrl}/${url}`])
      );
      const index = await fetch(`${mine}/layers.json`).then((r) => (r.ok ? r.json() : null));
      if (base !== mine || !index) return null;
      scene = index;
      quadMeta = index.layers;
      return scene;
    },
    /** Fetch what the switches ask for; the view calls this when it is shown. */
    async ensure() {
      await Promise.all(["reflectors", "occluders"].filter((n) => wanted[n]).map((n) => ensureLayer(n)));
    },
    setWanted(name, on) {
      wanted[name] = on;
      if (on && (name === "reflectors" || name === "occluders")) ensureLayer(name).catch(() => {});
      apply();
    },
    setColourBy(mode) {
      colourBy = mode;
      paint();
    },
    /** The receivers the engine was handed: the reference field's lattice. */
    setReceivers(positions) {
      if (receivers) dropMesh(receivers);
      receivers = null;
      if (!positions || !positions.length) return;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions.flat(), 3));
      receivers = new THREE.Points(
        geometry,
        new THREE.PointsMaterial({ color: 0x4fd1e8, size: 0.06, sizeAttenuation: true, depthTest: false })
      );
      receivers.name = "mirror-receivers";
      receivers.renderOrder = 4;
      group.add(receivers);
      apply();
    },
    /** The listener stands at `position` of source `id`'s field: draw its paths. */
    showCell(id, position) {
      cell = { id, position };
      ensurePaths(id);
      drawPaths(id, position);
    },
    /** The paths at a cell, as ``[order, vertices]``; null until the file is in. */
    pathsAt(id, position) {
      const file = pathFiles.get(id);
      return file && position !== null && position !== undefined ? file.points[position] || [] : null;
    },
    record: () => scene,
  };
}
