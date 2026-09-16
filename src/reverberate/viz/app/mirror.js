/** The mirror's audit in the room: what the geometric engine was handed, and the paths it found.
 *
 * Two layers of the derived geometry, drawn with the grid's own quad loader
 * and palette so a material has one colour across the views: the reflecting
 * facets, and the decimated occluders. And, at the listener's cell, the
 * validated paths of the selected source from the source through every hit
 * point to the cell, coloured by order. The census of the derivation is
 * shown beside the switches, because what was left out must be counted
 * where the picture is.
 */
import { fetchQuadMesh } from "./grid.js";

const ORDER_COLOURS = [0xf5f0e6, 0xf2a541, 0xe8783d, 0xd94f3a, 0xb03a5c, 0x7a3a8c, 0x4a4a9c];

export function createMirrorLayers(THREE, viewport, { facts, note }) {
  const group = new THREE.Group();
  group.name = "mirror";
  viewport.overlays.add(group);
  const layers = { reflectors: null, occluders: null };
  const wanted = { reflectors: false, occluders: false, paths: true };
  let base = null;
  let record = null;
  const pathFiles = new Map(); // source id -> parsed paths json
  let lines = null;

  async function ensureLayer(name) {
    if (!record || layers[name] || !record.layers[name]) return;
    const mesh = await fetchQuadMesh(THREE, base, record.layers[name]);
    mesh.name = `mirror-${name}`;
    mesh.material.transparent = true;
    mesh.material.opacity = name === "occluders" ? 0.35 : 0.85;
    mesh.material.depthWrite = name !== "occluders";
    layers[name] = mesh;
    group.add(mesh);
    apply();
  }

  function apply() {
    for (const name of ["reflectors", "occluders"]) {
      if (layers[name]) layers[name].visible = wanted[name];
    }
    if (lines) lines.visible = wanted.paths;
    viewport.invalidate();
  }

  function renderFacts() {
    facts.replaceChildren();
    if (!record) {
      note.textContent = "";
      return;
    }
    const totals = record.census.totals || {};
    const rows = [
      ["facets", `${totals.facets}`],
      ["reflecting", `${(totals.reflector_area_m2 || 0).toFixed(0)} of ${(totals.model_area_m2 || 0).toFixed(0)} m²`],
      ["diffuse", `${(totals.diffuse_area_m2 || 0).toFixed(0)} m²`],
      ["occluders", `${totals.occluder_triangles} of ${totals.model_triangles} tri`],
      ["open meshes", `${(totals.labels_kept_open || []).length}`],
      ["over bound", `${(totals.labels_over_bound || []).length}`],
    ];
    facts.replaceChildren(
      ...rows.flatMap(([name, value]) => {
        const dt = document.createElement("dt");
        dt.textContent = name;
        const dd = document.createElement("dd");
        dd.textContent = value;
        return [dt, dd];
      })
    );
    note.textContent = record.note || "";
  }

  function drawPaths(id, position) {
    if (lines) {
      group.remove(lines);
      lines.geometry.dispose();
      lines.material.dispose();
      lines = null;
    }
    const file = pathFiles.get(id);
    if (!file || position === null || position === undefined) return;
    const rows = file.points[position] || [];
    const vertices = [];
    const colours = [];
    const colour = new THREE.Color();
    for (const [order, , , flat] of rows) {
      colour.setHex(ORDER_COLOURS[Math.min(order, ORDER_COLOURS.length - 1)]);
      for (let i = 0; i + 5 < flat.length; i += 3) {
        vertices.push(flat[i], flat[i + 1], flat[i + 2], flat[i + 3], flat[i + 4], flat[i + 5]);
        colours.push(colour.r, colour.g, colour.b, colour.r, colour.g, colour.b);
      }
    }
    if (!vertices.length) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3));
    lines = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthTest: false })
    );
    lines.name = "mirror-paths";
    lines.renderOrder = 5;
    group.add(lines);
    apply();
  }

  return {
    /** Forget the previous run's layers. */
    clear() {
      for (const name of ["reflectors", "occluders"]) {
        if (layers[name]) {
          group.remove(layers[name]);
          layers[name].geometry.dispose();
          layers[name].material.dispose();
          layers[name] = null;
        }
      }
      drawPaths(null, null);
      pathFiles.clear();
      record = null;
      base = null;
      renderFacts();
    },
    /** The mirror record of run.json, with the run's base url. */
    async load(run) {
      this.clear();
      if (!run || !run.mirror || !run.mirror.audit) return;
      base = `${run.baseUrl}/${run.mirror.audit}`;
      record = await fetch(`${base}/layers.json`).then((r) => (r.ok ? r.json() : null));
      renderFacts();
      for (const [id, url] of Object.entries(run.mirror.paths || {})) {
        fetch(`${run.baseUrl}/${url}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((file) => {
            if (file) pathFiles.set(id, file);
          })
          .catch(() => {});
      }
      for (const name of ["reflectors", "occluders"]) if (wanted[name]) await ensureLayer(name);
    },
    setWanted(name, on) {
      wanted[name] = on;
      if (on && name !== "paths") ensureLayer(name).catch(() => {});
      apply();
    },
    /** The listener stands at `position` of source `id`'s field: draw its paths. */
    showCell(id, position) {
      drawPaths(id, position);
    },
    has: () => Boolean(record),
    counts: () => (record ? record.census.totals : null),
  };
}
