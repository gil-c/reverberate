/** Source glyphs in the 3D view: a thick-lined wire sphere for an omni source.
 *
 * The glyph stands for the directivity, so a cardioid will be a lobe drawn
 * with the same lines. Lines are `Line2` from the three.js addons, whose
 * width is in pixels rather than the one-pixel hairline WebGL gives
 * `LineBasicMaterial`, so the sphere reads from across a room.
 *
 * A run's sources stay where they were solved. A scene's move: `place` puts a
 * glyph somewhere else and turns it, and a source given `facing` carries an
 * arrow along its own front, which is the recipe's yaw (0 faces `+x`, +90
 * faces `-z`).
 */
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

const RADIUS = 0.18;
const AMBER = 0xf2a541;
const AMBER_DIM = 0x8a5f23;
const SEGMENTS = 64;
// The facing arrow: from the sphere's surface, this far out, with a head.
const ARROW = 0.5;
const ARROW_HEAD = 0.12;

function circle(THREE, tilt, spin) {
  const points = [];
  for (let k = 0; k <= SEGMENTS; k++) {
    const t = (k / SEGMENTS) * Math.PI * 2;
    const p = new THREE.Vector3(Math.cos(t) * RADIUS, Math.sin(t) * RADIUS, 0);
    p.applyAxisAngle(new THREE.Vector3(1, 0, 0), tilt);
    p.applyAxisAngle(new THREE.Vector3(0, 1, 0), spin);
    points.push(p.x, p.y, p.z);
  }
  return points;
}

/** The rings of an omni glyph: equator, two meridians, two tilted circles. */
const OMNI_RINGS = [
  [Math.PI / 2, 0],
  [0, 0],
  [0, Math.PI / 2],
  [0, Math.PI / 4],
  [0, -Math.PI / 4],
];

/** An arrow along the glyph's own `+x`, drawn flat. */
const ARROW_POINTS = [
  RADIUS, 0, 0,
  ARROW, 0, 0,
  ARROW - ARROW_HEAD, 0, ARROW_HEAD * 0.6,
  ARROW, 0, 0,
  ARROW - ARROW_HEAD, 0, -ARROW_HEAD * 0.6,
];

export function createSourceGlyphs(THREE, viewport) {
  const group = new THREE.Group();
  group.name = "sources";
  const materials = [];
  const glyphs = new Map();

  const makeMaterial = (colour, width) => {
    const material = new LineMaterial({ color: colour, linewidth: width, depthTest: false });
    material.resolution.set(innerWidth, innerHeight);
    materials.push(material);
    return material;
  };
  viewport.onResize((width, height) => {
    for (const material of materials) material.resolution.set(width, height);
  });

  function build(source) {
    const holder = new THREE.Group();
    holder.name = source.id;
    holder.position.fromArray(source.position);
    // Yaw about the up axis, then pitch about the glyph's own left.
    holder.rotation.order = "YZX";
    const colour = source.colour === undefined ? AMBER : source.colour;
    // Every directivity draws the omni glyph until a cardioid lobe exists.
    const line = makeMaterial(colour, 2.5);
    const outlines = [...OMNI_RINGS.map(([tilt, spin]) => circle(THREE, tilt, spin))];
    if (source.facing) outlines.push(ARROW_POINTS);
    for (const points of outlines) {
      const geometry = new LineGeometry();
      geometry.setPositions(points);
      const ring = new Line2(geometry, line);
      ring.computeLineDistances();
      ring.renderOrder = 4;
      holder.add(ring);
    }
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(RADIUS * 0.12, 16, 12),
      new THREE.MeshBasicMaterial({ color: colour, depthTest: false })
    );
    core.renderOrder = 4;
    holder.add(core);
    holder.userData.line = line;
    holder.userData.colour = colour;
    holder.userData.dim =
      source.colour === undefined ? AMBER_DIM : new THREE.Color(colour).multiplyScalar(0.45).getHex();
    return holder;
  }

  return {
    group,
    set(sources) {
      for (const glyph of glyphs.values()) {
        group.remove(glyph);
        const index = materials.indexOf(glyph.userData.line);
        if (index >= 0) materials.splice(index, 1);
        glyph.userData.line.dispose();
      }
      glyphs.clear();
      for (const source of sources) {
        const glyph = build(source);
        glyphs.set(source.id, glyph);
        group.add(glyph);
      }
    },
    /** `on` shows the glyph; `dim` keeps it drawn in its darker colour. */
    update(sources, selected) {
      for (const source of sources) {
        const glyph = glyphs.get(source.id);
        if (!glyph) continue;
        glyph.visible = source.on;
        const line = glyph.userData.line;
        line.color.setHex(source.on && !source.dim ? glyph.userData.colour : glyph.userData.dim);
        line.linewidth = source.id === selected ? 4 : 2.5;
      }
    },
    /** Move a glyph and turn it: `yaw` and `pitch` in radians, the recipe's sense. */
    place(id, x, y, z, yaw = 0, pitch = 0) {
      const glyph = glyphs.get(id);
      if (!glyph) return;
      glyph.position.set(x, y, z);
      glyph.rotation.set(0, yaw, pitch);
    },
  };
}
