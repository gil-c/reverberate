/** Source glyphs in the 3D view: a thick-lined wire sphere for an omni source.
 *
 * The glyph stands for the directivity, so a cardioid will be a lobe drawn
 * with the same lines. Lines are `Line2` from the three.js addons, whose
 * width is in pixels rather than the one-pixel hairline WebGL gives
 * `LineBasicMaterial`, so the sphere reads from across a room.
 */
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

const RADIUS = 0.18;
const AMBER = 0xf2a541;
const AMBER_DIM = 0x8a5f23;
const SEGMENTS = 64;

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
    // Every directivity draws the omni glyph until a cardioid lobe exists.
    const line = makeMaterial(AMBER, 2.5);
    for (const [tilt, spin] of OMNI_RINGS) {
      const geometry = new LineGeometry();
      geometry.setPositions(circle(THREE, tilt, spin));
      const ring = new Line2(geometry, line);
      ring.computeLineDistances();
      ring.renderOrder = 4;
      holder.add(ring);
    }
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(RADIUS * 0.12, 16, 12),
      new THREE.MeshBasicMaterial({ color: AMBER, depthTest: false })
    );
    core.renderOrder = 4;
    holder.add(core);
    holder.userData.line = line;
    return holder;
  }

  return {
    group,
    set(sources) {
      for (const glyph of glyphs.values()) group.remove(glyph);
      glyphs.clear();
      for (const source of sources) {
        const glyph = build(source);
        glyphs.set(source.id, glyph);
        group.add(glyph);
      }
    },
    update(sources, selected) {
      for (const source of sources) {
        const glyph = glyphs.get(source.id);
        if (!glyph) continue;
        glyph.visible = source.on;
        const line = glyph.userData.line;
        line.color.setHex(source.on ? AMBER : AMBER_DIM);
        line.linewidth = source.id === selected ? 4 : 2.5;
      }
    },
  };
}
