/** Where a source's sound comes from at the listener, drawn in the room.
 *
 * At the cursor's step the pack's `early` table holds every arrival of the
 * selected source above the crossover: a direction at the head, a delay, a
 * gain. Each is drawn as the mirror solver's paths are (`mirror.js`): a line
 * from the head along the direction the sound comes from, as long as the
 * sound travelled (`delay * c`, so it ends on the source for the direct path
 * and on its image for a reflection), in the colour of its reflection order,
 * with a bead at its far end whose size is the arrival's gain.
 *
 * The tail has no arrivals, only where its energy comes from: a shell round
 * the head, pushed out in each direction by that energy.
 */
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { orderColour } from "../mirror.js";
import { arrivalGlyphs, shellPoints } from "./sound-plan.js";

const PATH_PX = 2;
const TAIL_HEX = 0x4fd1e8;

export function createArrivals(THREE, viewport) {
  const group = new THREE.Group();
  group.name = "arrivals";
  const resolution = new THREE.Vector2(innerWidth, innerHeight);
  viewport.onResize((width, height) => {
    resolution.set(width, height);
    if (lines) lines.material.resolution.copy(resolution);
  });
  let lines = null;
  let beads = null;
  let shell = null;
  const colour = new THREE.Color();
  const matrix = new THREE.Matrix4();

  function drop(object) {
    if (!object) return;
    group.remove(object);
    object.geometry.dispose();
    object.material.dispose();
  }

  function clear() {
    drop(lines);
    drop(beads);
    drop(shell);
    lines = beads = shell = null;
  }

  return {
    group,
    clear() {
      clear();
      viewport.invalidate();
    },
    /** One step of `/api/audit/arrivals`, with the answer's grid and sound speed. */
    show(step, { soundSpeed, grid }) {
      clear();
      const glyphs = arrivalGlyphs(step, soundSpeed);
      if (glyphs.length) {
        const vertices = [];
        const colours = [];
        for (const glyph of glyphs) {
          colour.setHex(orderColour(glyph.order));
          vertices.push(...step.listener, ...glyph.end);
          colours.push(colour.r, colour.g, colour.b, colour.r, colour.g, colour.b);
        }
        const geometry = new LineSegmentsGeometry();
        geometry.setPositions(vertices);
        geometry.setColors(colours);
        const material = new LineMaterial({ vertexColors: true, linewidth: PATH_PX, depthTest: false });
        material.resolution.copy(resolution);
        lines = new LineSegments2(geometry, material);
        lines.renderOrder = 5;
        group.add(lines);

        beads = new THREE.InstancedMesh(
          new THREE.SphereGeometry(1, 12, 8),
          new THREE.MeshBasicMaterial({ depthTest: false }),
          glyphs.length
        );
        glyphs.forEach((glyph, i) => {
          matrix.makeScale(glyph.radius, glyph.radius, glyph.radius);
          matrix.setPosition(glyph.end[0], glyph.end[1], glyph.end[2]);
          beads.setMatrixAt(i, matrix);
          beads.setColorAt(i, colour.setHex(orderColour(glyph.order)));
        });
        beads.renderOrder = 6;
        beads.frustumCulled = false;
        group.add(beads);
      }
      if (step.tail && grid) {
        const { elevations, azimuths, directions } = grid;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(shellPoints(step.listener, directions, step.tail), 3));
        const index = [];
        for (let e = 0; e + 1 < elevations; e++) {
          for (let a = 0; a < azimuths; a++) {
            const b = (a + 1) % azimuths;
            index.push(e * azimuths + a, (e + 1) * azimuths + a, e * azimuths + b);
            index.push(e * azimuths + b, (e + 1) * azimuths + a, (e + 1) * azimuths + b);
          }
        }
        geometry.setIndex(index);
        shell = new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({ color: TAIL_HEX, wireframe: true, transparent: true, opacity: 0.45, depthTest: false })
        );
        shell.renderOrder = 4;
        shell.frustumCulled = false;
        group.add(shell);
      }
      viewport.invalidate();
    },
  };
}
