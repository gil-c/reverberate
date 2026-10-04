/** A scene in the 3D view: what it may use on the floor, and who is where.
 *
 * Stations and rails are drawn on the storey's floor. Every source and the
 * listener is a glyph of `sources.js`, the run's own, in the colour of its
 * kind and with an arrow along its facing, and leaves on the floor the trail
 * of where it has just been.
 */
import { createSourceGlyphs } from "../sources.js";

export const KIND_HEX = { near_voice: 0xf2a541, far_voice: 0xb48cf0, noise: 0x7fb7a4, listener: 0x4fd1e8 };
const STATION_HEX = { seat: 0x6fd18a, stand: 0xa3afbd, waypoint: 0x6d7a89 };
// Above the floor by this much, so a line on it is not lost in it.
const LIFT = 0.03;
const RAD = Math.PI / 180;

export function createSceneMarkers(THREE, viewport) {
  const group = new THREE.Group();
  group.name = "scene";
  const ground = new THREE.Group();
  const trails = new THREE.Group();
  const glyphs = createSourceGlyphs(THREE, viewport);
  group.add(ground, trails, glyphs.group);
  const lines = new Map();
  let floorY = 0;

  const drop = (holder) => {
    for (const child of [...holder.children]) {
      holder.remove(child);
      child.geometry.dispose();
      child.material.dispose();
    }
  };

  function railLines(rails, colour, opacity) {
    const points = [];
    for (const rail of rails) {
      for (let k = 0; k + 1 < rail.points.length; k++) {
        const [ax, az] = rail.points[k];
        const [bx, bz] = rail.points[k + 1];
        points.push(ax, floorY + LIFT, az, bx, floorY + LIFT, bz);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
    const material = new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity });
    return new THREE.LineSegments(geometry, material);
  }

  function stationPoints(stations, opacity) {
    const points = [];
    const colours = [];
    const colour = new THREE.Color();
    for (const station of stations) {
      points.push(station.position[0], floorY + LIFT, station.position[2]);
      colour.setHex(STATION_HEX[station.kind] || STATION_HEX.stand);
      colours.push(colour.r, colour.g, colour.b);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3));
    const material = new THREE.PointsMaterial({
      size: 7,
      sizeAttenuation: false,
      vertexColors: true,
      transparent: true,
      opacity,
    });
    return new THREE.Points(geometry, material);
  }

  return {
    group,
    /** The floor's furniture of a scene: `rest` is what the dwelling offers
     *  and the recipe leaves alone, `used` what it takes. */
    setGround({ floor_y_m, rest, used }) {
      drop(ground);
      floorY = floor_y_m;
      ground.add(railLines(rest.rails, 0x4a5868, 0.55), stationPoints(rest.stations, 0.45));
      ground.add(railLines(used.rails, 0xd0d8e2, 0.9), stationPoints(used.stations, 1));
    },
    clearGround() {
      drop(ground);
    },
    /** The scene's actors, `[{ id, kind }]`, each with room for `capacity` trail points. */
    setActors(actors, capacity) {
      drop(trails);
      lines.clear();
      glyphs.set(actors.map((actor) => ({ id: actor.id, position: [0, 0, 0], colour: KIND_HEX[actor.kind], facing: true })));
      for (const actor of actors) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * capacity), 3));
        geometry.setDrawRange(0, 0);
        const material = new THREE.LineBasicMaterial({
          color: KIND_HEX[actor.kind],
          transparent: true,
          opacity: 0.85,
          depthTest: false,
        });
        const line = new THREE.Line(geometry, material);
        line.frustumCulled = false;
        line.renderOrder = 3;
        trails.add(line);
        lines.set(actor.id, line);
      }
    },
    /** One actor at an instant; `trail` is `[x, y, z, ...]`, `points` of them, laid on the floor. */
    place(id, at, trail, points) {
      glyphs.place(id, at.x, at.y, at.z, at.yaw_deg * RAD, (at.pitch_deg || 0) * RAD);
      const line = lines.get(id);
      if (!line) return;
      const attribute = line.geometry.getAttribute("position");
      const count = Math.min(points, attribute.count);
      for (let k = 0; k < count; k++) attribute.setXYZ(k, trail[3 * k], floorY + LIFT, trail[3 * k + 2]);
      attribute.needsUpdate = true;
      line.geometry.setDrawRange(0, count);
    },
    /** `[{ id, on, dim }]`: `on` draws the actor at all, `dim` in its darker colour. */
    update(actors, selected) {
      glyphs.update(actors, selected);
      for (const actor of actors) {
        const line = lines.get(actor.id);
        if (line) line.visible = actor.on;
      }
    },
  };
}
