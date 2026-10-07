/** The scene as a listener needs it: the dwelling, his head, and who is sounding.
 *
 *   const view = createSceneView(element, scene);   // the answer of api/scene
 *   view.setLevels(levels);                         // { hop_s, stems: { id: [dB] } }
 *   view.setTime(seconds);
 *
 * A person is a head in his colour, with his name, turned the way he faces.
 * What is not a person is a marker: a cube, with its name. **Each lights up
 * while it sounds**, by the level of its own stem at that instant under its
 * fader (`gainOf`): dark at `LIGHT_DB[0]` and under, full at `LIGHT_DB[1]`.
 * With `lit: "intervals"` nothing is listened to and a source is lit while the
 * recipe says it sounds. The listener is the white head; his own voice has no
 * head of its own and lights his.
 *
 * **The people the listener talks with stand on a green ring**, joined to him
 * by a line, for as long as the recipe says they are of his conversation;
 * everybody else stands on nothing.
 *
 * The dwelling is `scene.plan` (`reverberate.viz.parts.scene.plan_of`): each
 * room's floor in a tint, the walls drawn low so that the rooms are seen
 * into, the seating as blocks. Without a plan the floor is the pack's cells
 * and no wall is drawn. Drag to look around, wheel to zoom.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

//: The level a light is dark at and the one it is full at, dB re full scale at the page's level.
export const LIGHT_DB = [-80, -45];
//: The colour of the listener's conversation, here and on a timeline.
export const WITH_YOU = "#22b866";
const RAD = Math.PI / 180;
const ROOM_TINTS = ["#d9d4c7", "#c9d6d3", "#d6cdd9", "#cfd8c6", "#dad0c4", "#c8cfdb", "#d8d3bd"];
//: How high a wall is drawn, metres: low, so that a room is seen into.
const WALL_M = 1.0;

function label(text, colour) {
  const canvas = document.createElement("canvas");
  const probe = canvas.getContext("2d");
  probe.font = "600 30px system-ui, sans-serif";
  const width = Math.ceil(probe.measureText(text).width) + 36;
  canvas.width = width;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  context.font = "600 30px system-ui, sans-serif";
  context.fillStyle = "rgba(20, 22, 26, 0.82)";
  context.beginPath();
  context.roundRect(0, 10, width, 44, 10);
  context.fill();
  context.fillStyle = colour;
  context.fillRect(0, 10, 8, 44);
  context.fillStyle = "#ffffff";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, width / 2 + 4, 33);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
  sprite.scale.set((0.4 * width) / 64, 0.4, 1);
  sprite.renderOrder = 2;
  return sprite;
}

/** A head facing +x, its centre at the origin; the material is the one that lights. */
function head(colour, size = 0.12) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color: colour, emissive: colour, emissiveIntensity: 0, roughness: 0.6 });
  const skull = new THREE.Mesh(new THREE.SphereGeometry(size, 24, 16), material);
  skull.scale.set(1.1, 1.25, 0.95);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(size * 0.3, size * 0.7, 12), material);
  nose.rotation.z = -Math.PI / 2;
  nose.position.set(size * 1.2, 0, 0);
  const ears = [-1, 1].map((side) => {
    const ear = new THREE.Mesh(new THREE.SphereGeometry(size * 0.3, 10, 8), material);
    ear.scale.set(0.6, 1, 0.4);
    ear.position.set(0, 0, side * size * 1.0);
    return ear;
  });
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.45, size * 0.9, size * 1.6, 12), material);
  neck.position.y = -size * 1.9;
  group.add(skull, nose, ...ears, neck);
  return { group, material };
}

/** What is not a person: a cube. */
function marker(colour, size = 0.14) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color: colour, emissive: colour, emissiveIntensity: 0, roughness: 0.5 });
  const box = new THREE.Mesh(new THREE.BoxGeometry(size * 1.6, size * 1.6, size * 1.6), material);
  box.rotation.y = Math.PI / 4;
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineBasicMaterial({ color: "#20242b" }));
  edges.rotation.y = Math.PI / 4;
  group.add(box, edges);
  return { group, material };
}

const at = (values, u) => {
  const last = values.length - 1;
  const k = Math.min(Math.floor(u), Math.max(last - 1, 0));
  const f = Math.min(Math.max(u - k, 0), 1);
  const a = values[Math.min(k, last)];
  const b = values[Math.min(k + 1, last)];
  return Array.isArray(a) ? a.map((value, i) => value + (b[i] - value) * f) : a + (b - a) * f;
};

/** The span of `spans` that holds `seconds`, or nothing. */
export const spanAt = (spans, seconds) => (spans || []).find((span) => seconds >= span.start_s && seconds < span.end_s) || null;

/** A flat shape of `[{ exterior, holes }]`, rings of `[x, z]`, laid at height `y`. */
function flat(parts, y, material) {
  const shapes = parts.map(({ exterior, holes }) => {
    const shape = new THREE.Shape(exterior.map(([x, z]) => new THREE.Vector2(x, z)));
    for (const hole of holes || []) shape.holes.push(new THREE.Path(hole.map(([x, z]) => new THREE.Vector2(x, z))));
    return shape;
  });
  const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shapes).rotateX(Math.PI / 2), material);
  mesh.position.y = y;
  return mesh;
}

/** Every ring of `parts` raised from `y` by `height`: a wall an edge. */
function raised(parts, y, height, material) {
  const corners = [];
  for (const { exterior, holes } of parts) {
    for (const ring of [exterior, ...(holes || [])]) {
      for (let k = 0; k + 1 < ring.length; k++) {
        const [ax, az] = ring[k];
        const [bx, bz] = ring[k + 1];
        corners.push(ax, y, az, bx, y, bz, bx, y + height, bz, ax, y, az, bx, y + height, bz, ax, y + height, az);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(corners, 3));
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, material);
}

/** The top edge of every ring of `parts`, as lines. */
function edges(parts, y, colour) {
  const points = [];
  for (const { exterior, holes } of parts) {
    for (const ring of [exterior, ...(holes || [])]) {
      for (let k = 0; k + 1 < ring.length; k++) points.push(ring[k][0], y, ring[k][1], ring[k + 1][0], y, ring[k + 1][1]);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: colour }));
}

export function createSceneView(element, scene, { gainOf = () => 1, lit: litBy = "levels" } = {}) {
  element.classList.add("p-scene");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  element.append(renderer.domElement);
  const world = new THREE.Scene();
  world.background = new THREE.Color("#15171c");
  world.add(new THREE.HemisphereLight("#ffffff", "#404550", 1.6));
  const sun = new THREE.DirectionalLight("#ffffff", 1.2);
  sun.position.set(3, 10, 4);
  world.add(sun);

  const box = new THREE.Box3();
  let floorY = 0;
  if (scene.plan) {
    // The dwelling: a floor a room, the walls low, the seating as blocks.
    floorY = scene.plan.floor_y;
    scene.plan.rooms.forEach((room, index) => {
      const tint = new THREE.MeshStandardMaterial({ color: ROOM_TINTS[index % ROOM_TINTS.length], roughness: 0.95, side: THREE.DoubleSide });
      world.add(flat(room.outline, floorY, tint));
      for (const part of room.outline) for (const [x, z] of part.exterior) box.expandByPoint(new THREE.Vector3(x, floorY, z));
    });
    const wall = new THREE.MeshStandardMaterial({ color: "#f1efe9", roughness: 0.9, side: THREE.DoubleSide, transparent: true, opacity: 0.5, depthWrite: false });
    world.add(raised(scene.plan.walls, floorY, WALL_M, wall), edges(scene.plan.walls, floorY + WALL_M, "#f7f5ef"), edges(scene.plan.walls, floorY + 0.005, "#6d6a62"));
    const seat = new THREE.MeshStandardMaterial({ color: "#8f8a80", roughness: 0.9, side: THREE.DoubleSide });
    for (const item of scene.plan.seating) {
      world.add(raised(item.footprint, floorY, 0.42, seat), flat(item.footprint, floorY + 0.42, seat));
    }
  } else if (scene.floor) {
    // Without a plan: a tile a cell the sound was computed at, tinted by room.
    const { cells, pitch_m: pitch, y } = scene.floor;
    floorY = y;
    const tiles = new THREE.InstancedMesh(
      new THREE.PlaneGeometry(pitch * 0.96, pitch * 0.96).rotateX(-Math.PI / 2),
      new THREE.MeshStandardMaterial({ roughness: 0.95 }),
      Math.max(cells.length, 1)
    );
    const matrix = new THREE.Matrix4();
    cells.forEach(([x, z, room], index) => {
      matrix.makeTranslation(x, floorY, z);
      tiles.setMatrixAt(index, matrix);
      tiles.setColorAt(index, new THREE.Color(ROOM_TINTS[room % ROOM_TINTS.length]));
      box.expandByPoint(new THREE.Vector3(x, floorY, z));
    });
    world.add(tiles);
  }

  // What lies on the floor is drawn over the seating, and under the heads: a ring is seen
  // where somebody sits.
  const ringAt = (colour, inner = 0.28, outer = 0.34) => {
    const mesh = new THREE.Mesh(new THREE.RingGeometry(inner, outer, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: colour, depthWrite: false, depthTest: false }));
    mesh.renderOrder = 1;
    return mesh;
  };
  const listener = head("#f4f4f4", 0.13);
  const ring = ringAt("#ffffff");
  const ownHalo = new THREE.Mesh(
    new THREE.CircleGeometry(0.45, 40).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0, depthWrite: false, depthTest: false })
  );
  const you = label("You", "#ffffff");
  listener.group.renderOrder = 2;
  world.add(listener.group, ring, ownHalo, you);

  const sources = scene.sources.map((source) => {
    if (source.shape === "self") return { source, material: listener.material, halo: ownHalo, lit: 0 };
    const made = source.shape === "marker" ? marker(source.colour) : head(source.colour);
    const halo = new THREE.Mesh(
      new THREE.CircleGeometry(0.45, 40).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: source.colour, transparent: true, opacity: 0, depthWrite: false, depthTest: false })
    );
    const name = label(source.label || source.id, source.colour);
    // Of the listener's conversation: a ring at the feet and a line to him.
    const withYou = ringAt(WITH_YOU, 0.36, 0.44);
    const link = new THREE.Line(
      new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3)),
      new THREE.LineBasicMaterial({ color: WITH_YOU, depthTest: false })
    );
    link.frustumCulled = false;
    link.renderOrder = 1;
    made.group.renderOrder = 2;
    withYou.visible = link.visible = false;
    world.add(made.group, halo, name, withYou, link);
    return { source, ...made, halo, name, withYou, link, lit: 0, inside: false };
  });

  // The camera frames where the listener and the sources are over the scene, not the storey.
  const action = new THREE.Box3();
  for (const track of [scene.listener, ...scene.sources]) {
    for (const place of track.position) action.expandByPoint(new THREE.Vector3(...place));
  }
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  const framed = action.isEmpty() ? box : action;
  const centre = framed.getCenter(new THREE.Vector3());
  centre.y = floorY + 0.8;
  const reach = Math.max(framed.getSize(new THREE.Vector3()).length(), 5);
  camera.position.set(centre.x + reach * 0.08, floorY + reach * 0.85, centre.z + reach * 0.62);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(centre);
  controls.maxPolarAngle = Math.PI / 2 - 0.05;
  controls.update();

  let levels = null;
  let seconds = 0;
  let turned = 0;
  let headYaw = null;

  function levelOf(source) {
    if (litBy === "intervals") return source.intervals.some((span) => seconds >= span.start_s && seconds < span.end_s) ? 1 : 0;
    const values = levels && levels.stems[source.id];
    if (!values) return 0;
    const value = values[Math.min(Math.floor(seconds / levels.hop_s), values.length - 1)];
    if (value === null || value === undefined) return 0;
    const db = value + 20 * Math.log10(Math.max(gainOf(source.id), 1e-6));
    return Math.max(0, Math.min(1, (db - LIGHT_DB[0]) / (LIGHT_DB[1] - LIGHT_DB[0])));
  }

  function place() {
    const u = seconds / scene.step_s;
    const here = at(scene.listener.position, u);
    listener.group.position.set(...here);
    listener.group.rotation.y = (headYaw === null ? at(scene.listener.yaw_deg, u) + turned : headYaw) * RAD;
    ring.position.set(here[0], floorY + 0.012, here[2]);
    ownHalo.position.set(here[0], floorY + 0.02, here[2]);
    you.position.set(here[0], here[1] + 0.45, here[2]);
    const talking = spanAt(scene.listener.conversation, seconds);
    ring.material.color.set(talking && talking.members.length ? WITH_YOU : "#ffffff");
    for (const entry of sources) {
      const { source } = entry;
      // A light comes on at once and goes out over a few frames, as a meter does.
      entry.lit = Math.max(levelOf(source), entry.lit * 0.85);
      entry.material.emissiveIntensity = (source.shape === "self" ? 0.9 : 1.6) * entry.lit;
      entry.halo.material.opacity = 0.55 * entry.lit;
      entry.halo.scale.setScalar(0.6 + 1.2 * entry.lit);
      if (source.shape === "self") continue;
      const there = at(source.position, u);
      entry.group.position.set(...there);
      if (source.shape !== "marker") entry.group.rotation.y = at(source.yaw_deg, u) * RAD;
      entry.halo.position.set(there[0], floorY + 0.02, there[2]);
      entry.name.position.set(there[0], there[1] + 0.42, there[2]);
      entry.inside = Boolean(talking && talking.members.includes(source.id));
      entry.withYou.visible = entry.link.visible = entry.inside;
      if (entry.inside) {
        entry.withYou.position.set(there[0], floorY + 0.014, there[2]);
        const line = entry.link.geometry.attributes.position;
        line.setXYZ(0, here[0], floorY + 0.03, here[2]);
        line.setXYZ(1, there[0], floorY + 0.03, there[2]);
        line.needsUpdate = true;
      }
    }
  }

  function resize() {
    const width = element.clientWidth || 600;
    const height = element.clientHeight || 400;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(element);
  resize();

  let alive = true;
  (function draw() {
    if (!alive) return;
    place();
    controls.update();
    renderer.render(world, camera);
    requestAnimationFrame(draw);
  })();

  return {
    setTime(value) {
      seconds = value;
    },
    setLevels(answer) {
      levels = answer;
    },
    /** What the listener adds to the scene's head, degrees to his left: his head is drawn turned. */
    setTurn(degrees) {
      turned = degrees;
    },
    /** The head drawn at this yaw, degrees in the scene's frame, whatever the scene says; `null` for the scene's. */
    setHead(degrees) {
      headYaw = degrees;
    },
    /** How lit each source is now, 0 to 1: what the picture shows, for a test or a legend. */
    lights: () => Object.fromEntries(sources.map((entry) => [entry.source.id, entry.lit])),
    /** What is drawn now: the listener's place and yaw, and who stands on a ring. */
    drawn: () => ({
      listener: { position: listener.group.position.toArray(), yaw_deg: listener.group.rotation.y / RAD },
      with_you: sources.filter((entry) => entry.inside).map((entry) => entry.source.id),
      sources: Object.fromEntries(sources.filter((entry) => entry.group).map((entry) => [entry.source.id, { position: entry.group.position.toArray(), yaw_deg: entry.group.rotation.y / RAD }])),
    }),
    dispose() {
      alive = false;
      renderer.dispose();
    },
  };
}
