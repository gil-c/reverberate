/** The scene as a listener needs it: the dwelling's floor, his head, and who is speaking.
 *
 *   const view = createSceneView(element, scene);   // the answer of api/scene
 *   view.setLevels(levels);                         // the answer of api/levels
 *   view.setTime(seconds);
 *
 * A voice is a head in its colour, with its name, turned the way it faces. A
 * noise is not a head: a cube on a stand, grey. **Each lights up while it
 * sounds**, by the level of its own stem at that instant under its fader
 * (`gainOf`): dark at `LIGHT_DB[0]` and under, full at `LIGHT_DB[1]`. The
 * listener is the white head with a ring at his feet.
 *
 * The floor is the pack's cells, a tile each, tinted by room
 * (`reverberate.viz.parts.scene`). No wall is drawn: none is in the pack.
 * Drag to orbit, wheel to zoom.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

//: The level a light is dark at and the one it is full at, dB re full scale at the page's level.
export const LIGHT_DB = [-80, -45];
const RAD = Math.PI / 180;
const ROOM_TINTS = ["#d9d4c7", "#c9d6d3", "#d6cdd9", "#cfd8c6", "#dad0c4", "#c8cfdb"];

function label(text, colour) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  context.font = "600 30px system-ui, sans-serif";
  const width = Math.min(context.measureText(text).width + 28, 256);
  context.fillStyle = "rgba(20, 22, 26, 0.82)";
  context.beginPath();
  context.roundRect((256 - width) / 2, 10, width, 44, 10);
  context.fill();
  context.fillStyle = colour;
  context.fillRect((256 - width) / 2, 10, 8, 44);
  context.fillStyle = "#ffffff";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 128 + 4, 33);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
  sprite.scale.set(1.2, 0.3, 1);
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

/** What is not a voice: a cube on a stand. */
function marker(colour, size = 0.14) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color: colour, emissive: "#ffd27a", emissiveIntensity: 0, roughness: 0.5 });
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

export function createSceneView(element, scene, { gainOf = () => 1 } = {}) {
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

  // The floor: a tile a cell, tinted by room.
  const { cells, pitch_m: pitch, y: floorY } = scene.floor;
  const tiles = new THREE.InstancedMesh(
    new THREE.PlaneGeometry(pitch * 0.96, pitch * 0.96).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ roughness: 0.95 }),
    Math.max(cells.length, 1)
  );
  const matrix = new THREE.Matrix4();
  const box = new THREE.Box3();
  cells.forEach(([x, z, room], index) => {
    matrix.makeTranslation(x, floorY, z);
    tiles.setMatrixAt(index, matrix);
    tiles.setColorAt(index, new THREE.Color(ROOM_TINTS[room % ROOM_TINTS.length]));
    box.expandByPoint(new THREE.Vector3(x, floorY, z));
  });
  world.add(tiles);
  for (const place of scene.listener.position) box.expandByPoint(new THREE.Vector3(...place));

  const listener = head("#f4f4f4", 0.13);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.28, 0.34, 40).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: "#ffffff" })
  );
  const you = label("you", "#ffffff");
  world.add(listener.group, ring, you);

  const sources = scene.sources.map((source) => {
    const made = source.kind === "noise" ? marker(source.colour) : head(source.colour);
    const halo = new THREE.Mesh(
      new THREE.CircleGeometry(0.45, 40).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: source.kind === "noise" ? "#ffd27a" : source.colour, transparent: true, opacity: 0, depthWrite: false })
    );
    const name = label(source.id, source.colour);
    world.add(made.group, halo, name);
    return { source, ...made, halo, name, lit: 0 };
  });

  // The camera frames where the listener and the sources are over the window, not the storey.
  const action = new THREE.Box3();
  for (const track of [scene.listener, ...scene.sources]) {
    for (const place of track.position) action.expandByPoint(new THREE.Vector3(...place));
  }
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  const centre = (action.isEmpty() ? box : action).getCenter(new THREE.Vector3());
  const reach = Math.max((action.isEmpty() ? box : action).getSize(new THREE.Vector3()).length(), 5);
  camera.position.set(centre.x + reach * 0.1, floorY + reach * 0.75, centre.z + reach * 0.7);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(centre);
  controls.maxPolarAngle = Math.PI / 2 - 0.05;
  controls.update();

  let levels = null;
  let seconds = 0;
  let turned = 0;

  function place() {
    const u = seconds / scene.step_s;
    const here = at(scene.listener.position, u);
    listener.group.position.set(...here);
    listener.group.rotation.y = (at(scene.listener.yaw_deg, u) + turned) * RAD;
    ring.position.set(here[0], floorY + 0.01, here[2]);
    you.position.set(here[0], here[1] + 0.45, here[2]);
    for (const entry of sources) {
      const { source } = entry;
      const there = at(source.position, u);
      entry.group.position.set(...there);
      if (source.kind !== "noise") entry.group.rotation.y = at(source.yaw_deg, u) * RAD;
      entry.halo.position.set(there[0], floorY + 0.02, there[2]);
      entry.name.position.set(there[0], there[1] + 0.42, there[2]);
      let lit = 0;
      const values = levels && levels.stems[source.id];
      if (values) {
        const gain = gainOf(source.id);
        const db = values[Math.min(Math.floor(seconds / levels.hop_s), values.length - 1)] + 20 * Math.log10(Math.max(gain, 1e-6));
        lit = Math.max(0, Math.min(1, (db - LIGHT_DB[0]) / (LIGHT_DB[1] - LIGHT_DB[0])));
      }
      // A light comes on at once and goes out over a few frames, as a meter does.
      entry.lit = Math.max(lit, entry.lit * 0.85);
      entry.material.emissiveIntensity = 1.6 * entry.lit;
      entry.halo.material.opacity = 0.55 * entry.lit;
      entry.halo.scale.setScalar(0.6 + 1.2 * entry.lit);
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
    /** How lit each source is now, 0 to 1: what the picture shows, for a test or a legend. */
    lights: () => Object.fromEntries(sources.map((entry) => [entry.source.id, entry.lit])),
    dispose() {
      alive = false;
      renderer.dispose();
    },
  };
}
