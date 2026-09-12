/** The first-person view: one scene, one camera, one walk.
 *
 * The camera *is* the listener. Its position and rotation are the pose every
 * other module reads, and the only ways it moves are the walk keys, a drag to
 * look, a wheel to zoom, or an explicit `moveTo` from the plan or the
 * listener tab. Nothing here knows about audio.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";

// Eye height of a standing adult, in metres. The default only: the height
// slider moves the listener up and down from here.
export const EYE_HEIGHT = 1.6;
const WALK_SPEED = 2.2;
// Stay this far from the walls, so the near plane never slips through a
// surface and the room cannot be left. A doorway is about 0.9 m wide, so the
// clearance has to be small enough to pass through one.
const WALL_MARGIN = 0.22;
const MAX_PITCH = Math.PI / 2 - 0.05;
const LOOK_SENSITIVITY = 0.0032;
// Radians per second when turning on the spot with the arrow keys.
const TURN_SPEED = 1.75;
// The widest the field of view ever gets. The wheel zooms in from here and
// stops at this value going back out, so a reader cannot roll themselves into
// a fish eye that misrepresents where anything is.
export const BASE_FOV = 70;
export const MIN_FOV = 12;
// Keep the listener's head this far from the floor and the ceiling.
const HEAD_CLEARANCE = 0.3;

function ringContains(x, z, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, zi] = ring[i];
    const [xj, zj] = ring[j];
    if (zi > z !== zj > z && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}

function partContains(x, z, part) {
  if (!ringContains(x, z, part.exterior)) return false;
  return !part.holes.some((hole) => ringContains(x, z, hole));
}

function ringClearance(x, z, ring) {
  let best = Infinity;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, zi] = ring[i];
    const [xj, zj] = ring[j];
    const dx = xj - xi;
    const dz = zj - zi;
    const lengthSquared = dx * dx + dz * dz;
    const t =
      lengthSquared === 0
        ? 0
        : Math.max(0, Math.min(1, ((x - xi) * dx + (z - zi) * dz) / lengthSquared));
    best = Math.min(best, Math.hypot(x - (xi + t * dx), z - (zi + t * dz)));
  }
  return best;
}

/** Which of `rooms` (manifest entries with `outline` rings) holds (x, z). */
export function roomAt(rooms, x, z) {
  for (const room of rooms || []) {
    for (const ring of room.outline || []) {
      if (ringContains(x, z, ring)) return room.name;
    }
  }
  return null;
}

export function createViewport(canvas, pane) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e12);
  const camera = new THREE.PerspectiveCamera(BASE_FOV, 1, 0.05, 200);
  camera.rotation.order = "YXZ";

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x555560, 1.4));
  const sun = new THREE.DirectionalLight(0xfff2e0, 1.8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0008;
  scene.add(sun);
  scene.add(sun.target);
  // A lamp travelling with the viewer keeps the interior readable: the
  // reconstructed ceiling blocks most of the directional light.
  const headLamp = new THREE.PointLight(0xffffff, 14, 20, 2);
  scene.add(headLamp);

  // HSSD stores its textures as KTX2/Basis. Decoding them here is why the
  // room is composed in the browser rather than merged into one glTF in
  // Python, where those textures are lost.
  const ktx2 = new KTX2Loader()
    .setTranscoderPath("https://unpkg.com/three@0.160.0/examples/jsm/libs/basis/")
    .detectSupport(renderer);
  const loader = new GLTFLoader().setKTX2Loader(ktx2);
  const loadGltf = (url) =>
    new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));

  let manifest = null;
  let yaw = 0;
  let pitch = 0;
  const groups = { colour: null, acoustic: null };
  let shown = null;
  const overlays = new THREE.Group();
  overlays.name = "overlays";
  scene.add(overlays);
  const resizeHandlers = [];
  const moveHandlers = [];
  // Rendered only when something changed, plus a slow heartbeat for what
  // arrives asynchronously (tiles, furniture): an idle scene costs nothing,
  // which keeps the rest of the page responsive.
  let dirty = true;
  let lastRenderAt = 0;
  const HEARTBEAT_S = 0.5;
  // Drag to look: the default turns the view with the mouse; "grab" drags the
  // picture with the cursor, as paper, so it goes the other way.
  let grab = false;

  function resize() {
    const { width, height } = pane.getBoundingClientRect();
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    for (const handler of resizeHandlers) handler(width, height);
    dirty = true;
  }
  addEventListener("resize", resize);
  resize();

  function canStandAt(x, z) {
    // Before the apartment is known there is nothing to bump into.
    if (!manifest) return true;
    for (const part of manifest.outline) {
      if (!partContains(x, z, part)) continue;
      const rings = [part.exterior, ...part.holes];
      return Math.min(...rings.map((ring) => ringClearance(x, z, ring))) >= WALL_MARGIN;
    }
    return false;
  }

  function heightRange() {
    if (!manifest) return [0.2, 3.0];
    return [manifest.floorHeight + HEAD_CLEARANCE, manifest.ceilingHeight - HEAD_CLEARANCE];
  }

  function startingPosition() {
    // Spawn at the most open point of the largest part, so a narrow doorway
    // or a cluttered corner never traps the viewer at start.
    let best = null;
    let bestClearance = -Infinity;
    for (const part of manifest.outline) {
      const xs = part.exterior.map((p) => p[0]);
      const zs = part.exterior.map((p) => p[1]);
      const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
      const [minZ, maxZ] = [Math.min(...zs), Math.max(...zs)];
      for (let i = 1; i < 30; i++) {
        for (let j = 1; j < 30; j++) {
          const x = minX + ((maxX - minX) * i) / 30;
          const z = minZ + ((maxZ - minZ) * j) / 30;
          if (!partContains(x, z, part)) continue;
          const rings = [part.exterior, ...part.holes];
          const clearance = Math.min(...rings.map((ring) => ringClearance(x, z, ring)));
          if (clearance > bestClearance) {
            bestClearance = clearance;
            best = [x, z];
          }
        }
      }
    }
    return best || [0, 0];
  }

  // --- input --------------------------------------------------------------
  const pressed = new Set();
  const typing = (event) => ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName);
  addEventListener("keydown", (event) => {
    if (typing(event)) return;
    if (event.code.startsWith("Arrow")) event.preventDefault();
    pressed.add(event.code);
  });
  addEventListener("keyup", (event) => pressed.delete(event.code));
  addEventListener("blur", () => pressed.clear());

  // Drag to look, deliberately not pointer lock: the cursor stays visible and
  // the page never captures the mouse, so the panels stay usable.
  // Drag to look, deliberately not pointer lock: the cursor stays visible and
  // the page never captures the mouse, so the panels stay usable.
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  });
  const endDrag = (event) => {
    dragging = false;
    canvas.classList.remove("dragging");
    if (event.pointerId !== undefined && canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const sign = grab ? 1 : -1;
    yaw += sign * (event.clientX - lastX) * LOOK_SENSITIVITY;
    pitch += sign * (event.clientY - lastY) * LOOK_SENSITIVITY;
    pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch));
    lastX = event.clientX;
    lastY = event.clientY;
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = Math.exp(event.deltaY * 0.0015);
      camera.fov = Math.min(BASE_FOV, Math.max(MIN_FOV, camera.fov * factor));
      camera.updateProjectionMatrix();
      dirty = true;
    },
    { passive: false }
  );

  // --- the apartment ------------------------------------------------------
  function applyDisplaySettings(root) {
    root.traverse((node) => {
      if (!node.isMesh) return;
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      node.material = materials.map((material) => {
        const copy = material.clone();
        // Standing inside the room means looking at the back of every wall,
        // floor and ceiling face, since the shell keeps its outward normals.
        copy.side = THREE.DoubleSide;
        return copy;
      });
      if (node.material.length === 1) node.material = node.material[0];
      node.castShadow = true;
      node.receiveShadow = true;
    });
  }

  async function buildColour(base) {
    const group = new THREE.Group();
    group.name = "colour";
    const shell = await loadGltf(`${base}shell_colour.glb`);
    applyDisplaySettings(shell.scene);
    group.add(shell.scene);
    const cache = new Map();
    for (const entry of manifest.instances) {
      const url = entry.render_url;
      if (!url) continue;
      if (!cache.has(url)) cache.set(url, loadGltf(base + url));
      const gltf = await cache.get(url);
      const piece = gltf.scene.clone(true);
      applyDisplaySettings(piece);
      // The instance matrix, applied whole: if a piece lands wrongly, the
      // reconstruction is wrong, because the viewer adds no correction.
      piece.matrixAutoUpdate = false;
      piece.matrix.fromArray(entry.matrix);
      group.add(piece);
    }
    return group;
  }

  function fitSunTo(group) {
    const box = new THREE.Box3().setFromObject(group);
    if (box.isEmpty()) return;
    const centre = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    sun.position.set(centre.x + size.x, box.max.y + size.y, centre.z + size.z);
    sun.target.position.copy(centre);
    const extent = Math.max(size.x, size.z);
    Object.assign(sun.shadow.camera, {
      left: -extent,
      right: extent,
      top: extent,
      bottom: -extent,
      far: extent * 4,
    });
    sun.shadow.camera.updateProjectionMatrix();
  }

  function show(name) {
    if (shown && groups[shown]) scene.remove(groups[shown]);
    shown = name;
    if (groups[name]) scene.add(groups[name]);
    dirty = true;
  }

  // --- the walk -----------------------------------------------------------
  const forward = new THREE.Vector3();
  const right = new THREE.Vector3();
  const clock = new THREE.Clock();
  // A glide in progress: from, to, start time and length, in seconds.
  let glide = null;
  const smoothstep = (t) => t * t * (3 - 2 * t);
  const last = { x: NaN, y: NaN, z: NaN, yaw: NaN, pitch: NaN, fov: NaN };

  function pose() {
    return {
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
      yaw,
      pitch,
      fov: camera.fov,
    };
  }

  function animate() {
    requestAnimationFrame(animate);
    const delta = Math.min(clock.getDelta(), 0.1);
    forward.set(-Math.sin(yaw), 0, -Math.cos(yaw));
    right.set(Math.cos(yaw), 0, -Math.sin(yaw));
    let stepX = 0;
    let stepZ = 0;
    const distance = WALK_SPEED * delta;
    const push = (vector, sign) => {
      stepX += vector.x * distance * sign;
      stepZ += vector.z * distance * sign;
    };
    // The arrows walk and turn; the letters strafe. Turning on the spot is
    // what a listener does to hear a source move.
    if (pressed.has("KeyW") || pressed.has("ArrowUp")) push(forward, 1);
    if (pressed.has("KeyS") || pressed.has("ArrowDown")) push(forward, -1);
    if (pressed.has("KeyD")) push(right, 1);
    if (pressed.has("KeyA")) push(right, -1);
    if (pressed.has("ArrowLeft")) yaw += TURN_SPEED * delta;
    if (pressed.has("ArrowRight")) yaw -= TURN_SPEED * delta;
    if (stepX !== 0 || stepZ !== 0 || pressed.size) glide = null;
    if (glide) {
      const t = Math.min(1, (performance.now() / 1000 - glide.started) / glide.seconds);
      const w = smoothstep(t);
      camera.position.set(
        glide.from.x + (glide.to.x - glide.from.x) * w,
        glide.from.y + (glide.to.y - glide.from.y) * w,
        glide.from.z + (glide.to.z - glide.from.z) * w
      );
      if (t >= 1) {
        const done = glide.onDone;
        glide = null;
        if (done) done();
      }
    }
    if (stepX !== 0 || stepZ !== 0) {
      const { x, z } = camera.position;
      if (canStandAt(x + stepX, z + stepZ)) {
        camera.position.x = x + stepX;
        camera.position.z = z + stepZ;
      } else if (canStandAt(x + stepX, z)) {
        camera.position.x = x + stepX; // slide along the wall
      } else if (canStandAt(x, z + stepZ)) {
        camera.position.z = z + stepZ;
      }
    }
    camera.rotation.set(pitch, yaw, 0);
    headLamp.position.copy(camera.position);

    const now = pose();
    if (Object.keys(last).some((key) => last[key] !== now[key])) {
      Object.assign(last, now);
      dirty = true;
      for (const handler of moveHandlers) handler(now);
    }
    const seconds = performance.now() / 1000;
    if (dirty || glide || pressed.size || seconds - lastRenderAt > HEARTBEAT_S) {
      renderer.render(scene, camera);
      lastRenderAt = seconds;
      dirty = false;
    }
  }
  animate();

  return {
    THREE,
    scene,
    camera,
    renderer,
    overlays,
    canStandAt,
    pose,
    onMove: (handler) => moveHandlers.push(handler),
    onResize: (handler) => resizeHandlers.push(handler),

    /** Glide the listener to a point over `ms`, easing in and out; any
     *  walking key, drag or `moveTo` interrupts it. */
    glideTo({ x, y, z }, ms, onDone) {
      glide = {
        from: camera.position.clone(),
        to: new THREE.Vector3(x, y, z),
        started: performance.now() / 1000,
        seconds: Math.max(0.05, ms / 1000),
        onDone,
      };
    },
    isGliding: () => glide !== null,

    /** Place the listener. Any field may be omitted; x and z are refused
     *  together when the point is not walkable. */
    moveTo({ x, y, z, yaw: newYaw, pitch: newPitch, fov } = {}) {
      if (x !== undefined || y !== undefined || z !== undefined) glide = null;
      if (x !== undefined || z !== undefined) {
        // One coordinate alone keeps the other; the walls still apply.
        const nx = x !== undefined ? x : camera.position.x;
        const nz = z !== undefined ? z : camera.position.z;
        if (canStandAt(nx, nz)) camera.position.set(nx, camera.position.y, nz);
      }
      if (y !== undefined) {
        const [low, high] = heightRange();
        camera.position.y = Math.min(high, Math.max(low, y));
      }
      if (newYaw !== undefined) yaw = newYaw;
      if (newPitch !== undefined) pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, newPitch));
      if (fov !== undefined) {
        camera.fov = Math.min(BASE_FOV, Math.max(MIN_FOV, fov));
        camera.updateProjectionMatrix();
      }
    },

    /** A new apartment: drop the old one, keep the listener if they are inside it. */
    async setApartment(newManifest, base) {
      manifest = newManifest;
      if (groups.colour) scene.remove(groups.colour);
      groups.colour = null;
      const { x, y, z } = camera.position;
      if (!canStandAt(x, z)) {
        const [sx, sz] = startingPosition();
        camera.position.set(sx, manifest.floorHeight + EYE_HEIGHT, sz);
        yaw = 0;
        pitch = 0;
      }
      // A height chosen before the floor was known is a guess; stand up.
      const [low, high] = heightRange();
      if (y < low || y > high) camera.position.y = manifest.floorHeight + EYE_HEIGHT;
      const group = await buildColour(base);
      // The apartment may have changed again during the loads.
      if (manifest !== newManifest) return;
      groups.colour = group;
      fitSunTo(group);
      if (shown === "colour") show("colour");
    },

    setAcoustic(group) {
      if (groups.acoustic) scene.remove(groups.acoustic);
      groups.acoustic = group;
      if (shown === "acoustic") show("acoustic");
    },

    show,
    resize,
    /** Something in the scene changed outside the camera: draw next frame. */
    invalidate: () => {
      dirty = true;
    },
    setGrab(on) {
      grab = Boolean(on);
    },
  };
}
