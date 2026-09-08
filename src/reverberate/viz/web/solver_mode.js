/**
 * The solver mode of the apartment viewer: what entered and left the wave run.
 *
 * The other two modes render the authored HSSD scene. This one renders the
 * serialised surface list the solver actually read, so a picture and a response
 * can be shown to be of the same object. It is a separate module because it is
 * the only mode that needs signal plots and audio, not because it is a separate
 * application: it shares the apartment selector, the camera and the canvas.
 *
 * No signal processing happens here. The envelopes, spectrograms and decay
 * curves arrive precomputed from the tested Python path; this file draws them.
 */

const escapeHtml = (value) =>
  String(value).replace(/[<&>]/g, (ch) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[ch]);

export async function loadRun(url) {
  const response = await fetch(`${url}/run.json`);
  if (!response.ok) throw new Error(`run payload ${response.status}`);
  const data = await response.json();
  data.baseUrl = url;
  return data;
}

/** Build the solver's surfaces, plus a marker per source and receiver. */
export function buildRunGroup(THREE, data) {
  const group = new THREE.Group();
  // Kept apart from the markers: the surfaces and the voxel cloud describe the
  // same boundary, so showing both puts opaque triangles exactly where the
  // points are and the points lose. They are exclusive, and this is the handle
  // that lets the caller switch.
  const surfaces = [];

  for (const entry of data.groups) {
    if (!entry.indices.length) continue;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(entry.positions), 3)
    );
    geometry.setIndex(entry.indices);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(...entry.colour.map((v) => v / 255)),
      // The solver's shell keeps its outward normals, so standing inside means
      // looking at the back of every face.
      side: THREE.DoubleSide,
      roughness: 1,
      metalness: 0,
      transparent: true,
      opacity: 0.92,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = entry.label;
    surfaces.push(mesh);
    group.add(mesh);
  }

  const bounds = new THREE.Box3().setFromObject(group);
  const size = bounds.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z) || 1;
  const radius = Math.max(span * 0.012, 0.03);

  const SOURCE = 0xff6b4a;
  const RECEIVER = 0x4ac1ff;
  const PICKED = 0xffe066;
  // The markers live in their own group, not in the surfaces one. They belong
  // to every mode: a reader listening to two head orientations in the colour
  // view needs to see where the source and the listening point are, and a run
  // whose geometry is only visible from inside the acoustic mode is one that
  // can be heard but not placed.
  const markers = new THREE.Group();
  const marker = (position, colour, isSource) => {
    const geometry = isSource
      ? new THREE.SphereGeometry(radius, 20, 14)
      : new THREE.BoxGeometry(radius * 1.6, radius * 1.6, radius * 1.6);
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: colour }));
    mesh.position.fromArray(position);
    // Drawn through whatever is in front of them: the point of a marker is to
    // say where something is, and a source behind a wardrobe is still where the
    // sound came from.
    mesh.material.depthTest = false;
    mesh.renderOrder = 3;
    markers.add(mesh);
    return mesh;
  };
  const sources = data.sources.map((s) => marker(s.position, SOURCE, true));
  const receivers = data.receivers.map((r) => marker(r.position, RECEIVER, false));

  // The ball the field was expanded about, drawn at its own radius. A spatial
  // run is about one point, and a point on the screen says nothing about how
  // much of the room the encoder actually looked at: 16 cm of it, which is a
  // head's width and is the reason the interior expansion is valid at all. The
  // outer shell is drawn as a wire sphere and the inner ones as rings, so the
  // shells that carry the top of the band are visible as well as the one that
  // carries the bottom.
  const array = data.array || null;
  if (array && array.centre) {
    const centre = new THREE.Vector3().fromArray(array.centre);
    const outer = Number(array.outer_radius_m) || 0.16;
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(outer, 32, 20),
      new THREE.MeshBasicMaterial({ color: 0x8be9c0, wireframe: true, transparent: true, opacity: 0.35 }),
    );
    ball.position.copy(centre);
    // Depth tested, unlike the point markers: it is a metre wide on screen when
    // you stand next to it, and drawing it through the walls would put a green
    // cage over the whole room. It also hides itself when the camera is inside
    // it, which is exactly where the listener stands: from outside it says how
    // much of the room the encoder looked at, and from inside it is in the way
    // of the thing it describes.
    ball.material.depthTest = true;
    ball.userData.hideWithin = outer;
    markers.add(ball);
    for (const shell of array.shells || []) {
      const radius = Number(shell.nominal_radius_m);
      if (!(radius > 0) || radius >= outer) continue;
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius, Math.max(radius * 0.012, 0.0015), 8, 48),
        new THREE.MeshBasicMaterial({ color: 0x8be9c0, transparent: true, opacity: 0.5 }),
      );
      ring.position.copy(centre);
      ring.rotation.x = Math.PI / 2;
      markers.add(ring);
    }
  }

  const highlight = (sourceIndex, receiverIndex) => {
    sources.forEach((m, i) => m.material.color.setHex(i === sourceIndex ? PICKED : SOURCE));
    receivers.forEach((m, i) => m.material.color.setHex(i === receiverIndex ? PICKED : RECEIVER));
  };

  return { group, markers, highlight, bounds, receivers, sources, surfaces };
}

// ------------------------------------------------------------------- plots
function fit(canvas) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, canvas.clientWidth * ratio);
  canvas.height = Math.max(1, canvas.clientHeight * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return [context, canvas.clientWidth, canvas.clientHeight];
}

function drawEnvelope(canvas, pairs) {
  const [c, w, h] = fit(canvas);
  c.clearRect(0, 0, w, h);
  const peak = Math.max(1e-9, ...pairs.map(([lo, hi]) => Math.max(Math.abs(lo), Math.abs(hi))));
  c.strokeStyle = "#2b3038";
  c.beginPath();
  c.moveTo(0, h / 2);
  c.lineTo(w, h / 2);
  c.stroke();
  c.strokeStyle = "#4ac1ff";
  c.beginPath();
  pairs.forEach(([lo, hi], i) => {
    const x = (i / Math.max(1, pairs.length - 1)) * w;
    c.moveTo(x, h / 2 - (hi / peak) * (h / 2 - 2));
    c.lineTo(x, h / 2 - (lo / peak) * (h / 2 - 2));
  });
  c.stroke();
}

function drawDecay(canvas, decay) {
  const [c, w, h] = fit(canvas);
  c.clearRect(0, 0, w, h);
  const { seconds, db } = decay;
  if (!seconds.length) return;
  const tMax = seconds[seconds.length - 1] || 1;
  const floor = -70;
  const x = (t) => (t / tMax) * w;
  const y = (v) => (1 - (Math.max(v, floor) - floor) / -floor) * (h - 4) + 2;
  c.strokeStyle = "#2b3038";
  c.fillStyle = "#6b7280";
  c.font = "10px system-ui";
  for (const level of [-10, -30, -60]) {
    c.beginPath();
    c.moveTo(0, y(level));
    c.lineTo(w, y(level));
    c.stroke();
    c.fillText(`${level} dB`, 3, y(level) - 2);
  }
  c.strokeStyle = "#8be08b";
  c.lineWidth = 1.5;
  c.beginPath();
  seconds.forEach((t, i) => (i ? c.lineTo(x(t), y(db[i])) : c.moveTo(x(t), y(db[i]))));
  c.stroke();
  c.lineWidth = 1;
}

function drawSpectrogram(canvas, spec) {
  const [c, w, h] = fit(canvas);
  c.clearRect(0, 0, w, h);
  const raw = atob(spec.data);
  const { bins, frames } = spec;
  const image = new ImageData(frames, bins);
  for (let row = 0; row < bins; row += 1) {
    for (let column = 0; column < frames; column += 1) {
      // Rows arrive low frequency first; flip so low sits at the bottom.
      const level = raw.charCodeAt(row * frames + column);
      const target = ((bins - 1 - row) * frames + column) * 4;
      // Blue through green to white: dark where there is nothing, bright where
      // the energy is, and monotonic in level throughout.
      image.data[target] = level < 128 ? level * 0.5 : (level - 128) * 2;
      image.data[target + 1] = level < 128 ? level * 1.6 : 205 + (level - 128) * 0.4;
      image.data[target + 2] = level < 128 ? 40 + level : 255 - (level - 128) * 0.2;
      image.data[target + 3] = 255;
    }
  }
  // Drawn at native size into an offscreen buffer, then scaled to the canvas,
  // so the browser does the interpolation rather than us.
  const buffer = new OffscreenCanvas(frames, bins);
  buffer.getContext("2d").putImageData(image, 0, 0);
  c.imageSmoothingEnabled = true;
  c.drawImage(buffer, 0, 0, w, h);

  // The frequency axis, drawn on the picture. Without it a reader can see that
  // the energy sits low and cannot say how low, which is the difference
  // between "the band stops at 4 kHz" and "the band runs to 16 kHz and the top
  // of it decays in 30 ms". The band limit is drawn as a line of its own,
  // because everything above it is the encoder's own zero and not the room's
  // silence.
  const top = spec.max_hz || 0;
  if (!top) return;
  const y = (hz) => h - (hz / top) * h;
  c.font = "10px system-ui";
  c.textBaseline = "middle";
  for (const hz of [4000, 8000, 16000]) {
    if (hz > top) continue;
    const line = y(hz);
    c.strokeStyle = "#ffffff55";
    c.setLineDash(hz === (spec.band_limit_hz || 0) ? [] : [3, 4]);
    c.beginPath();
    c.moveTo(0, line);
    c.lineTo(w, line);
    c.stroke();
    c.setLineDash([]);
    const label = `${hz / 1000} kHz`;
    c.fillStyle = "#00000099";
    const width = c.measureText(label).width + 6;
    c.fillRect(2, line - 7, width, 14);
    c.fillStyle = "#e6e9ee";
    c.fillText(label, 5, line);
  }
}

// ------------------------------------------------------------------- panel
/**
 * Fill `element` with the run's panel and return `show(index)`.
 *
 * `onSelect` is called with the chosen sample so the host can highlight the
 * markers, and `onStand` is offered as a button so the listener can put the
 * camera exactly where the receiver they are hearing was.
 */
/** The voxelisation, as a merged mesh, or null when there is no grid on disk.
 *
 * This is the last picture in the chain and the only one taken after the
 * solver's own reading of the scene: triangles say what was sent, this says
 * what was received. Blocks holding no material at all get a colour of their
 * own, because those are the sealed insides of solid objects and sealing stops
 * the simulation carrying sound through a region.
 *
 * Faces between touching blocks are not drawn and coplanar faces of the same
 * material are merged into rectangles, on the Python side. A bedroom at 4 mm
 * is 14 million blocks -- 167 M triangles as solid cubes -- and reaches here
 * as 951 480, without one face moving.
 */
export async function buildVoxelCloud(THREE, data) {
  const meta = data.voxels;
  if (!meta) return null;
  return fetchQuadMesh(THREE, data.baseUrl, meta);
}

//: The sealed inside of a solid object, which the solver carries no sound
//: through, and the far side of a boundary the room cannot hear.
//:
//: **Both are reserved, and the first version's was not.** It painted sealed
//: at (0.85, 0.25, 0.55) and walked the material hues by the golden ratio at
//: one lightness, which put ``shell`` -- the walls, a third of the drawn area
//: of every room -- at (0.83, 0.27, 0.72), a distance of 0.036 in RGB. A
//: reader standing in the bedroom could not tell a wall from the inside of one,
//: which is exactly the judgement this view exists to support. These are the
//: furthest usable pair from the material palette below: 0.379 and 0.372,
//: against 0.134 before, and 0.563 from each other.
const SEALED_RGB = [0.48, 0.03, 0.14];
const RIGID_RGB = [0.45, 0.45, 0.48];

/** A material's colour: seventeen hues over three lightnesses.
 *
 * Stable across runs, because the label list is sorted before it is written and
 * the index is a position in it.
 *
 * Not a golden-ratio walk over one lightness, which is what this replaces. With
 * this flat's fifty-one materials that walk spaces hues 0.02 apart and puts the
 * closest pair 0.018 apart in RGB; spreading the same count over three
 * lightnesses gives 0.066, which is 3.7 times better. **It is still not enough
 * to name a material by its colour at fifty-one of them**, and the page says so:
 * the palette separates a floor from a sofa, and the legend is what names them.
 */
function materialColour(rgb, index) {
  const hue = (index % 17) / 17;
  const lightness = [0.78, 0.62, 0.46][Math.floor(index / 17) % 3];
  return rgb.setHSL(hue, 0.6, lightness);
}

/** One payload of merged quads, as a mesh: fetch the three arrays and shade them.
 *
 * Shared by the single-payload view and the tiered audit view, because the
 * colour *is* the datum and two copies of this would be two chances for a
 * material to be drawn one hue in one tier and another hue in the next.
 */
export async function fetchQuadMesh(THREE, base, meta) {
  const [corners, index, label] = await Promise.all([
    fetch(`${base}/${meta.corners_url}`).then((r) => r.arrayBuffer()),
    fetch(`${base}/${meta.index_url}`).then((r) => r.arrayBuffer()),
    fetch(`${base}/${meta.label_url}`).then((r) => r.arrayBuffer()),
  ]);
  const position = new Float32Array(corners);
  const labels = new Int16Array(label);

  // Unlit, with the shading in the vertex colours.
  //
  // The scene's lights are tuned for the textured HSSD modes -- hemisphere
  // 1.4, sun 1.8, a head lamp at 14 -- where light makes the realism. Here the
  // colour *is* the datum: it says which material a face carries, and any
  // albedo above about 0.4 washed out under that rig, which is the one failure
  // a data view cannot have. So the face's own normal decides its brightness,
  // and nothing can exceed the albedo.
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
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(index), 1));
  geometry.computeVertexNormals();

  // Shade from the normal rather than from a light, so a face keeps its own
  // colour and only its orientation changes how bright it is.
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader.replace(
      "#include <color_vertex>",
      `#include <color_vertex>
       vec3 n = normalize(normalMatrix * normal);
       vColor.rgb *= 0.55 + 0.45 * abs(n.y) + 0.15 * abs(n.x);`
    );
  };
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "voxels";
  return mesh;
}

/** The tiered audit view: the room you stand in at the grid's own step.
 *
 * A whole flat at 16 kHz is 1 089 464 499 boundary nodes and roughly 28 M
 * merged quads, about 2.2 GB. No browser holds that, and thinning it would
 * make the picture a sample rather than an audit. So the flat is published
 * room by room in two tiers of the *same* grid -- the solver's own step, and
 * that step aggregated to the cell size of the band below -- and this draws the
 * coarse tier everywhere plus the fine tier of one room.
 *
 * Nothing is loaded that is not being looked at. Coarse tiers come in at the
 * start because together they are tens of megabytes; a fine tier is fetched
 * the first time its room is entered, tile by tile, nearest first, and kept.
 */
//: The most quads the fine tier may hold at once, across every tile drawn.
//:
//: Sixteen million, which is enough that **every room of this flat draws
//: whole**: the largest is the living room at 13.7 M quads, 15.25 M with the
//: coarse tier of everywhere else, about 1.2 GB of buffers. Measured on an
//: M-series laptop and it holds.
//:
//: Set for that deliberately, against a cheaper eight million. At eight the
//: living room drew 17 of its 28 tiles and the ceiling had holes in it, and on
//: an audit view a hole is the worst thing a picture can have: it is
//: indistinguishable from geometry the voxeliser lost, which is the exact
//: judgement a reader is here to make. Lower it if a card cannot hold this,
//: and the status line will say how many tiles of the room are drawn.
const FINE_QUAD_BUDGET = 16_000_000;

export async function buildAuditGrid(THREE, data, onStatus) {
  const audit = data.audit;
  if (!audit || !audit.rooms || !audit.rooms.length) return null;
  const base = `${data.baseUrl}/${audit.dir || "voxels"}`;
  const group = new THREE.Group();
  group.name = "audit-grid";

  const state = audit.rooms.map((room) => ({
    room,
    coarse: null,
    // One entry per fine tile: its mesh once fetched, and the promise while it
    // is in flight, so walking in and out of a room does not fetch it twice.
    tiles: room.fine.tiles.map((file) => ({ file, mesh: null, loading: null, drawn: false })),
  }));
  let selected = null;
  let drawnQuads = 0;

  const say = () => {
    if (!onStatus) return;
    const entry = selected === null ? null : state[selected];
    const coarse = state.reduce(
      (sum, other, i) => sum + (i === selected ? 0 : other.room.coarse.quads),
      0
    );
    onStatus({
      room: entry ? entry.room.name : null,
      fine_mm: entry ? entry.room.fine.cell_m * 1000 : null,
      coarse_mm: audit.rooms[0].coarse.cell_m * 1000,
      tiles_drawn: entry ? entry.tiles.filter((tile) => tile.drawn).length : 0,
      tiles: entry ? entry.tiles.length : 0,
      quads: coarse + drawnQuads,
      megabytes: Math.round(((coarse + drawnQuads) * 80) / 1e6),
    });
  };

  // Every room's coarse tier, up front: this is the picture of the whole flat
  // and it is what a reader sees before choosing anywhere to stand. On this
  // scene it is 2.57 M quads, which is about what the 4 kHz whole-flat view
  // already draws.
  for (const entry of state) {
    const holder = new THREE.Group();
    holder.name = `${entry.room.name}-coarse`;
    for (const file of entry.room.coarse.tiles) {
      holder.add(await fetchQuadMesh(THREE, `${base}/${entry.room.dir}`, file));
    }
    entry.coarse = holder;
    group.add(holder);
  }
  say();

  /** The distance from a point to a tile's own bounding box, zero inside it. */
  const reach = (file, at) => {
    const [lo, hi] = file.bounds;
    let sum = 0;
    for (let axis = 0; axis < 3; axis++) {
      const value = at.getComponent(axis);
      const gap = Math.max(lo[axis] - value, 0, value - hi[axis]);
      sum += gap * gap;
    }
    return sum;
  };

  /** Draw as much of the selected room's fine tier as the budget allows.
   *
   * Nearest first, because a reader auditing a room is looking at the wall in
   * front of them. A tile already fetched is kept in memory and only hidden, so
   * turning round is free after the first pass.
   */
  const refresh = async (at) => {
    if (selected === null) return;
    const entry = state[selected];
    const order = entry.tiles
      .map((tile, index) => ({ index, far: reach(tile.file, at) }))
      .sort((left, right) => left.far - right.far);

    let spent = 0;
    const wanted = new Set();
    for (const { index } of order) {
      const quads = Number(entry.tiles[index].file.quads);
      if (spent && spent + quads > FINE_QUAD_BUDGET) break;
      spent += quads;
      wanted.add(index);
    }

    for (const [index, tile] of entry.tiles.entries()) {
      if (!wanted.has(index)) {
        if (tile.mesh) tile.mesh.visible = false;
        tile.drawn = false;
        continue;
      }
      if (!tile.mesh) {
        tile.loading =
          tile.loading || fetchQuadMesh(THREE, `${base}/${entry.room.dir}`, tile.file);
        const mesh = await tile.loading;
        // The reader may have walked into another room while this was in
        // flight; the tile is kept, but it is not put on screen.
        if (selected === null || state[selected] !== entry) return;
        tile.mesh = mesh;
        group.add(mesh);
      }
      tile.mesh.visible = true;
      tile.drawn = true;
    }
    drawnQuads = spent;
    entry.coarse.visible = false;
    say();
  };

  /** Draw one room at the grid's own step, and everything else coarse. */
  const select = async (name, at) => {
    const next = state.findIndex((entry) => entry.room.name === name);
    if (next < 0 || next === selected) return;
    if (selected !== null) {
      const previous = state[selected];
      previous.coarse.visible = true;
      for (const tile of previous.tiles) {
        if (tile.mesh) tile.mesh.visible = false;
        tile.drawn = false;
      }
    }
    selected = next;
    drawnQuads = 0;
    say();
    await refresh(at);
  };

  /** Which room a point on the floor is in, or null outside every outline. */
  const roomAt = (x, z) => {
    for (const entry of state) {
      for (const ring of entry.room.outline || []) {
        let inside = false;
        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
          const [xi, zi] = ring[i];
          const [xj, zj] = ring[j];
          if (zi > z !== zj > z && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
        }
        if (inside) return entry.room.name;
      }
    }
    return null;
  };

  /** Where to stand to look at a room, and how much clear space is there.
   *
   * Measured against the grid rather than against the floor plan, and measured
   * on the Python side where the grid is: the outline is a plan and cannot say
   * where the wardrobe is, so the most open point *of the outline* put the
   * camera inside solid geometry, filling the view with the crimson of a sealed
   * interior. A reader who has just opened the page reads that as a broken
   * page, not as a camera inside the furniture.
   */
  const standIn = (name) => {
    const entry = state.find((other) => other.room.name === name) || state[0];
    return entry.room.stand || null;
  };

  // The grid's own extent, from the tiles rather than from the meshes: it is
  // wanted before any fine tile is fetched, and a page that ships no triangles
  // has nothing else to stand the camera in. Every quad of the flat lies inside
  // it, so it is the same box the mesh view would have reported.
  const bounds = new THREE.Box3();
  const corner = new THREE.Vector3();
  for (const entry of state) {
    for (const tier of [entry.room.fine, entry.room.coarse]) {
      for (const file of tier.tiles) {
        for (const point of file.bounds) bounds.expandByPoint(corner.fromArray(point));
      }
    }
  }

  return {
    group,
    select,
    refresh,
    say,
    roomAt,
    standIn,
    bounds,
    rooms: audit.rooms,
    note: audit.note,
  };
}


/** The tiered grid's controls and, more importantly, what is actually drawn.
 *
 * A reader looking at a room drawn at 8.17 mm while the page says 16 kHz would
 * take an aggregated feature for a missing one, which is the exact mistake this
 * view exists to prevent. So the step in force is on screen at all times, next
 * to the room it applies to, and the coarse step is named beside it.
 */
function auditSection(audit) {
  if (!audit) return "";
  const options = audit.rooms
    .map(
      (room) =>
        `<option value="${escapeHtml(room.name)}">${escapeHtml(room.name)} ` +
        `(${room.area_m2.toFixed(1)} m², ${room.fine.quads.toLocaleString()} quads` +
        `${room.fine.tiles.length > 1 ? `, ${room.fine.tiles.length} tiles` : ""})</option>`
    )
    .join("");
  const fine = audit.rooms[0].fine.cell_m * 1000;
  const coarse = audit.rooms[0].coarse.cell_m * 1000;
  const nodes = audit.total_nodes.toLocaleString();
  return `<h2>Voxelisation</h2>
    <p class="caption">${escapeHtml(audit.note)}</p>
    <label>room drawn at ${fine.toFixed(2)} mm
      <select id="audit-room">${options}</select></label>
    <label><input type="checkbox" id="audit-follow" checked>
      follow me: draw the room I am standing in</label>
    <p class="caption">${escapeHtml(audit.sealed_note || "")}
      ${(audit.sealed_left_out || 0).toLocaleString()} of them on this grid.</p>
    <p class="caption" id="audit-status">${nodes} boundary nodes at
      ${fine.toFixed(2)} mm. One room at that step, every other room at
      ${coarse.toFixed(2)} mm.</p>
    <p class="caption">The triangle mesh is not on this page. This flat's
      exported model is 192 MB of triangles and they would reach the browser as
      JSON to be hidden behind the grid; the grid is the picture here. The
      material table below still names every group the solver was sent.</p>`;
}

/** The air the solver sealed off, as a table a reader can challenge.
 *
 * Sealing stops the simulation carrying sound through a region, so it is shown
 * rather than assumed. The frequency is what makes the row actionable: a cavity
 * of side L would have rung at c/2L had it been left rigid, and that is how a
 * 125 Hz boom announces itself before anyone pays for a solve.
 */
function sealedSection(sealed) {
  if (!sealed) return "";
  const rows = (sealed.interiors || [])
    .slice(0, 8)
    .map(
      (r) =>
        `<tr><td>${escapeHtml(r.owner)}</td><td>${r.volume_m3.toFixed(3)} m³</td>` +
        `<td>${r.first_mode_hz.toFixed(0)} Hz</td></tr>`
    )
    .join("");
  const unclosed = sealed.unclosed_bodies || [];
  return `
    <h2>Sealed air</h2>
    <p class="caption">There is no air inside a solid object, so the solver is
    made to carry none there. Left coupled it is a cavity with rigid walls and no
    absorption at all, and it rings: that is what put 3.6 s of decay into the
    125 Hz band of an earlier run. Total ${sealed.sealed_volume_m3.toFixed(3)} m³
    in ${(sealed.interiors || []).length} closed bodies.</p>
    <table><tr><th>body</th><th>volume</th><th>would have rung at</th></tr>${rows}</table>
    ${
      unclosed.length
        ? `<p class="note">${unclosed.length} bodies are not closed, so their
           inside cannot be told from their outside and nothing was sealed for
           them: ${escapeHtml(unclosed.slice(0, 6).join(", "))}</p>`
        : `<p class="caption">Every body is closed, so no interior was left
           undecided.</p>`
    }`;
}

/** What a spatial run measured, which a point run has no equivalent of.
 *
 * Four things, and each is here because it can fail silently. The array says
 * how much of the room the encoder looked at. The centre identity is exact by
 * construction, so any number in it but zero is a defect. The direction of
 * arrival is the only check that would catch a mirrored frame. And the
 * effective order says where the expansion stops describing the field, which
 * is not the order that was asked for.
 */
function spatialSection(data) {
  if (!data.spatial) return "";
  const array = data.array || {};
  const identity = data.centre_identity;
  const doa = (data.direction_of_arrival || []).filter((row) => row.usable);
  const orders = data.conditioning || {};
  const air = data.air_absorption || {};
  const heads = data.heads || {};

  const identityRows = !identity
    ? ""
    : identity.per_band
        .map(
          (row) =>
            `<tr><td>${row.band_hz} Hz</td><td>${row.residual_db.toFixed(1)} dB</td>` +
            `<td class="${row.whole_band_fitted === false ? "out" : ""}">${(
              100 * row.share_of_energy
            ).toFixed(1)} %${row.whole_band_fitted === false ? " ¹" : ""}</td></tr>`
        )
        .join("");

  const orderRows = (orders.frequency_hz || [])
    .map(
      (hz, i) => `<tr><td>${Math.round(hz)} Hz</td><td>${orders.effective_order[i]}</td></tr>`
    )
    .join("");

  return `
    <h2>The array the field was expanded about</h2>
    <table>
      <tr><td>nodes</td><td>${array.nodes ?? "?"}</td></tr>
      <tr><td>outer radius</td><td>${array.outer_radius_m ?? "?"} m</td></tr>
      <tr><td>shells</td><td>${(array.shells || []).length}</td></tr>
      <tr><td>order out / fitted</td><td>${data.encoder?.order ?? "?"} / ${
        data.encoder?.fit_order ?? "?"
      }</td></tr>
    </table>
    <p class="caption">Drawn in the room as a wire ball with its shells. Every
    receiver is a grid node, so there is no interpolation error; the radii are
    the nodes' own, not the nominal ones.</p>

    ${
      identity
        ? `<h2>The check that costs nothing</h2>
           <table><tr><th>band</th><th>residual</th><th>energy</th></tr>${identityRows}</table>
           <p class="caption">The omnidirectional channel against the pressure
           measured at the array's centre node. At zero radius every radial term
           but the first vanishes, so these are the same signal by construction
           and any residual is the fit's own error. ¹ marks a band the encoder's
           limit cuts through.</p>`
        : ""
    }

    ${
      doa.length
        ? `<h2>Direction of arrival, against the geometry</h2>
           <table><tr><th>band</th><th>error</th></tr>${doa
             .map((row) => `<tr><td>${row.band_hz} Hz</td><td>${row.error_deg.toFixed(2)}°</td></tr>`)
             .join("")}</table>
           <p class="caption">The only measurement here that would catch a
           mirrored frame or an inverted odd order: both leave every level
           untouched.</p>`
        : ""
    }

    ${
      orderRows
        ? `<h2>Effective order</h2>
           <table><tr><th>frequency</th><th>order</th></tr>${orderRows}</table>
           <p class="caption">Measured on the array rather than assumed from the
           order that was asked for. Below 500 Hz the high orders are physically
           absent, which is the wavelength and not a defect.</p>`
        : ""
    }

    ${
      air.applied
        ? `<p class="caption">Air absorption applied, ${escapeHtml(
            air.standard || ""
          )}, ${air.temperature_c} °C and ${air.humidity_percent} % relative humidity.
           Both are part of the result: at 16 kHz the coefficient varies by 1.85
           across ordinary indoor humidity.</p>`
        : `<p class="note">No air absorption: the treble of the tail is overstated.</p>`
    }
    ${
      data.licence_conflict
        ? `<p class="note">The measured head is ${escapeHtml(
            data.licence_conflict.head_licence
          )} and this project's artefacts are ${escapeHtml(
            data.licence_conflict.project_licence
          )}. ${escapeHtml(data.licence_conflict.so)}</p>`
        : ""
    }
    <p class="caption">${Object.entries(heads)
      .map(([name, head]) => `${escapeHtml(name)}: ${escapeHtml(head.description || "")}`)
      .join("<br>")}</p>`;
}

/** The per band table of whichever measures the selected sample carries.
 *
 * Two shapes reach this panel. A pressure response carries room measures per
 * octave. A binaural response carries interaural ones, because RT60 of one ear
 * says nothing the omnidirectional row has not already said, while the tail's
 * coherence between the ears is the thing a decode can get wrong.
 */
function measuresTable(m) {
  if (!m) return "";
  if (m.bands_hz) {
    return (
      `<tr><th>band</th><th>RT60</th><th>EDT</th><th>C50</th><th>DRR</th></tr>` +
      m.bands_hz
        .map((band, i) => {
          const out = m.in_band && !m.in_band[i] ? ' class="out"' : "";
          const cell = (v, unit) => `<td${out}>${v === null ? "n/a" : v.toFixed(2) + unit}</td>`;
          return (
            `<tr><td${out}>${band} Hz${out ? " \u2717" : ""}</td>` +
            cell(m.rt60_s[i], " s") +
            cell(m.edt_s[i], " s") +
            cell(m.c50_db[i], " dB") +
            cell(m.drr_db[i], " dB") +
            "</tr>"
          );
        })
        .join("")
    );
  }
  if (m.late_coherence_per_band) {
    return (
      `<tr><th>band</th><th>zero lag</th><th>floor</th><th>peak</th></tr>` +
      m.late_coherence_per_band
        .map(
          (b) =>
            `<tr><td>${b.band_hz} Hz</td><td>${b.zero_lag.toFixed(2)}</td>` +
            `<td>${b.zero_lag_floor.toFixed(2)}</td>` +
            `<td>${b.max_lag.toFixed(2)}</td></tr>`
        )
        .join("")
    );
  }
  return "";
}

/** What the table under it means, for the shape of measures actually shown. */
function measuresCaption(m, roomNote) {
  if (!m) return "";
  if (m.bands_hz) return roomNote || "";
  if (m.late_coherence_per_band) {
    const parts = [];
    if (m.itd_us !== undefined) {
      parts.push(
        `direct sound: interaural time difference ${m.itd_us.toFixed(0)} \u00b5s ` +
          `against Woodworth's ${m.woodworth_itd_us.toFixed(0)} \u00b5s, ` +
          `level difference ${m.ild_db.toFixed(1)} dB.`
      );
    }
    parts.push(
      "Coherence of the late tail between the ears, per octave. Only the zero " +
        "lag column tests the diffuse field prediction; the floor beside it is " +
        "what this estimator reads on two independent signals."
    );
    return parts.join(" ");
  }
  return "";
}

export function renderRunPanel(element, data, { onSelect, onStand }) {
  const room = data.room;
  const theory = data.theory;
  const shell = data.theory_shell_only;

  const legend = data.groups
    .filter((g) => g.indices.length)
    .map(
      (g) =>
        `<span><i class="swatch" style="background: rgb(${g.colour.join(",")})"></i>` +
        // The number is on the label because four of the thirteen materials in
        // this room share an absorption, so the colour alone cannot separate
        // them and should not pretend to.
        `${escapeHtml(g.label)} ${g.absorption === null ? "α?" : "α " + g.absorption}</span>`
    )
    .join("");

  element.innerHTML = `
    <h1>${escapeHtml(data.run)}</h1>
    <p class="sub">scene ${escapeHtml(data.scene_id)}, room ${escapeHtml(data.room_name)}<br>
    mesh ${escapeHtml(data.scene_sha256.slice(0, 16))}…, grid ${escapeHtml(data.cache_key.slice(0, 12))}…</p>
    <p class="caption">The geometry drawn here is the serialised surface list the
    solver read, not the authored scene, so the picture and the responses are
    provably of the same object. That is why it looks blockier than the other
    two modes: this is the room the wave equation was solved in.</p>

    <h2>Room, measured on the solver's boundary</h2>
    <table>
      <tr><td>volume</td><td>${room.volume_m3.toFixed(2)} m³</td></tr>
      <tr><td>boundary the field meets</td><td>${room.surface_area_m2.toFixed(1)} m²</td></tr>
      <tr><td>shell alone</td><td>${room.shell_area_m2.toFixed(1)} m²</td></tr>
      <tr><td>mean absorption</td><td>${room.mean_absorption.toFixed(3)}</td></tr>
      <tr><td>first axial mode</td><td>${room.first_axial_mode_hz.toFixed(1)} Hz</td></tr>
      <tr><td>Sabine, full boundary</td><td>${theory.sabine_rt60_s.toFixed(2)} s</td></tr>
      ${shell ? `<tr><td>Sabine, shell only</td><td>${shell.sabine_rt60_s.toFixed(2)} s</td></tr>` : ""}
    </table>
    <p class="caption">${escapeHtml(data.theory_note || "")}</p>

    <h2>Surfaces</h2>
    <p class="caption">Coloured by absorption at 1 kHz, red reflective through to
    blue absorbent. Grey means the material carried no measured coefficient.</p>
    <div class="legend">${legend}</div>

    ${sealedSection(data.sealed)}
    ${auditSection(data.audit)}
    ${
      data.audit
        ? ""
        : data.voxels
        ? `<h2>Voxelisation</h2>
           <label><input type="checkbox" id="run-voxels" checked> the grid the solver read (untick for the triangles it was sent)</label>
           <p class="caption">${escapeHtml(data.voxels.note)}
           ${data.voxels.quads.toLocaleString()} quads stand for
           ${data.voxels.blocks.toLocaleString()} blocks and
           ${data.voxels.total_nodes.toLocaleString()} boundary nodes;
           ${data.voxels.sealed_quads.toLocaleString()} of them are sealed.</p>`
        : `<h2>Voxelisation</h2>
           <p class="caption">Not drawn. This run's grid is not on the machine
           serving this page and the shared store does not hold it either, so
           what you see above is the triangle mesh the solver was sent, not the
           grid it read. The two are not the same picture: the gap between them
           is where a mislaid material or an unsealed interior hides.</p>`
    }

    ${
      // A grid published on its own has no responses, and every panel below
      // reads one. Rendering them empty would be worse than leaving them out:
      // an empty decay curve looks like a decay that was measured and came out
      // flat. What is missing is said once, here, and the reason is carried in
      // `omissions` further down.
      data.samples.length === 0
        ? `<h2>Responses</h2>
           <p class="caption">None. Nothing was solved on this grid -- this page
           is the geometry and the grid the solver would read, published so the
           scene can be checked before any GPU time is spent on it.</p>`
        : `<h2>Sample</h2>
    <select id="run-pick">${data.samples
      .map((s, i) => `<option value="${i}">${escapeHtml(s.label)}</option>`)
      .join("")}</select>
    <button id="run-stand" style="margin-top:6px">${
      data.spatial
        ? "Stand where this was heard, facing the way it was decoded"
        : "Stand at this receiver"
    }</button>

    <h2>Impulse response</h2>
    <canvas class="plot" id="run-wave"></canvas>
    <p class="caption" id="run-wave-caption"></p>

    <h2>Spectrogram</h2>
    <canvas class="plot" id="run-spec" style="height: 132px"></canvas>
    <p class="caption" id="run-spec-caption"></p>

    <h2>Energy decay</h2>
    <canvas class="plot" id="run-edc"></canvas>
    <p class="caption">Schroeder backward integration, broadband, low cut at
    ${data.low_cut_hz ?? "?"} Hz.</p>

    <h2>Per band</h2>
    <table id="run-bands"></table>
    <p class="caption" id="run-bands-caption"></p>

    <h2>Listen</h2>
    <p class="caption">dry: ${escapeHtml(data.dry_voice?.member_name ?? "")},
    ${escapeHtml(data.dry_voice?.licence ?? "")}${data.dry_voice?.anechoic ? ", anechoic" : ""}</p>
    <audio id="run-dry" controls preload="none"></audio>
    <p class="caption" style="margin-top:.6rem">wet: the same voice through this response</p>
    <audio id="run-wet" controls preload="none"></audio>`
    }

    ${spatialSection(data)}

    ${
      data.binaural_note
        ? `<h2>What this is not</h2><p class="note">${escapeHtml(data.binaural_note)}</p>`
        : ""
    }
    <ul class="caption">${(data.omissions || [])
      .map((o) => `<li>${escapeHtml(o)}</li>`)
      .join("")}</ul>
    ${
      data.measured_anomaly
        // The heading follows the record rather than asserting either state:
        // it said "Open question" for as long as the anomaly was closed, which
        // is the way a page quietly goes stale.
        ? `<h2>${
            String(data.measured_anomaly.status || "").startsWith("closed")
              ? "A defect that was found and fixed"
              : "Open question"
          }</h2>
      <p class="note">${escapeHtml(data.measured_anomaly.what)}</p>
      <p class="caption">${escapeHtml(data.measured_anomaly.status)}</p>
      ${
        data.measured_anomaly.cause
          ? `<p class="caption"><b>Cause.</b> ${escapeHtml(data.measured_anomaly.cause)}</p>`
          : ""
      }
      ${
        data.measured_anomaly.fix
          ? `<p class="caption"><b>Fix.</b> ${escapeHtml(data.measured_anomaly.fix)}</p>`
          : ""
      }`
        : ""
    }
  `;

  const byId = (id) => element.querySelector(`#${id}`);
  const dry = byId("run-dry");
  if (dry) {
    if (data.dry_audio) dry.src = `${data.baseUrl}/${data.dry_audio}`;
    else dry.hidden = true;
  }
  const wet = byId("run-wet");
  const pick = byId("run-pick");

  function show(index) {
    const sample = data.samples[index];
    if (!sample) return;
    onSelect?.(sample);
    drawEnvelope(byId("run-wave"), sample.envelope);
    drawSpectrogram(byId("run-spec"), sample.spectrogram);
    drawDecay(byId("run-edc"), sample.decay);
    byId("run-spec-caption").textContent =
      `0 to ${(sample.spectrogram.max_hz / 1000).toFixed(1)} kHz, ` +
      `${sample.spectrogram.range_db} dB range, scaled to this response's own peak ` +
      `so two samples are not comparable by eye`;
    byId("run-wave-caption").textContent =
      `${sample.seconds.toFixed(2)} s at ${(sample.sample_rate_hz / 1000).toFixed(1)} kHz, ` +
      `peak ${sample.peak}`;
    // The audio is set before the plots. A binaural sample carries interaural
    // measures and no room measures, and reading the room shape off it threw
    // here, which left the player still pointed at the previously selected
    // response: the panel said one thing and played another.
    if (sample.wet_audio) {
      wet.src = `${data.baseUrl}/${sample.wet_audio}`;
      wet.hidden = false;
    } else {
      wet.removeAttribute("src");
      wet.hidden = true;
    }
    const m = sample.measures;
    byId("run-bands").innerHTML = measuresTable(m);
    byId("run-bands-caption").textContent = measuresCaption(m, data.band_note);
  }

  // A grid published without a solve renders none of the response panels, so
  // none of these elements exist. Wiring them unconditionally threw before the
  // voxels had a chance to draw, which turned a page missing its lower half
  // into a page missing everything.
  if (pick) {
    pick.addEventListener("change", () => show(Number(pick.value)));
    byId("run-stand")?.addEventListener("click", () => {
      const sample = data.samples[Number(pick.value)];
      if (sample) onStand?.(sample);
    });
    show(0);
  }
  return show;
}
