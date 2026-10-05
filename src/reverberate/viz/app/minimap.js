/** The plan: rooms with their names, the listener, the sources.
 *
 * Drawn as SVG from the manifest's room outlines; x runs right and z runs
 * down, which is the scene frame seen from above with y towards the viewer.
 * The listener is dragged here; height is set in the listener tab.
 *
 * A scene adds two layers: what it may use (the free floor, rails, stations)
 * under the run's sources, and its actors over them, each a mark with a line
 * along its facing and the trail of where it has just been.
 */
const NS = "http://www.w3.org/2000/svg";
const SIZE = 240;
const PAD = 8;
const MIN_NAMED_M2 = 6;

const el = (tag, attributes = {}) => {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
};

// A scene's marks, in the plan's own units: the dot and the line along the facing.
const ACTOR_R = 3.2;
const FACING = 9;

export function createMinimap(svg, { onMove, onSelectSource, onSelectActor = () => {} }) {
  let scale = 1;
  let origin = [0, 0];
  let rooms = [];
  const layerRooms = el("g");
  const layerUnder = el("g", { class: "underlay" });
  const layerScene = el("g", { class: "scene-plan" });
  const layerPoints = el("g");
  const layerSources = el("g");
  const layerActors = el("g", { class: "actors" });
  const wedge = el("path", { class: "wedge" });
  const ear = el("circle", { class: "ear", r: 4.5 });
  svg.append(layerRooms, layerUnder, layerScene, layerPoints, layerSources, layerActors, wedge, ear);
  let scene = null;
  let underlay = null;
  const actors = new Map();
  const names = new Map();
  const shapes = new Map();

  const toSvg = (x, z) => [(x - origin[0]) * scale + PAD, (z - origin[1]) * scale + PAD];
  let dots = [];
  const fromSvg = (px, py) => [(px - PAD) / scale + origin[0], (py - PAD) / scale + origin[1]];
  const pointOf = (event) => {
    const rect = svg.getBoundingClientRect();
    return fromSvg(
      ((event.clientX - rect.left) / rect.width) * SIZE,
      ((event.clientY - rect.top) / rect.height) * SIZE
    );
  };
  const ringPoints = (ring) => ring.map(([x, z]) => toSvg(x, z).join(",")).join(" ");

  function fit(outline, roomList) {
    const xs = [];
    const zs = [];
    for (const part of outline || []) for (const [x, z] of part.exterior) xs.push(x), zs.push(z);
    for (const room of roomList) for (const ring of room.outline) for (const [x, z] of ring) xs.push(x), zs.push(z);
    if (!xs.length) return;
    const minX = Math.min(...xs);
    const minZ = Math.min(...zs);
    const span = Math.max(Math.max(...xs) - minX, Math.max(...zs) - minZ, 1e-6);
    scale = (SIZE - 2 * PAD) / span;
    // Centre the shorter axis.
    const offX = ((SIZE - 2 * PAD) / scale - (Math.max(...xs) - minX)) / 2;
    const offZ = ((SIZE - 2 * PAD) / scale - (Math.max(...zs) - minZ)) / 2;
    origin = [minX - offX, minZ - offZ];
  }

  let dragging = false;
  ear.addEventListener("pointerdown", (event) => {
    dragging = true;
    ear.setPointerCapture(event.pointerId);
  });
  ear.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const [x, z] = pointOf(event);
    onMove(x, z);
  });
  ear.addEventListener("pointerup", () => {
    dragging = false;
  });
  svg.addEventListener("dblclick", (event) => {
    const [x, z] = pointOf(event);
    onMove(x, z);
  });

  function drawPoints() {
    layerPoints.replaceChildren(
      ...dots.map(([x, , z]) => {
        const [px, py] = toSvg(x, z);
        return el("circle", { class: "measure", cx: px, cy: py, r: 1.1 });
      })
    );
  }

  function drawUnderlay() {
    layerUnder.replaceChildren();
    if (!underlay) return;
    const [x, y] = toSvg(underlay.x0, underlay.z0);
    const [x1, y1] = toSvg(underlay.x1, underlay.z1);
    const image = el("image", { x, y, width: x1 - x, height: y1 - y, preserveAspectRatio: "none", opacity: 0.9 });
    image.setAttribute("href", underlay.url);
    image.style.imageRendering = "pixelated";
    image.style.pointerEvents = "none";
    layerUnder.append(image);
  }

  function drawScene() {
    layerScene.replaceChildren();
    if (!scene) return;
    for (const part of scene.free || []) {
      const rings = [part.exterior, ...(part.holes || [])];
      const d = rings.map((ring) => `M${ring.map(([x, z]) => toSvg(x, z).join(",")).join(" L")} Z`).join(" ");
      layerScene.append(el("path", { class: "free", d, "fill-rule": "evenodd" }));
    }
    const used = scene.used || null;
    for (const rail of scene.rails || []) {
      const on = used && used.rails.has(rail.id);
      layerScene.append(el("polyline", { class: `rail${on ? " used" : ""}`, points: ringPoints(rail.points) }));
    }
    for (const station of scene.stations || []) {
      const [px, py] = toSvg(station.position[0], station.position[2]);
      const on = used && used.stations.has(station.id);
      const mark = el("circle", { class: `station ${station.kind}${on ? " used" : ""}`, cx: px, cy: py, r: 1.5 });
      const title = el("title");
      title.textContent = `${station.id} · ${station.kind} · ${station.room}`;
      mark.append(title);
      layerScene.append(mark);
    }
  }

  return {
    /** What a scene may use: `{ free, rails, stations, used }`, or null for none.
     *  `used` holds the ids a recipe takes, drawn brighter than the rest. */
    setScene(next) {
      scene = next;
      drawScene();
    },
    /** A picture laid on the plan between two corners of the scene, `{ url, x0, z0, x1, z1 }`,
     *  its first row at `z0`; null takes it away. A slice of the wave solver's grid. */
    setUnderlay(next) {
      underlay = next;
      drawUnderlay();
    },
    showScene(on) {
      layerScene.style.display = on ? "" : "none";
    },
    /** A scene's actors, `[{ id, kind }]`; every mark is made once and then moved. */
    setActors(list) {
      layerActors.replaceChildren();
      actors.clear();
      for (const actor of list) {
        const holder = el("g", { class: `actor ${actor.kind}` });
        const trail = el("polyline", { class: "trail" });
        const facing = el("line", { class: "facing" });
        const dot = el("circle", { class: "dot", r: actor.kind === "listener" ? ACTOR_R + 1 : ACTOR_R });
        const title = el("title");
        title.textContent = actor.id;
        dot.append(title);
        dot.addEventListener("click", () => onSelectActor(actor.id));
        holder.append(trail, facing, dot);
        layerActors.append(holder);
        actors.set(actor.id, { holder, trail, facing, dot });
      }
    },
    /** One actor at an instant: where, facing `[dx, dz]`, and how it is drawn.
     *  `trail` is `[x, y, z, ...]` and `points` how many of them count. */
    moveActor(id, { x, z, facing, active, dim, selected, trail, points }) {
      const actor = actors.get(id);
      if (!actor) return;
      const [px, py] = toSvg(x, z);
      actor.dot.setAttribute("cx", px);
      actor.dot.setAttribute("cy", py);
      actor.facing.setAttribute("x1", px);
      actor.facing.setAttribute("y1", py);
      actor.facing.setAttribute("x2", px + facing[0] * FACING);
      actor.facing.setAttribute("y2", py + facing[1] * FACING);
      actor.holder.classList.toggle("active", Boolean(active));
      actor.holder.classList.toggle("dim", Boolean(dim));
      actor.holder.classList.toggle("sel", Boolean(selected));
      let path = "";
      for (let k = 0; k < (points || 0); k++) {
        const [tx, ty] = toSvg(trail[3 * k], trail[3 * k + 2]);
        path += `${tx.toFixed(1)},${ty.toFixed(1)} `;
      }
      actor.trail.setAttribute("points", path);
    },
    /** The measurement points, redrawn whenever the plan is rescaled. */
    setPoints(positions) {
      dots = positions;
      drawPoints();
    },
    showPoints(on) {
      layerPoints.style.display = on ? "" : "none";
    },
    /** New apartment: its rooms and walkable outline. */
    setPlan({ rooms: roomList, outline }) {
      rooms = roomList || [];
      layerRooms.replaceChildren();
      names.clear();
      shapes.clear();
      fit(outline, rooms);
      for (const part of outline || []) {
        layerRooms.append(el("polygon", { class: "walk", points: ringPoints(part.exterior) }));
      }
      for (const room of rooms) {
        for (const ring of room.outline) {
          const shape = el("polygon", { class: "room", points: ringPoints(ring) });
          layerRooms.append(shape);
          if (!shapes.has(room.name)) shapes.set(room.name, []);
          shapes.get(room.name).push(shape);
        }
        // A name needs room to be read: below this the plan is a thicket of
        // overlapping labels and the highlight alone says where you are.
        if ((room.area_m2 || 0) < MIN_NAMED_M2) continue;
        const [lx, lz] = room.label_at || room.outline[0][0];
        const [px, py] = toSvg(lx, lz);
        const text = el("text", { class: "name", x: px, y: py + 3, "text-anchor": "middle" });
        text.textContent = room.name;
        layerRooms.append(text);
        names.set(room.name, text);
      }
      drawPoints();
      drawScene();
      drawUnderlay();
    },

    setSources(sources, selected) {
      layerSources.replaceChildren();
      for (const source of sources) {
        const [px, py] = toSvg(source.position[0], source.position[2]);
        const ring = el("circle", {
          class: `src-mark${source.on ? "" : " off"}${source.id === selected ? " sel" : ""}`,
          cx: px,
          cy: py,
          r: 6,
        });
        ring.addEventListener("click", () => onSelectSource(source.id));
        const dot = el("circle", { class: "src-dot", cx: px, cy: py, r: 1.6 });
        const label = el("text", { class: "src-name", x: px + 9, y: py + 3 });
        label.textContent = source.id;
        layerSources.append(ring, dot, label);
      }
    },

    /** The listener moved: dot, wedge, room highlight. */
    update(listener, room) {
      const [px, py] = toSvg(listener.x, listener.z);
      ear.setAttribute("cx", px);
      ear.setAttribute("cy", py);
      // Heading on the plan: forward is (-sin yaw, -cos yaw) in (x, z).
      const heading = Math.atan2(-Math.cos(listener.yaw), -Math.sin(listener.yaw));
      const spread = (listener.fov / 2) * (Math.PI / 180);
      const R = 26;
      const a1 = heading - spread;
      const a2 = heading + spread;
      wedge.setAttribute(
        "d",
        `M${px},${py} L${px + R * Math.cos(a1)},${py + R * Math.sin(a1)} ` +
          `A${R},${R} 0 0 1 ${px + R * Math.cos(a2)},${py + R * Math.sin(a2)} Z`
      );
      for (const [name, list] of shapes) {
        for (const shape of list) shape.classList.toggle("here", name === room);
      }
      for (const [name, text] of names) text.classList.toggle("here", name === room);
    },
  };
}
