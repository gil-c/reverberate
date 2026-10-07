/** A player's controls: play, loop, where it is, how loud, and the listener's head.
 *
 *   createTransport(element, player)
 *
 * builds, in `element`: a play button, a loop button, the time, a bar that
 * is clicked to move and dragged to set the region the loop turns in (a
 * double click clears it), the level, and a dial for the head. The head is
 * turned by dragging the dial (left and right turn it, up and down tilt it)
 * and by the arrow keys; `0` puts it back on the scene's own.
 *
 * Keys, unless a field has the focus: space plays and stops, `L` loops, the
 * arrows turn the head by five degrees, `0` centres it.
 */
const el = (tag, attributes = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
};

const clock = (seconds) => {
  const s = Math.max(seconds, 0);
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
};

export function createTransport(element, player, { keys = true } = {}) {
  const play = el("button", { type: "button", class: "p-button p-play", title: "Play or stop (space)" });
  const loop = el("button", { type: "button", class: "p-button", title: "Loop the region, or the whole (L)", text: "Loop" });
  const time = el("span", { class: "p-time" });
  const region = el("div", { class: "p-region" });
  const cursor = el("div", { class: "p-cursor" });
  const bar = el("div", { class: "p-bar", title: "Click to move; drag to set the loop's region; double click to clear it" }, region, cursor);
  const level = el("input", { type: "range", min: "-30", max: "20", step: "1", value: "0", class: "p-level", title: "Level, dB over the default" });
  const levelSaid = el("span", { class: "p-said" });
  const clip = el("span", { class: "p-clip" });
  const dial = el("div", { class: "p-dial", title: "Drag to turn your head; arrow keys; 0 centres it", tabindex: "0" });
  dial.innerHTML =
    '<svg viewBox="-50 -50 100 100" aria-hidden="true">' +
    '<circle r="46" class="p-dial-rim"/><line y1="-46" y2="-38" class="p-dial-front"/>' +
    '<g class="p-dial-head"><ellipse rx="20" ry="24" class="p-dial-skull"/>' +
    '<path d="M -6 -22 L 0 -34 L 6 -22 Z" class="p-dial-nose"/>' +
    '<rect x="-25" y="-6" width="5" height="12" rx="2" class="p-dial-ear"/>' +
    '<rect x="20" y="-6" width="5" height="12" rx="2" class="p-dial-ear"/></g></svg>';
  const headSaid = el("span", { class: "p-said" });
  const centre = el("button", { type: "button", class: "p-button p-small", title: "Back on the scene's own head (0)", text: "Centre" });
  const note = el("span", { class: "p-note" });
  element.classList.add("p-transport");
  element.append(
    el("div", { class: "p-row" }, play, loop, time, bar),
    el(
      "div",
      { class: "p-row" },
      el("label", { class: "p-field" }, el("span", { text: "Level" }), level, levelSaid, clip),
      el("div", { class: "p-field p-head" }, el("span", { text: "Head" }), dial, headSaid, centre),
      note
    )
  );

  let clipTimer = 0;
  function show() {
    const state = player.state();
    play.textContent = state.playing ? "Stop" : "Play";
    play.classList.toggle("on", state.playing);
    loop.classList.toggle("on", state.looping);
    levelSaid.textContent = `${state.levelDb >= 0 ? "+" : ""}${state.levelDb} dB`;
    const { yaw, pitch } = state.turned;
    const fixed = state.kind === "binaural";
    dial.classList.toggle("off", fixed);
    dial.querySelector(".p-dial-head").setAttribute("transform", `rotate(${-yaw})`);
    headSaid.textContent = fixed
      ? "fixed"
      : `${Math.abs(Math.round(yaw))}° ${yaw > 0.5 ? "left" : yaw < -0.5 ? "right" : ""}${Math.abs(pitch) > 0.5 ? `, ${Math.abs(Math.round(pitch))}° ${pitch > 0 ? "up" : "down"}` : ""}`;
    note.textContent = state.failure
      ? state.failure
      : fixed
        ? "These files are two ears, decoded for the scene's own head: the head does not turn."
        : "";
    note.classList.toggle("bad", Boolean(state.failure));
    if (state.loop && state.duration) {
      region.hidden = false;
      region.style.left = `${(100 * state.loop[0]) / state.duration}%`;
      region.style.width = `${(100 * (state.loop[1] - state.loop[0])) / state.duration}%`;
    } else region.hidden = true;
    showTime(player.time());
  }
  function showTime(seconds) {
    const { duration } = player.state();
    time.textContent = `${clock(seconds)} / ${clock(duration)}`;
    cursor.style.left = duration ? `${(100 * seconds) / duration}%` : "0";
  }
  player.on("state", show).on("head", show).on("time", showTime);
  player.on("clip", ({ overDb }) => {
    clip.textContent = `clip +${overDb.toFixed(1)} dB`;
    clearTimeout(clipTimer);
    clipTimer = setTimeout(() => (clip.textContent = ""), 3000);
  });

  play.addEventListener("click", () => player.toggle());
  loop.addEventListener("click", () => player.setLooping(!player.state().looping));
  level.addEventListener("input", () => player.setLevel(Number(level.value)));
  centre.addEventListener("click", () => player.turn({ yaw: 0, pitch: 0 }));

  // The bar: a click moves, a drag is the loop's region.
  const secondsAt = (event) => {
    const box = bar.getBoundingClientRect();
    return Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)) * player.state().duration;
  };
  attachRegion(bar, secondsAt, player);

  // The head: a drag of the dial, 0.5 degrees a pixel.
  let held = null;
  dial.addEventListener("pointerdown", (event) => {
    held = { x: event.clientX, y: event.clientY, ...player.state().turned };
    dial.setPointerCapture(event.pointerId);
  });
  dial.addEventListener("pointermove", (event) => {
    if (!held) return;
    // Dragging to the right turns the head to the right: a yaw to the left is positive.
    player.turn({ yaw: held.yaw - (event.clientX - held.x) * 0.5, pitch: held.pitch - (event.clientY - held.y) * 0.5 });
  });
  dial.addEventListener("pointerup", () => (held = null));
  dial.addEventListener("dblclick", () => player.turn({ yaw: 0, pitch: 0 }));

  if (keys) {
    document.addEventListener("keydown", (event) => {
      const target = event.target;
      if (target instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName) && target.type !== "range") return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const { yaw, pitch } = player.state().turned;
      const step = event.shiftKey ? 15 : 5;
      if (event.code === "Space") player.toggle();
      else if (event.key === "l" || event.key === "L") player.setLooping(!player.state().looping);
      else if (event.key === "ArrowLeft") player.turn({ yaw: yaw + step });
      else if (event.key === "ArrowRight") player.turn({ yaw: yaw - step });
      else if (event.key === "ArrowUp") player.turn({ pitch: pitch + step });
      else if (event.key === "ArrowDown") player.turn({ pitch: pitch - step });
      else if (event.key === "0") player.turn({ yaw: 0, pitch: 0 });
      else return;
      event.preventDefault();
    });
  }
  show();
  return { show };
}

/** On `element`: a click moves the player, a drag sets its loop's region, a double click clears it. */
export function attachRegion(element, secondsAt, player) {
  let from = null;
  let dragged = false;
  element.addEventListener("pointerdown", (event) => {
    from = secondsAt(event);
    dragged = false;
    element.setPointerCapture(event.pointerId);
  });
  element.addEventListener("pointermove", (event) => {
    if (from === null) return;
    const to = secondsAt(event);
    if (Math.abs(to - from) > 0.1) {
      dragged = true;
      player.setLoop([Math.min(from, to), Math.max(from, to)]);
    }
  });
  element.addEventListener("pointerup", (event) => {
    if (from === null) return;
    if (dragged) {
      player.setLooping(true);
      if (player.state().playing) player.seek(player.state().loop[0]);
    } else player.seek(secondsAt(event));
    from = null;
  });
  element.addEventListener("dblclick", () => {
    player.setLoop(null);
  });
}
