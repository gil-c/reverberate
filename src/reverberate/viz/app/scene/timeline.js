/** The timeline: transport controls, a ruler, and a lane per source and one for the listener.
 *
 * A source's lane has two bands. The upper says when it emits, in the colour
 * of its kind. The lower says what it does: a line high in the band while it
 * stands, low while it sits, a filled block while it travels, a slope while
 * it rises or sits down. The listener's lane is that lower band alone.
 *
 * It draws what the transport and the mix say and tells them what the user
 * did; it keeps no time of its own. The arithmetic is in `timemap.js`.
 */
import { SPEEDS } from "./transport.js";
import { createTimeMap, formatTime, laneLayout, panned, posture, resized, ticks, visible, zoomed } from "./timemap.js";
import { activeAt, spanAt } from "./tracks.js";

export const KIND_COLOURS = {
  near_voice: "#f2a541",
  far_voice: "#b48cf0",
  noise: "#7fb7a4",
  listener: "#4fd1e8",
};
const KIND_NAMES = { near_voice: "near voice", far_voice: "far voice", noise: "noise", listener: "listener" };
const LANE = 22;
const RULER = 20;

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

export function createTimeline(root, { transport, mix, onSelect, onFollow, onTrail }) {
  // --- the bar ------------------------------------------------------------
  const bar = make("div", "tl-bar");
  const play = make("button", "tl-play", "▶");
  play.type = "button";
  play.title = "play or pause (space)";
  const readout = make("span", "tl-time", "0:00.0 / 0:00");
  const speeds = make("div", "seg sub tl-speed");
  for (const value of SPEEDS) {
    const button = make("button", value === 1 ? "on" : "", `${value}×`);
    button.type = "button";
    button.dataset.speed = value;
    speeds.append(button);
  }
  const follow = make("label", "check");
  const followBox = make("input");
  followBox.type = "checkbox";
  follow.append(followBox, make("span", "", "through the listener's eyes"));
  const trail = make("label", "tl-trail");
  const trailInput = make("input");
  trailInput.type = "number";
  Object.assign(trailInput, { min: 0, max: 120, step: 1, value: 10 });
  trail.append(make("span", "", "trails"), trailInput, make("em", "", "s"));
  const legend = make("span", "tl-legend");
  for (const kind of ["near_voice", "far_voice", "noise", "listener"]) {
    const item = make("span", "", KIND_NAMES[kind]);
    item.style.setProperty("--kind", KIND_COLOURS[kind]);
    legend.append(item);
  }
  const note = make("span", "note tl-note");
  bar.append(play, readout, speeds, follow, trail, legend, note);

  // --- ruler and lanes ------------------------------------------------------
  const top = make("div", "tl-row tl-top");
  const ruler = make("canvas", "tl-ruler");
  const rule = make("div", "tl-rule");
  const mark = make("i", "tl-mark");
  rule.append(ruler, mark);
  top.append(make("div", "tl-corner"), rule);
  const scroll = make("div", "tl-row tl-scroll");
  const heads = make("div", "tl-heads");
  const lanes = make("canvas", "tl-lanes");
  scroll.append(heads, lanes);
  root.replaceChildren(bar, top, scroll);

  let tracks = null;
  let actors = [];
  let layout = laneLayout([]);
  let map = createTimeMap({ duration: 1, width: 1 });
  let selected = null;
  // The lanes without the playhead, kept so a frame is one copy and one line.
  const still = document.createElement("canvas");
  let stale = true;

  const ratio = () => Math.min(devicePixelRatio || 1, 2);
  function fit(canvas, width, height) {
    canvas.style.height = `${height}px`;
    const scale = ratio();
    if (canvas.width !== Math.round(width * scale) || canvas.height !== Math.round(height * scale)) {
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
    }
    const context = canvas.getContext("2d");
    context.setTransform(scale, 0, 0, scale, 0, 0);
    return context;
  }

  function drawLane(context, row, actor) {
    const colour = KIND_COLOURS[actor.kind];
    const dim = actor.id !== "listener" && !mix.audible(actor.id);
    if (actor.id === selected) {
      context.fillStyle = "rgba(255, 255, 255, 0.07)";
      context.fillRect(0, row.y, map.width, row.height);
    }
    context.globalAlpha = dim ? 0.28 : 1;
    if (row.activity) {
      context.fillStyle = colour;
      for (const [start, end] of visible(actor.activity, map.start, map.end, (a) => a[0], (a) => a[1])) {
        const x = map.toX(start);
        context.fillRect(x, row.activity.y, Math.max(1, map.toX(end) - x), row.activity.height);
      }
    }
    const band = row.movement;
    const high = band.y + 1.5;
    const low = band.y + band.height - 1.5;
    const level = (value) => low + (high - low) * value;
    for (const span of visible(actor.movement, map.start, map.end)) {
      const x0 = map.toX(span.start_s);
      const x1 = Math.max(x0 + 1, map.toX(span.end_s));
      const shape = posture(span);
      if (shape.moving && shape.from === shape.to) {
        context.fillStyle = colour;
        context.globalAlpha = (dim ? 0.28 : 1) * 0.55;
        context.fillRect(x0, band.y, x1 - x0, band.height);
        context.globalAlpha = dim ? 0.28 : 1;
        continue;
      }
      context.strokeStyle = shape.moving ? colour : "#a3afbd";
      context.lineWidth = shape.moving ? 2 : 1.5;
      context.beginPath();
      context.moveTo(x0, level(shape.from));
      context.lineTo(x1, level(shape.to));
      context.stroke();
    }
    context.globalAlpha = 1;
    context.fillStyle = "#262e38";
    context.fillRect(0, row.y + row.height - 1, map.width, 1);
  }

  function drawStill() {
    const width = lanes.clientWidth;
    if (!width) return;
    map = resized(map, width);
    const context = fit(still, width, Math.max(layout.height, 1));
    context.clearRect(0, 0, width, layout.height);
    for (const row of layout.rows) drawLane(context, row, actors[row.index]);
    const scale = fit(ruler, width, RULER);
    scale.clearRect(0, 0, width, RULER);
    scale.font = "11px ui-monospace, Menlo, monospace";
    scale.fillStyle = "#a3afbd";
    scale.strokeStyle = "#34404d";
    for (const tick of ticks(map).marks) {
      scale.beginPath();
      scale.moveTo(tick.x + 0.5, RULER - 6);
      scale.lineTo(tick.x + 0.5, RULER);
      scale.stroke();
      scale.fillText(tick.label, tick.x + 3, RULER - 8);
    }
    stale = false;
  }

  /** One frame: the lanes as they stand, and the playhead. */
  function draw() {
    if (root.hidden) return;
    const { t, playing, duration } = transport.state();
    // Playing off the window's edge turns the page.
    if (playing && (t > map.end || t < map.start) && map.span < map.duration) {
      map = createTimeMap({ duration: map.duration, width: map.width, span: map.span, start: t });
      stale = true;
    }
    if (stale || still.width !== Math.round(lanes.clientWidth * ratio())) drawStill();
    const width = lanes.clientWidth;
    const context = fit(lanes, width, Math.max(layout.height, 1));
    context.clearRect(0, 0, width, layout.height);
    context.drawImage(still, 0, 0, width, Math.max(layout.height, 1));
    const x = map.toX(t);
    context.fillStyle = "#e6ebf1";
    context.fillRect(Math.round(x), 0, 1, layout.height);
    // The ruler's own playhead is a mark over it, moved and not redrawn.
    mark.style.transform = `translateX(${Math.round(x)}px)`;
    mark.hidden = x < 0 || x > width;
    readout.textContent = `${formatTime(t, true)} / ${formatTime(duration)}`;
    play.textContent = playing ? "❚❚" : "▶";
  }

  function renderHeads() {
    heads.replaceChildren(
      ...actors.map((actor) => {
        const head = make("div", `tl-head${actor.id === selected ? " sel" : ""}`);
        head.style.setProperty("--kind", KIND_COLOURS[actor.kind]);
        head.style.height = `${LANE}px`;
        const name = make("span", "nm", actor.id);
        name.title = actor.subtype ? `${KIND_NAMES[actor.kind]} · ${actor.subtype}` : KIND_NAMES[actor.kind];
        head.append(make("i"), name);
        if (actor.id !== "listener") {
          head.classList.toggle("silent", !mix.audible(actor.id));
          for (const [letter, on, flip, title] of [
            ["S", mix.isSolo(actor.id), mix.toggleSolo, "solo"],
            ["M", mix.isMuted(actor.id), mix.toggleMute, "mute"],
          ]) {
            const button = make("button", on ? `on ${title}` : title, letter);
            button.type = "button";
            button.title = title;
            button.addEventListener("click", (event) => {
              event.stopPropagation();
              flip(actor.id);
            });
            head.append(button);
          }
        }
        head.addEventListener("click", () => onSelect(actor.id));
        return head;
      })
    );
  }

  // --- input ----------------------------------------------------------------
  play.addEventListener("click", () => transport.toggle());
  speeds.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (button) transport.setSpeed(Number(button.dataset.speed));
  });
  followBox.addEventListener("change", () => onFollow(followBox.checked));
  trailInput.addEventListener("change", () => {
    const seconds = Math.min(120, Math.max(0, Number(trailInput.value) || 0));
    trailInput.value = seconds;
    onTrail(seconds);
  });

  const xOf = (canvas, event) => event.clientX - canvas.getBoundingClientRect().left;
  for (const canvas of [ruler, lanes]) {
    let scrubbing = false;
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !tracks) return;
      scrubbing = true;
      canvas.setPointerCapture(event.pointerId);
      transport.seek(map.toT(xOf(canvas, event)));
    });
    canvas.addEventListener("pointermove", (event) => {
      if (scrubbing) transport.seek(map.toT(xOf(canvas, event)));
      else if (canvas === lanes) describe(event);
    });
    const stop = () => {
      scrubbing = false;
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener(
      "wheel",
      (event) => {
        if (!tracks) return;
        // Over the lanes a plain wheel scrolls them; over the ruler it zooms.
        const zooming = canvas === ruler || event.ctrlKey || event.metaKey;
        const sideways = Math.abs(event.deltaX) > Math.abs(event.deltaY);
        if (!zooming && !sideways) return;
        event.preventDefault();
        map = sideways ? panned(map, event.deltaX) : zoomed(map, Math.exp(event.deltaY * 0.002), xOf(canvas, event));
        stale = true;
        draw();
      },
      { passive: false }
    );
    canvas.addEventListener("dblclick", () => {
      if (canvas !== ruler || !tracks) return;
      map = createTimeMap({ duration: map.duration, width: map.width });
      stale = true;
      draw();
    });
  }
  lanes.addEventListener("pointerleave", () => {
    note.textContent = "";
  });

  /** What the lane under the pointer is doing at the time under it. */
  function describe(event) {
    const rect = lanes.getBoundingClientRect();
    const row = layout.at(event.clientY - rect.top);
    if (!row) return;
    const actor = actors[row.index];
    const t = map.toT(event.clientX - rect.left);
    const span = spanAt(actor.movement, t);
    const parts = [actor.id, formatTime(t, true)];
    if (span) parts.push([span.type, span.station || span.rail || "", span.height || (span.to ? `to ${span.to}` : "")].filter(Boolean).join(" "));
    if (actor.activity.length) parts.push(activeAt(actor.activity, t) ? "emitting" : "silent");
    note.textContent = parts.join(" · ");
  }

  new ResizeObserver(() => {
    stale = true;
    draw();
  }).observe(scroll);
  transport.on("speed", ({ speed }) => {
    for (const button of speeds.querySelectorAll("button")) {
      button.classList.toggle("on", Number(button.dataset.speed) === speed);
    }
  });
  mix.on("change", () => {
    stale = true;
    renderHeads();
    draw();
  });

  return {
    draw,
    /** A scene's tracks, or null: the lanes are its sources in order, then the listener. */
    setTracks(next) {
      tracks = next;
      actors = next
        ? [
            ...next.sources,
            { id: "listener", kind: "listener", activity: [], movement: next.listener.movement },
          ]
        : [];
      layout = laneLayout(actors.map((actor) => actor.id), { laneHeight: LANE });
      map = createTimeMap({ duration: next ? next.duration_s : 1, width: lanes.clientWidth || 1 });
      stale = true;
      renderHeads();
      draw();
    },
    setSelected(id) {
      selected = id;
      stale = true;
      renderHeads();
      draw();
    },
    setFollow(on) {
      followBox.checked = on;
    },
    trailSeconds: () => Number(trailInput.value) || 0,
    /** The window on show, for a check that cannot look. */
    window: () => ({ start: map.start, end: map.end, width: map.width, lanes: layout.rows.length }),
  };
}
