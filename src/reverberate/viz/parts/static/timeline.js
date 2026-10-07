/** A scene's timeline: a lane a source, its intervals coloured by what they are to the listener.
 *
 *   createTimeline(element, scene, player);   // the answer of api/scene, a player or a clock
 *
 * The first lane is the listener's: who he talks with, span by span. Under
 * it each source has a lane, in the order of the track list, and each of its
 * intervals is a block: green while it is of the listener's conversation (his
 * own voice too), grey for anybody else, sand for what is not a person. A
 * word thrown in and a laugh are drawn half as high as a turn.
 *
 * A click moves the player, a drag sets the region its loop turns in, a
 * double click clears it: `attachRegion`, as on a transport's bar.
 */
import { WITH_YOU, spanAt } from "./scene-view.js";
import { attachRegion } from "./transport.js";

export const ROLE_COLOURS = { conversation: WITH_YOU, outside: "#8b93a1", noise: "#cdb88c" };
const ROLE_WORDS = { conversation: "in your conversation", outside: "other people", noise: "not a person" };

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

const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`;

/** What the listener is doing at `seconds`, in words: who he is with and what he looks at. */
export function whoAt(scene, seconds) {
  const names = Object.fromEntries(scene.sources.map((source) => [source.id, source.label]));
  const talking = spanAt(scene.listener.conversation, seconds);
  const gaze = spanAt(scene.listener.gaze, seconds);
  const group = talking && talking.members.length ? `You are with ${talking.members.map((id) => names[id]).join(", ")}` : "You are with nobody";
  return gaze && gaze.said ? `${group}, ${gaze.said}.` : `${group}.`;
}

export function createTimeline(element, scene, player) {
  element.classList.add("p-timeline");
  const duration = scene.duration_s;
  const pct = (seconds) => `${(100 * seconds) / duration}%`;
  const block = (span, colour, extra = "") => {
    const node = el("i", { class: `p-tl-block ${extra}` });
    node.style.left = pct(span.start_s);
    node.style.width = pct(span.end_s - span.start_s);
    node.style.background = colour;
    return node;
  };
  const row = (name, colour, shape, blocks) => {
    const chip = el("i", { class: `p-chip ${shape}` });
    chip.style.background = colour;
    return el("div", { class: "p-tl-row" }, el("span", { class: "p-tl-name" }, chip, el("span", { text: name })), el("div", { class: "p-tl-lane" }, ...blocks));
  };

  const names = Object.fromEntries(scene.sources.map((source) => [source.id, source.label]));
  const rows = el("div", { class: "p-tl-rows" });
  rows.append(
    row(
      "You talk with",
      "#ffffff",
      "you",
      scene.listener.conversation
        .filter((span) => span.members.length)
        .map((span) => {
          const node = block(span, WITH_YOU, "p-tl-with");
          node.textContent = span.members.map((id) => names[id]).join(", ");
          return node;
        })
    )
  );
  for (const source of scene.sources) {
    const blocks = source.intervals.map((span) => block(span, ROLE_COLOURS[span.role] || ROLE_COLOURS.outside, span.event && span.event !== "turn" ? "p-tl-short" : ""));
    rows.append(row(source.label, source.colour, source.shape === "marker" ? "noise" : source.shape === "self" ? "you" : "voice", blocks));
  }

  // The ruler: a mark every half minute, or every ten seconds of a short scene.
  const every = duration > 90 ? 30 : duration > 30 ? 10 : 5;
  const ruler = el("div", { class: "p-tl-lane p-tl-ruler" });
  for (let t = 0; t <= duration + 1e-6; t += every) {
    const tick = el("span", { text: clock(t) });
    tick.style.left = pct(t);
    ruler.append(tick);
  }
  rows.append(el("div", { class: "p-tl-row" }, el("span", { class: "p-tl-name" }), ruler));

  const region = el("div", { class: "p-region" });
  const cursor = el("div", { class: "p-cursor" });
  const over = el("div", { class: "p-tl-over", title: "Click to move; drag to set the region the loop turns in; double click to clear it" }, region, cursor);
  const legend = el("div", { class: "p-tl-legend" });
  for (const role of ["conversation", "outside", "noise"]) {
    if (!scene.sources.some((source) => source.intervals.some((span) => span.role === role))) continue;
    const chip = el("i", { class: "p-tl-key" });
    chip.style.background = ROLE_COLOURS[role];
    legend.append(el("span", {}, chip, ROLE_WORDS[role]));
  }
  element.append(el("div", { class: "p-tl-box" }, rows, over), legend);

  const secondsAt = (event) => {
    const box = over.getBoundingClientRect();
    return Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)) * duration;
  };
  attachRegion(over, secondsAt, player);

  function show() {
    const state = player.state();
    if (state.loop) {
      region.hidden = false;
      region.style.left = pct(state.loop[0]);
      region.style.width = pct(state.loop[1] - state.loop[0]);
    } else region.hidden = true;
    cursor.style.left = pct(Math.min(player.time(), duration));
  }
  player.on("state", show).on("time", show);
  show();
  return { show };
}
