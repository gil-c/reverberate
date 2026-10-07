/** Recipe viewer: the scene player's view and timeline on a clock, with nothing to hear.
 *
 * Where everything is, where the listener looks and who is of his
 * conversation are the recipe's own, sampled by the server through
 * `reverberate.scenes.kinematics`.
 */
import { createClock } from "./parts/clock.js";
import { createSceneView } from "./parts/scene-view.js";
import { createTimeline, whoAt } from "./parts/timeline.js";
import { createTransport } from "./parts/transport.js";

const $ = (id) => document.getElementById(id);
const name = new URLSearchParams(location.search).get("recipe") || "";
const response = await fetch(`api/view/${name.split("/").map(encodeURIComponent).join("/")}`);
const scene = await response.json();
if (!response.ok) {
  $("what").textContent = `${name}: ${scene.error}`;
  $("what").className = "p-bad";
} else {
  document.title = `${name}: recipe viewer`;
  const people = scene.sources.filter((source) => source.shape === "head").length;
  const things = scene.sources.filter((source) => source.shape === "marker").length;
  $("what").textContent = `${name}: ${Math.floor(scene.duration_s / 60)} min ${Math.round(scene.duration_s % 60)} s in ${scene.dwelling}, ${people} people and ${things} noise(s).`;
  if (scene.note) $("scene-note").textContent += ` ${scene.note}`;
  const clock = createClock(scene.duration_s);
  createTransport($("transport"), clock, { bar: false, level: false, head: false });
  const view = createSceneView($("scene"), scene, { lit: "intervals" });
  createTimeline($("timeline"), scene, clock);
  const show = (seconds) => {
    view.setTime(seconds);
    const said = whoAt(scene, seconds);
    if ($("said").textContent !== said) $("said").textContent = said;
  };
  clock.on("time", show);
  show(0);
  window.recipeViewer = { clock, view, scene };
}
