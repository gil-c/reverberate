/** Scene player: the parts joined, and nothing else.
 *
 * The scene is drawn where the pack says everything is, the player plays the
 * sources' stems summed under the track list's balance, and the lights and
 * the meters follow each stem's own level.
 */
import { createPlayer } from "./parts/player.js";
import { createSceneView } from "./parts/scene-view.js";
import { createTrackList } from "./parts/track-list.js";
import { createTransport } from "./parts/transport.js";

const $ = (id) => document.getElementById(id);
const told = await fetch("api/player").then((response) => response.json());
const [from, to] = told.window_s || [0, 0];
$("what").textContent = `${told.folder.split("/").pop()}, ${from} s to ${to} s of the scene.`;

const player = createPlayer();
createTransport($("transport"), player);
player.setSceneHead(told.head);
const tracks = createTrackList($("tracks"), { onBalance: (version) => player.setBalance(version) });
await tracks.ready;

const items = told.variants.map((name) => told.items[name]);
await player.load(items);
const levels = await fetch(`api/levels?${new URLSearchParams({ item: items[0].id })}`).then((response) => response.json());
tracks.setLevels(levels);

let view = null;
const scene = await fetch("api/scene").then((response) => (response.ok ? response.json() : null));
if (scene) {
  view = createSceneView($("scene"), scene, { gainOf: (id) => tracks.gainOf(id) });
  view.setLevels(levels);
  player.on("head", ({ yaw }) => view.setTurn(yaw));
} else {
  $("scene").textContent = "The pack this folder was rendered from is not there: start again with --pack to see the scene.";
  $("scene-note").hidden = true;
}
player.on("time", (seconds) => {
  tracks.setTime(seconds);
  if (view) view.setTime(seconds);
});

// The renders of the folder, when it holds several: the same instant of another.
$("render-row").hidden = items.length < 2;
function showRenders() {
  $("renders").textContent = "";
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `p-button p-small${player.state().item === item.id ? " on" : ""}`;
    button.textContent = item.label;
    button.addEventListener("click", () => player.select(item.id));
    $("renders").append(button);
  }
}
player.on("state", showRenders);
showRenders();
window.scenePlayer = { player, tracks, view };
