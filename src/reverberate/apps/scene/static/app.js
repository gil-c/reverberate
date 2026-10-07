/** Scene player: the parts joined, and nothing else.
 *
 * The scene is drawn where the pack says everything is, the player plays the
 * sources' stems summed under the track list's balance, and the lights and
 * the meters follow each stem's own level, which comes with its chunks as
 * they are rendered.
 */
import { createPlayer } from "./parts/player.js";
import { createSceneView } from "./parts/scene-view.js";
import { createTimeline, whoAt } from "./parts/timeline.js";
import { createTrackList } from "./parts/track-list.js";
import { createTransport } from "./parts/transport.js";

const $ = (id) => document.getElementById(id);
const json = (url, options) => fetch(url, options).then((response) => response.json());
const [told, scene] = await Promise.all([json("api/player"), json("api/scene")]);
const minutes = `${Math.floor(told.duration_s / 60)} min ${Math.round(told.duration_s % 60)} s`;
$("what").textContent = `${told.name}: ${minutes} in ${told.dwelling}, ${scene.sources.length} sources.`;
if (told.note) $("scene-note").textContent += ` ${told.note}`;

const player = createPlayer();
createTransport($("transport"), player, { bar: false, follow: true });
player.setSceneHead(told.head);
const tracks = createTrackList($("tracks"), { onBalance: (version) => player.setBalance(version), save: true, lanes: false });
await tracks.ready;
await player.load([told.item]);

// Each stem's level a step, filled as its chunks are rendered; nothing where they are not yet.
const steps = Math.ceil(told.duration_s / told.head.step_s);
const levels = { hop_s: told.head.step_s, stems: Object.fromEntries(scene.sources.map((source) => [source.id, new Array(steps).fill(null)])) };
async function readLevels(seconds) {
  const query = new URLSearchParams({ item: told.item.id, from: Math.max(seconds - 0.5, 0), to: seconds + 2.5 });
  const answer = await json(`api/levels?${query}`);
  for (const [id, values] of Object.entries(answer.stems)) {
    values.forEach((value, k) => {
      if (value !== null) levels.stems[id][answer.first + k] = value;
    });
  }
}
tracks.setLevels(levels);

const view = createSceneView($("scene"), scene, { gainOf: (id) => tracks.gainOf(id) });
view.setLevels(levels);
createTimeline($("timeline"), scene, player);

function showTime(seconds) {
  tracks.setTime(seconds);
  view.setTime(seconds);
  // The head drawn is the head listened with: the scene's while it is followed, the listener's once freed.
  view.setHead(player.state().following ? null : player.pose().yaw);
  const said = whoAt(scene, seconds);
  if ($("said").textContent !== said) $("said").textContent = said;
}
player.on("time", showTime).on("head", () => showTime(player.time())).on("state", () => showTime(player.time()));
showTime(0);

// The render is told where the listener is before he plays, so that the first sound waits less.
let warmed = -1;
function warm(seconds) {
  if (Math.abs(seconds - warmed) < 0.25) return;
  warmed = seconds;
  fetch(`api/warm?${new URLSearchParams({ at: seconds })}`, { method: "POST", body: "{}" });
}
warm(0);

let before = 0;
setInterval(async () => {
  const { playing } = player.state();
  const now = player.time();
  if (playing) await readLevels(now);
  else warm(now);
  // What is asked for is rendered then: a wait is said, and is the render's.
  const waiting = playing && Math.abs(now - before) < 0.02;
  $("said").classList.toggle("p-note", waiting);
  if (waiting) $("said").textContent = "Rendering the sound from here…";
  before = now;
}, 250);

window.scenePlayer = { player, tracks, view, levels, scene, told };
