/** Compare: the variants of one scene, switched at the same instant, seen, and judged blind.
 *
 * The page is the components of `reverberate.viz.parts` and what joins them:
 * which set is heard (a signal, a source), the buttons of the variants, and
 * the blind test's screens. It computes nothing: the frames, the sonograms,
 * the measured differences and the test's key are the server's.
 */
import { drawBands } from "./parts/bands.js";
import { createPlayer } from "./parts/player.js";
import { createSonogram } from "./parts/sonogram.js";
import { createTransport } from "./parts/transport.js";

const $ = (id) => document.getElementById(id);
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
const SIGNAL_NAMES = { clips: "Speech", clicks: "Clicks", pink: "Pink noise" };
const PART_COLOURS = { early: "#1f6feb", late: "#e4572e", clips: "#17a65b" };

const kit = await fetch("api/kit").then((response) => response.json());
const itemOf = Object.fromEntries(kit.items.map((item) => [item.id, item]));
const player = createPlayer();
createTransport($("transport"), player);
const sonogram = createSonogram($("sonogram"), { player });
player.setSceneHead(kit.head);
player.setLooping(true);

let signal = Object.keys(kit.sets)[0];
let track = "mix";
let mode = "free";
let view = "side";
let blind = null; // the test's state, as the server tells it
let blindKind = "abx";
let ranked = [];

const [from, to] = kit.window_s || [0, 0];
$("what").textContent =
  `${kit.variants.length} renders of one scene, ${from} s to ${to} s. ` +
  `The reference is ${kit.variants[0]}. ` +
  (kit.head_name ? `Order 7 is decoded with ${kit.head_name}.` : "");

const variantsOf = () => kit.sets[signal][track] || kit.sets[signal][Object.keys(kit.sets[signal])[0]];
const rows = () => kit.variants.map((name) => itemOf[variantsOf()[name]]);

// --- what is heard ------------------------------------------------------------------

function showChoices() {
  $("signals").textContent = "";
  for (const name of Object.keys(kit.sets)) {
    const button = el("button", { type: "button", class: `p-button p-small${name === signal ? " on" : ""}`, title: kit.signals[name] || "", text: SIGNAL_NAMES[name] || name });
    button.addEventListener("click", () => chooseSet(name, track));
    $("signals").append(button);
  }
  $("signal-field").hidden = Object.keys(kit.sets).length < 2;
  const tracks = Object.keys(kit.sets[signal]);
  if (!tracks.includes(track)) track = tracks.includes("mix") ? "mix" : tracks[0];
  $("tracks").textContent = "";
  for (const name of tracks) $("tracks").append(el("option", { value: name, text: name === "mix" ? "all (the mix)" : name }));
  $("tracks").value = track;
  const alone = tracks.length < 2 || (tracks.length === 2 && tracks.includes("mix") && JSON.stringify(kit.sets[signal].mix) === JSON.stringify(kit.sets[signal][tracks.find((t) => t !== "mix")]));
  $("track-field").hidden = alone;
}
$("tracks").addEventListener("change", () => chooseSet(signal, $("tracks").value));

async function chooseSet(nextSignal, nextTrack) {
  signal = nextSignal;
  track = nextTrack;
  showChoices();
  if (mode === "blind" && blind && !blind.finished) return;
  await player.load(rows(), { keep: true });
  showVariants();
  showSonograms();
}

function showVariants() {
  const heard = player.state().item;
  $("variants").textContent = "";
  rows().forEach((item, index) => {
    const button = el("button", { type: "button", class: `p-button p-big${item.id === heard ? " on" : ""}` }, el("kbd", { text: `${index + 1}` }), item.label);
    button.addEventListener("click", () => player.select(item.id));
    $("variants").append(button);
  });
  const item = itemOf[heard];
  $("variant-note").textContent = item
    ? item.kind === "ambisonic"
      ? "Order 7, decoded here: turn your head with the dial or the arrow keys. A switch is at the same instant."
      : "Two ears as the kit wrote them, for the scene's own head. A switch is at the same instant."
    : "";
}
player.on("state", () => {
  if (mode === "free") showVariants();
  else showBlindButtons();
});

function showSonograms() {
  if (mode !== "free") return sonogram.clear();
  const items = rows().map((item) => ({ id: item.id, label: item.label }));
  sonogram.show(items, view === "difference" ? { less: 0 } : {});
}
for (const button of $("view").children) {
  button.addEventListener("click", () => {
    view = button.dataset.view;
    for (const other of $("view").children) other.classList.toggle("on", other === button);
    showSonograms();
  });
}

// --- what was measured --------------------------------------------------------------

const measured = kit.differences;
function showMeasured() {
  const variants = Object.keys(measured.by_variant);
  $("measured").hidden = !variants.length || !measured.third_octaves_hz.length;
  if ($("measured").hidden) return;
  if (!$("diff-variant").options.length) {
    for (const name of variants) $("diff-variant").append(el("option", { value: name, text: `${name} less ${measured.reference}` }));
  }
  const sources = Object.keys(measured.by_variant[$("diff-variant").value]);
  const kept = $("diff-source").value;
  $("diff-source").textContent = "";
  for (const name of sources) $("diff-source").append(el("option", { value: name, text: name }));
  $("diff-source").value = sources.includes(kept) ? kept : sources.includes(track) ? track : sources[0];
  const record = measured.by_variant[$("diff-variant").value][$("diff-source").value];
  const said = {
    early: `the first ${1000 * (measured.early_s || 0.05)} ms of the response to an impulse`,
    late: "the rest of that response: the room",
    clips: "the speech over the whole window",
  };
  const series = ["early", "late", "clips"]
    .filter((part) => record[part])
    .map((part) => ({ label: `${part}: ${said[part]}`, values: record[part], colour: PART_COLOURS[part] }));
  drawBands($("bands"), { bandsHz: measured.third_octaves_hz, series, rangeDb: 12, markHz: measured.crossover_hz, unit: "dB" });
  $("bands-note").textContent =
    `The dashed line is the crossover, ${measured.crossover_hz} Hz: under it the wave solve, over it the mirror. ` +
    "A band left out holds nothing in one of the two. The scale is fixed at 12 dB either side.";
  const table = $("worst");
  table.textContent = "";
  table.append(el("tr", {}, ...["Largest difference, dB", "early, low", "early, high", "late, low", "late, high", "speech, low", "speech, high"].map((text) => el("th", { text }))));
  const cell = (value) => el("td", { text: value === null || value === undefined ? "" : value.toFixed(1) });
  for (const [name, values] of Object.entries(measured.by_variant[$("diff-variant").value])) {
    const worst = values.worst || {};
    table.append(
      el("tr", {}, el("td", { text: name }), ...["early", "late", "clips"].flatMap((part) => ["under_the_crossover", "over_the_crossover"].map((side) => cell((worst[part] || {})[side]))))
    );
  }
}
$("diff-variant").addEventListener("change", showMeasured);
$("diff-source").addEventListener("change", showMeasured);

// --- the blind test -----------------------------------------------------------------

const post = (path, body) => fetch(path, { method: "POST", body: JSON.stringify(body || {}) }).then((response) => response.json());

function showMode() {
  for (const button of $("mode").children) button.classList.toggle("on", button.dataset.mode === mode);
  const testing = mode === "blind";
  $("free").hidden = testing;
  $("blind").hidden = !testing;
  // Nothing that names a variant is in sight while a test runs.
  const running = testing && blind && !blind.finished && blind.kind;
  $("look").hidden = testing;
  $("measured").hidden = testing || !Object.keys(measured.by_variant).length;
  $("signal-field").style.visibility = running ? "hidden" : "";
  $("track-field").style.visibility = running ? "hidden" : "";
  $("blind-setup").hidden = Boolean(running) || Boolean(blind && blind.finished);
  $("blind-run").hidden = !running;
  $("blind-result").hidden = !(testing && blind && blind.finished);
}
for (const button of $("mode").children) {
  button.addEventListener("click", async () => {
    if (button.dataset.mode === mode) return;
    if (mode === "blind" && blind && !blind.finished && blind.kind) {
      if (!window.confirm("Stop the blind test? What was answered is written.")) return;
      blind = await post("api/blind/stop");
    }
    mode = button.dataset.mode;
    player.stop();
    if (mode === "free") {
      blind = null;
      await player.load(rows());
      showVariants();
      showMeasured();
    } else showBlindSetup();
    showMode();
    showSonograms();
  });
}

function showBlindSetup() {
  for (const select of [$("blind-a"), $("blind-b")]) {
    select.textContent = "";
    for (const name of kit.variants) select.append(el("option", { value: name, text: name }));
  }
  $("blind-a").value = kit.variants[0];
  $("blind-b").value = kit.variants[1];
  showBlindKind();
}
function showBlindKind() {
  for (const button of $("blind-kind").children) button.classList.toggle("on", button.dataset.kind === blindKind);
  $("blind-pair").hidden = blindKind !== "abx";
  $("blind-says").textContent =
    blindKind === "abx"
      ? "A and B are named. X is one of the two, drawn again for each trial: listen to all three as long as you like, then say which X is. Nothing is told until the last trial; the score is then given with the probability of reaching it by guessing."
      : "Every variant is under a name that says nothing, drawn again for each trial. Listen, then put them in the order you prefer. The names are told after the last trial.";
}
for (const button of $("blind-kind").children) {
  button.addEventListener("click", () => {
    blindKind = button.dataset.kind;
    showBlindKind();
  });
}

$("blind-start").addEventListener("click", async () => {
  const chosen = blindKind === "abx" ? [$("blind-a").value, $("blind-b").value] : kit.variants;
  if (new Set(chosen).size !== chosen.length) {
    $("blind-says").textContent = "A and B are the same variant: there is nothing to tell apart.";
    return;
  }
  const state = player.state();
  blind = await post("api/blind/start", {
    kind: blindKind,
    trials: Number($("blind-trials").value),
    items: Object.fromEntries(chosen.map((name) => [name, variantsOf()[name]])),
    context: { signal, track, level_db: state.levelDb, loop_s: state.loop },
  });
  if (blind.error) {
    $("blind-says").textContent = blind.error;
    blind = null;
    return;
  }
  await loadBlind();
  showMode();
});

/** The names the test plays, as items of the player: each asks the server for what it hides. */
async function loadBlind() {
  const like = rows()[0];
  const names = [...Object.keys(blind.known), ...blind.hidden];
  player.stop();
  await player.load(names.map((name) => ({ ...like, id: `blind:${name}`, label: name })));
  ranked = [];
  showBlind();
}

function showBlindButtons() {
  if (!blind || blind.finished) return;
  const heard = player.state().item;
  $("blind-listen").textContent = "";
  player.items().forEach((item, index) => {
    const button = el("button", { type: "button", class: `p-button p-big${item.id === heard ? " on" : ""}` }, el("kbd", { text: `${index + 1}` }), item.label);
    button.addEventListener("click", () => player.select(item.id));
    $("blind-listen").append(button);
  });
}

function showBlind() {
  if (!blind) return;
  if (blind.finished) return showResult();
  $("blind-trial").textContent = `Trial ${blind.trial} of ${blind.trials}`;
  showBlindButtons();
  const answer = $("blind-answer");
  answer.textContent = "";
  if (blind.kind === "abx") {
    $("blind-hint").textContent = "Listen, then say which one X is.";
    for (const name of ["A", "B"]) {
      const button = el("button", { type: "button", class: "p-button", text: `X is ${name}` });
      button.addEventListener("click", () => answered(name));
      answer.append(button);
    }
  } else {
    $("blind-hint").textContent = "Listen, then click the names in the order you prefer, the best first.";
    const order = el("span", { class: "rank-order p-said", text: ranked.length ? `Your order: ${ranked.join(", ")}` : "Your order: none yet" });
    for (const name of blind.hidden) {
      const button = el("button", { type: "button", class: "p-button", text: `${name} next` });
      button.disabled = ranked.includes(name);
      button.addEventListener("click", () => {
        ranked.push(name);
        showBlind();
      });
      answer.append(button);
    }
    const clear = el("button", { type: "button", class: "p-button p-small", text: "Clear" });
    clear.addEventListener("click", () => {
      ranked = [];
      showBlind();
    });
    const submit = el("button", { type: "button", class: "p-button on", text: "This is my order" });
    submit.disabled = ranked.length !== blind.hidden.length;
    submit.addEventListener("click", () => answered(ranked));
    answer.append(order, clear, submit);
  }
}

async function answered(given) {
  const state = await post("api/blind/answer", { answer: given });
  if (state.error) {
    $("blind-hint").textContent = state.error;
    return;
  }
  blind = state;
  ranked = [];
  if (blind.finished) {
    player.stop();
    showMode();
    return showResult();
  }
  // What the hidden names stand for has been drawn again: the one heard is asked again.
  player.refresh();
  showBlind();
}

$("blind-stop").addEventListener("click", async () => {
  blind = await post("api/blind/stop");
  player.stop();
  showMode();
  showResult();
});

function showResult() {
  const box = $("blind-result");
  const { result } = blind;
  box.textContent = "";
  const percent = (p) => (p === null ? "" : p < 0.001 ? "under 0.1 %" : `${(100 * p).toFixed(1)} %`);
  if (blind.kind === "abx") {
    box.append(
      el("div", { class: "score", text: `${result.right} right out of ${result.answered}` }),
      el("p", { text: result.answered ? `The probability of this score or a better one by guessing: ${percent(result.chance)}.` : "No trial was answered." })
    );
  } else {
    const table = el("table", {}, el("tr", {}, ...["Variant", "came first", "mean rank"].map((text) => el("th", { text }))));
    for (const name of Object.keys(result.first)) {
      table.append(el("tr", {}, el("td", { text: name }), el("td", { text: `${result.first[name]} of ${result.answered}` }), el("td", { text: result.mean_rank[name] === null ? "" : result.mean_rank[name].toFixed(2) })));
    }
    box.append(table, el("p", { text: result.answered ? `The probability that some variant comes first this often when the order is chance: ${percent(result.chance)}.` : "No trial was answered." }));
  }
  if (blind.answered < blind.trials) box.append(el("p", { class: "p-note", text: `Stopped after ${blind.answered} of ${blind.trials} trials.` }));
  if (blind.saved) box.append(el("p", { class: "p-note", text: `Written in ${blind.saved}` }));
  const again = el("button", { type: "button", class: "p-button", text: "Another test" });
  again.addEventListener("click", () => {
    blind = null;
    showBlindSetup();
    showMode();
  });
  box.append(again);
}

// --- keys: 1 to N are the buttons in sight ------------------------------------------

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) return;
  if (event.metaKey || event.ctrlKey || event.altKey || !/^[1-9]$/.test(event.key)) return;
  const item = player.items()[Number(event.key) - 1];
  if (item && (mode === "free" || (blind && !blind.finished))) player.select(item.id);
});

showChoices();
await player.load(rows());
showVariants();
showMeasured();
showMode();
showSonograms();
window.compare = { player, kit };
