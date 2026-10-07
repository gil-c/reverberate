/** Recipes: a line a recipe, one more drawn, one thrown away, one opened in the viewer.
 *
 * Nothing is judged here: every number and every rule is the server's, which
 * are those of `reverberate.scenes`.
 */
const $ = (id) => document.getElementById(id);
const json = async (url, options) => {
  const response = await fetch(url, options);
  const answer = await response.json();
  if (!response.ok) throw new Error(answer.error || `${response.status}`);
  return answer;
};
const el = (tag, text = "", className = "") => {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
};
const path = (name) => name.split("/").map(encodeURIComponent).join("/");
const length = (seconds) => (seconds >= 60 ? `${Math.floor(seconds / 60)} min ${String(Math.round(seconds % 60)).padStart(2, "0")} s` : `${Math.round(seconds)} s`);

let told = null;
let preset = "medium";
let chosen = null;
const rows = {};

function showOne(report) {
  chosen = report.name;
  for (const [name, row] of Object.entries(rows)) row.classList.toggle("on", name === chosen);
  $("one").hidden = false;
  $("one-name").textContent = report.name;
  $("verdict").textContent = report.valid ? "keeps every rule" : `breaks ${report.violations.length} rule(s)`;
  $("verdict").className = report.valid ? "good" : "p-bad";
  $("violations").textContent = "";
  for (const { rule, message } of report.violations) $("violations").append(el("li", `rule ${rule}: ${message}`));
  const notes = [];
  if (report.floor_checked === false) notes.push("Checked without the dwelling's floor: the rules that need it were applied in part.");
  if (report.placeholders && report.placeholders.length) notes.push(`It names placeholders for ${report.placeholders.join(" and ")}: it can be looked at, and a trace refuses it.`);
  if (report.cost && report.cost.note) notes.push(`Cost: ${report.cost.note}.`);
  $("one-note").textContent = notes.join(" ");
  $("summary").textContent = report.summary || report.error || "";
}

/** The cells that take a moment, filled when the server has them. */
function fill(row, report) {
  const [snr, cost, rules] = row.late;
  const heard = report.speech_over_noise;
  snr.textContent = heard ? `${heard.least_db} dB at worst, ${heard.median_db} dB typical` : "";
  snr.className = "n w";
  cost.textContent = report.cost && report.cost.usd !== undefined ? `${report.cost.usd.toFixed(2)} USD, ${Math.round(report.cost.hours * 60)} min` : "";
  rules.textContent = report.valid ? "kept" : `${report.violations.length} broken`;
  rules.className = report.valid ? "good" : "p-bad";
}

function line(recipe) {
  const row = document.createElement("tr");
  rows[recipe.name] = row;
  if (recipe.error) {
    const said = el("td", recipe.error, "p-bad");
    said.colSpan = 8;
    row.append(el("td", recipe.name), said);
    row.late = [el("td"), el("td"), el("td")];
  } else {
    row.late = [el("td", "…", "n"), el("td", "…", "n"), el("td", "…")];
    row.append(
      el("td", recipe.name),
      el("td", length(recipe.duration_s), "n"),
      el("td", `${recipe.people}${recipe.own_voice ? " and you" : ""}`, "n"),
      el("td", `${recipe.groups || ""}`, "n"),
      el("td", recipe.noises.join(", ") || "none"),
      el("td", recipe.calmness === null ? "" : `${recipe.calmness}`, "n"),
      ...row.late
    );
  }
  const actions = el("td", "", "do");
  const open = el("button", "Open", "p-button p-small");
  open.title = "See the scene play, without sound";
  open.disabled = Boolean(recipe.error);
  open.addEventListener("click", (event) => {
    event.stopPropagation();
    window.open(`viewer.html?recipe=${encodeURIComponent(recipe.name)}`, "_blank");
  });
  const remove = el("button", "Delete", "p-button p-small");
  remove.title = "Move it to the trash folder: nothing is erased";
  remove.addEventListener("click", async (event) => {
    event.stopPropagation();
    try {
      const answer = await json("api/delete", { method: "POST", body: JSON.stringify({ name: recipe.name }) });
      $("list-note").textContent = `${recipe.name} was moved to ${answer.moved_to}`;
      if (chosen === recipe.name) $("one").hidden = true;
      await list();
    } catch (error) {
      $("list-note").textContent = error.message;
    }
  });
  actions.append(open, remove);
  row.append(actions);
  row.addEventListener("click", async () => showOne(await json(`api/recipes/${path(recipe.name)}`)));
  return row;
}

async function list() {
  told = await json("api/recipes");
  $("what").textContent = `${told.recipes.length} recipe(s) in ${told.folder}`;
  $("list").textContent = "";
  for (const name of Object.keys(rows)) delete rows[name];
  for (const recipe of told.recipes) $("list").append(line(recipe));
  $("dwellings").textContent = "";
  for (const dwelling of told.dwellings) $("dwellings").append(Object.assign(document.createElement("option"), { value: dwelling }));
  if (!$("dwelling").value && told.dwellings.length) $("dwelling").value = told.dwellings[0];
  // One at a time: the cost of a recipe is its trace's plan, a second of work.
  for (const recipe of told.recipes) {
    if (recipe.error || !rows[recipe.name]) continue;
    try {
      fill(rows[recipe.name], await json(`api/recipes/${path(recipe.name)}`));
    } catch {
      /* the recipe went while the list was read: the next list says so */
    }
  }
}

function showPresets() {
  $("preset").textContent = "";
  for (const name of told.presets) {
    const button = el("button", name[0].toUpperCase() + name.slice(1), `p-button p-small${name === preset ? " on" : ""}`);
    button.type = "button";
    button.addEventListener("click", () => {
      preset = name;
      showPresets();
    });
    $("preset").append(button);
  }
}

$("draw").addEventListener("click", async () => {
  $("generate-note").className = "p-note";
  $("generate-note").textContent = "Generating…";
  const body = { dwelling: $("dwelling").value.trim(), seed: Number($("seed").value), preset, duration_s: Number($("duration").value) };
  try {
    const report = await json("api/generate", { method: "POST", body: JSON.stringify(body) });
    $("generate-note").textContent = `Written: ${report.name}`;
    await list();
    showOne(report);
  } catch (error) {
    $("generate-note").textContent = error.message;
    $("generate-note").className = "p-note bad";
  }
});

await list();
showPresets();
if (!told.can_generate) {
  $("draw").disabled = true;
  $("generate-note").textContent = "No dataset on this server: start it with --hssd-root to generate.";
}
window.recipesApp = { list, rows };
