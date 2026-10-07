/** Recipes: a list, and for the one chosen what it says and what it breaks.
 *
 * Nothing is judged here: the summary and the violations are the server's,
 * which are those of `reverberate.scenes`.
 */
const $ = (id) => document.getElementById(id);
const cell = (text) => {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
};
const sources = (counts) =>
  ["near_voice", "far_voice", "noise"]
    .filter((kind) => counts[kind])
    .map((kind) => `${counts[kind]} ${kind.replace("_", " ")}`)
    .join(", ");

const told = await fetch("api/recipes").then((response) => response.json());
$("what").textContent = `${told.recipes.length} recipe(s) under ${told.folder}`;

function show(report) {
  $("one").hidden = false;
  $("one-name").textContent = report.name;
  $("verdict").textContent = report.valid ? "valid" : `${report.violations.length} rule(s) broken`;
  $("verdict").className = `verdict ${report.valid ? "good" : "p-bad"}`;
  $("violations").textContent = "";
  for (const { rule, message } of report.violations) {
    const item = document.createElement("li");
    item.textContent = `rule ${rule}: ${message}`;
    $("violations").append(item);
  }
  const notes = [];
  if (report.floor_checked === false) notes.push("Checked without the dwelling's floor: the rules that need it were not applied.");
  if (report.placeholders) notes.push(`Generated, not saved: ${report.placeholders}.`);
  $("summary").textContent = [report.summary || report.error || "", ...notes].join("\n\n");
}

for (const recipe of told.recipes) {
  const row = document.createElement("tr");
  if (recipe.error) row.append(cell(recipe.name), cell(recipe.error));
  else {
    row.append(
      cell(recipe.name),
      cell(recipe.dwelling),
      cell(`${recipe.seed}`),
      cell(`${recipe.duration_s} s`),
      cell(sources(recipe.sources)),
      cell(recipe.sha256.slice(0, 12))
    );
  }
  if (recipe.error) row.lastChild.colSpan = 5;
  row.addEventListener("click", async () => {
    for (const other of $("list").children) other.classList.toggle("on", other === row);
    $("one").hidden = false;
    $("one-name").textContent = recipe.name;
    $("verdict").textContent = "reading";
    $("summary").textContent = "";
    $("violations").textContent = "";
    const path = recipe.name.split("/").map(encodeURIComponent).join("/");
    show(await fetch(`api/recipes/${path}`).then((response) => response.json()));
  });
  $("list").append(row);
}

if (!told.can_generate) {
  $("draw").disabled = true;
  $("generate-note").textContent = "No dataset on this server: start it with --hssd-root to generate.";
}
$("draw").addEventListener("click", async () => {
  $("generate-note").textContent = "generating";
  const body = JSON.stringify({ dwelling: $("dwelling").value.trim(), seed: Number($("seed").value) });
  const report = await fetch("api/generate", { method: "POST", body }).then((response) => response.json());
  if (report.error) {
    $("generate-note").textContent = report.error;
    return;
  }
  $("generate-note").textContent = report.can_save ? "" : "Start with --save-in to keep it.";
  $("save").disabled = !report.can_save;
  show(report);
});
$("save").addEventListener("click", async () => {
  const answer = await fetch("api/save", { method: "POST", body: "{}" }).then((response) => response.json());
  $("generate-note").textContent = answer.error || `saved in ${answer.saved}`;
});
