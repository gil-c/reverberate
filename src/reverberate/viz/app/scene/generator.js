/** The generator's panel and the scene's summary.
 *
 * The panel is built from the schema the server reads off the generator's own
 * parameters: one row per parameter, a minimum and a maximum for a range, each
 * held inside the limits the schema gives. Nothing about a parameter is
 * written here but how its unit is spelled.
 */
const UNITS = [
  ["_m_s", "m/s"],
  ["_deg_s", "°/s"],
  ["_deg", "°"],
  ["_db", "dB"],
  ["_share", ""],
  ["_s", "s"],
  ["_m", "m"],
];

/** A key of the format as a label and a unit: `turn_rate_deg_s` is "turn rate", "°/s". */
export function labelOf(key) {
  for (const [suffix, unit] of UNITS) {
    if (key.endsWith(suffix)) {
      const bare = key.slice(0, -suffix.length) + (suffix === "_share" ? " share" : "");
      return { label: bare.replaceAll("_", " "), unit };
    }
  }
  return { label: key.replaceAll("_", " "), unit: "" };
}

/** A value brought inside `[min, max]` and onto the parameter's kind. */
export function held(entry, value) {
  const number = entry.type === "integer" ? Math.round(value) : value;
  return Math.min(entry.max, Math.max(entry.min, number));
}

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

export function createGeneratorPanel(root, { onGenerate, onSave, onLoad }) {
  const saved = make("select");
  const seedRow = make("label", "field gen-seed");
  const seed = make("input");
  seed.type = "number";
  Object.assign(seed, { min: 0, step: 1, value: 1 });
  const dice = make("button", "gen-dice", "⚄");
  dice.type = "button";
  dice.title = "another seed";
  seedRow.append(make("span", "", "seed"), seed, dice);
  const actions = make("div", "gen-actions");
  const generate = make("button", "gen-go", "Generate");
  generate.type = "button";
  const reset = make("button", "", "Defaults");
  reset.type = "button";
  actions.append(generate, reset);
  const saveRow = make("div", "gen-save");
  const name = make("input");
  name.placeholder = "name";
  name.spellcheck = false;
  const save = make("button", "", "Save");
  save.type = "button";
  save.disabled = true;
  saveRow.append(name, save);
  const status = make("div", "note gen-status");
  const fields = make("div", "gen-fields");
  root.replaceChildren(saved, seedRow, actions, saveRow, status, fields);

  let schema = [];
  const inputs = new Map();

  function number(entry) {
    const input = make("input");
    input.type = "number";
    Object.assign(input, { min: entry.min, max: entry.max, step: entry.step, disabled: entry.fixed });
    input.title = `${entry.path}: ${entry.min} to ${entry.max}`;
    return input;
  }

  function build() {
    fields.replaceChildren();
    inputs.clear();
    // The dataclass lists its fields by subject; the panel gathers them by
    // where the recipe writes them, in the order each place first appears.
    const places = [...new Set(schema.map((entry) => entry.group))];
    const ordered = places.flatMap((place) => schema.filter((entry) => entry.group === place));
    let group = null;
    for (const entry of ordered) {
      if (entry.group !== group) {
        group = entry.group;
        if (group) fields.append(make("p", "label gen-group", group.replaceAll(".", " · ").replaceAll("_", " ")));
      }
      const { label, unit } = labelOf(entry.key);
      const row = make("label", `gen-row ${entry.kind}`);
      const boxes = entry.kind === "range" ? [number(entry), number(entry)] : [number(entry)];
      row.append(make("span", "", label), ...boxes, make("em", "", unit));
      fields.append(row);
      inputs.set(entry.name, boxes);
      boxes.forEach((box, index) => {
        box.addEventListener("change", () => {
          const value = Number.parseFloat(box.value);
          box.value = held(entry, Number.isFinite(value) ? value : [entry.default].flat()[index]);
          // A range is kept in order by moving its other end.
          const other = boxes[1 - index];
          if (other && index === 0 && Number(other.value) < Number(box.value)) other.value = box.value;
          if (other && index === 1 && Number(other.value) > Number(box.value)) other.value = box.value;
        });
      });
    }
  }

  function put(flat) {
    for (const entry of schema) {
      if (!(entry.name in flat)) continue;
      const values = [flat[entry.name]].flat();
      inputs.get(entry.name).forEach((box, index) => {
        box.value = values[index];
      });
    }
  }
  const defaults = () => Object.fromEntries(schema.map((entry) => [entry.name, entry.default]));

  generate.addEventListener("click", () => onGenerate());
  reset.addEventListener("click", () => put(defaults()));
  dice.addEventListener("click", () => {
    seed.value = Math.floor(Math.random() * 1e9);
  });
  save.addEventListener("click", () => onSave(name.value.trim()));
  saved.addEventListener("change", () => {
    if (saved.value) onLoad(saved.value);
  });

  return {
    setSchema(next) {
      schema = next.parameters;
      build();
      put(defaults());
    },
    /** `{ field: value }` as the server takes it; a fixed parameter is left to its default. */
    values() {
      const flat = {};
      for (const entry of schema) {
        if (entry.fixed) continue;
        const values = inputs.get(entry.name).map((box) => held(entry, Number.parseFloat(box.value)));
        flat[entry.name] = entry.kind === "range" ? values : values[0];
      }
      return flat;
    },
    setValues: put,
    seed: () => Math.max(0, Math.round(Number(seed.value) || 0)),
    setSeed(value) {
      seed.value = value;
    },
    setName(value) {
      name.value = value;
    },
    setSaved(list, current = "") {
      const first = make("option", "", list.length ? "saved recipes" : "no saved recipe");
      first.value = "";
      saved.replaceChildren(
        first,
        ...list.map((entry) => {
          const text = entry.error
            ? `${entry.name} · unreadable`
            : `${entry.name} · seed ${entry.seed} · ${Math.round(entry.duration_s)} s · ${entry.sources} sources`;
          const option = make("option", "", text);
          option.value = entry.name;
          return option;
        })
      );
      saved.value = list.some((entry) => entry.name === current) ? current : "";
    },
    setBusy(on) {
      generate.disabled = on;
    },
    canSave(on) {
      save.disabled = !on;
    },
    setStatus(text, bad = false) {
      status.textContent = text || "";
      status.classList.toggle("bad", bad);
    },
  };
}

/** The right panel in scene mode: what the recipe is, what it costs, what it breaks. */
export function createSummary(root) {
  const facts = make("dl", "facts");
  const rules = make("div", "scene-rules");
  const words = make("pre", "scene-describe");
  const first = make("div", "stack");
  first.append(make("p", "label", "Scene"), facts);
  const second = make("div");
  second.append(make("p", "label", "Validation"), rules);
  const third = make("div");
  third.append(make("p", "label", "Summary"), words);
  root.replaceChildren(first, second, third);

  const fact = (list) =>
    facts.replaceChildren(
      ...list.flatMap(([key, value, title]) => {
        const dd = make("dd", "", value);
        if (title) dd.title = title;
        return [make("dt", "", key), dd];
      })
    );

  return {
    /** A dwelling with no recipe yet: what its layout offers. */
    showLayout(layout) {
      fact(layout ? [["dwelling", layout.dwelling.name], ["layout", layout.summary.split(": ").slice(1).join(": ")]] : []);
      rules.replaceChildren(make("div", "note", layout ? "no recipe: generate one, or load one" : ""));
      words.textContent = "";
    },
    show(report) {
      const recipe = report.recipe;
      const cost = report.low_band_positions;
      const lines = [
        ["recipe", report.recipe_sha256.slice(0, 16), report.recipe_sha256],
        ["dwelling", recipe.dwelling.name],
        ["seed", String(recipe.seed)],
        ["duration", `${recipe.duration_s} s`],
        ["sources", String(recipe.sources.length)],
        [
          "low band positions",
          String(cost.total),
          `${cost.stations} stations, ${cost.rail_samples} on rails, ${cost.seat_rail_samples} on seats' vertical rails: one wave solve each`,
        ],
        ["  stations", String(cost.stations)],
        ["  on rails", String(cost.rail_samples)],
        ["  on seats", String(cost.seat_rail_samples)],
      ];
      if (report.placeholder_clips) lines.push(["clips", "placeholders", "the recipe names no audio; a trace refuses it"]);
      if (report.placeholder_assets) lines.push(["assets", "placeholders", "the asset keys match nothing; a trace refuses it"]);
      if (recipe.generator) lines.push(["generator", `${recipe.generator.name} ${recipe.generator.version}`]);
      fact(lines);
      const found = report.violations.map((violation) => {
        const row = make("div", "violation");
        row.append(make("b", "", `rule ${violation.rule}`), make("span", "", violation.message));
        return row;
      });
      if (!found.length) found.push(make("div", "note ok", "valid: no rule is broken"));
      if (!report.floor_checked) {
        found.push(make("div", "note", `the floor was not read: rules ${report.floor_rules.join(", ")} were checked in part`));
      }
      rules.replaceChildren(...found);
      words.textContent = report.describe;
    },
  };
}
