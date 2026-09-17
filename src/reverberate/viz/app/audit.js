/** The Audit section of the left panel: what the view on screen shows, named.
 *
 * Colour shows the furnished model and needs no legend. Wave shows the grid
 * the wave solver read: a legend of its labels and, for the one picked, the
 * absorption the export gave it on PFFDTD's eleven bands. Mirror shows the
 * derived scene the geometric engine read: its census, a legend of labels
 * (or facet kinds) with what each contributed, and, for the label picked,
 * the absorption and scattering per band the catalogue holds and that each
 * mirror field's rays and image sources were handed.
 *
 * Picking a label, in the legend or by clicking a face in the view, singles
 * it out in the picture; picking it again, or Escape, clears it.
 */
import { HIGHLIGHT, materialColour } from "./grid.js";
import { KIND_COLOURS, ORDER_COLOURS } from "./mirror.js";

const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
};
const hex = (n) => `#${n.toString(16).padStart(6, "0")}`;
const fixed = (v, digits = 3) => (v === null || v === undefined ? "–" : Number(v).toFixed(digits));
const bandName = (hz) => (hz >= 1000 ? `${Number((hz / 1000).toFixed(2))}k` : `${Number(hz.toFixed(1))}`);

export function createAuditPanel(root, THREE, { onHighlight, onMirrorSwitch, onColourBy }) {
  const $ = (s) => root.querySelector(s);
  const rgb = new THREE.Color();
  const labelHex = (index) => `#${materialColour(rgb, index).getHexString()}`;
  let mode = "colour";
  let wave = null; // the mesh record of the band on screen
  let mirror = null; // run.mirror.check
  let picked = null; // { label, facet?, layer? }
  let colourBy = "label";
  let cell = null; // { id, position, paths }

  for (const box of root.querySelectorAll("[data-mirror]")) {
    box.addEventListener("change", () => onMirrorSwitch(box.dataset.mirror, box.checked));
  }
  $("#audit-colour-by").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    colourBy = button.dataset.by;
    for (const b of $("#audit-colour-by").querySelectorAll("button")) b.classList.toggle("on", b === button);
    onColourBy(colourBy);
    render();
  });
  addEventListener("keydown", (event) => {
    if (event.key === "Escape" && picked) pick(null);
  });

  const labels = () => (mode === "acoustic" ? (wave ? wave.labels : []) : mirror ? mirror.labels : []);

  function pick(next) {
    picked = next;
    const index = picked ? labels().indexOf(picked.label) : -1;
    HIGHLIGHT.value = index;
    onHighlight(index);
    render();
  }

  function problems(list) {
    const box = $("#audit-problems");
    box.replaceChildren(...(list || []).map((text) => el("div", { textContent: text })));
  }

  function facts(rows) {
    $("#audit-facts").replaceChildren(
      ...rows.flatMap(([name, value, title]) => [
        el("dt", { textContent: name }),
        el("dd", { textContent: value, title: title || "" }),
      ])
    );
  }

  function legend(rows, head) {
    const table = $("#audit-legend");
    table.replaceChildren(
      el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { textContent: h })))),
      el(
        "tbody",
        {},
        ...rows.map((row) => {
          const tr = el(
            "tr",
            {
              className: `${row.key !== undefined ? "pick" : ""} ${picked && row.key === picked.label ? "on" : ""}`,
              title: row.title || "",
            },
            el("td", {}, el("i", { className: "swatch", style: `background:${row.colour}` }), row.name),
            ...row.cells.map((c) => el("td", { textContent: c }))
          );
          if (row.key !== undefined) {
            tr.addEventListener("click", () => pick(picked && picked.label === row.key ? null : { label: row.key }));
          }
          return tr;
        })
      )
    );
  }

  function bandTable(bands, columns) {
    const head = el(
      "tr",
      {},
      el("th", { textContent: "Hz" }),
      ...columns.map((c) => el("th", { textContent: c.name, title: c.title || "" }))
    );
    const rows = bands.map((hz, b) =>
      el(
        "tr",
        {},
        el("th", { textContent: bandName(hz) }),
        ...columns.map((c) => el("td", { textContent: fixed(c.absorption ? c.absorption[b] : null) }))
      )
    );
    // Scattering is one number per label, and the wave solver has none.
    const scatter = columns.some((c) => c.scattering !== undefined)
      ? el(
          "tr",
          { className: "scatter" },
          el("th", { textContent: "s", title: "scattering coefficient, one for all bands" }),
          ...columns.map((c) => el("td", { textContent: c.scattering === undefined ? "" : fixed(c.scattering) }))
        )
      : null;
    return el("table", { className: "bands" }, el("thead", {}, head), el("tbody", {}, ...rows, scatter));
  }

  function renderWave() {
    $("#audit-lead").textContent = "What the wave solver read: its voxel grid, faces coloured by material label.";
    if (!wave) {
      problems(["no grid for this run"]);
      return;
    }
    const materials = wave.materials || {};
    problems(materials.problems);
    facts([
      ["band limit", `${Number(wave.fmax) / 1000} kHz`],
      ["solver step", wave.h_m ? `${(wave.h_m * 1000).toFixed(3)} mm` : "–"],
      ["far step", wave.coarse_m ? `${(wave.coarse_m * 1000).toFixed(2)} mm` : "–"],
      ["voxel cache", wave.cache_key ? wave.cache_key.slice(0, 12) : "–", wave.cache_key],
      ["model", materials.model ? materials.model.split("/").slice(-3).join("/") : "–", materials.model],
    ]);
    legend(
      [
        ...wave.labels.map((name, i) => ({
          key: name,
          name,
          colour: labelHex(i),
          cells: [materials.absorption && materials.absorption[i] ? fixed(materials.absorption[i][6], 2) : "–"],
          title: "absorption at 1 kHz; pick for every band",
        })),
        { name: "rigid, no material", colour: "rgb(115,115,122)", cells: [""] },
      ],
      ["label", "α 1k"]
    );
    const detail = $("#audit-detail");
    if (!picked) {
      detail.replaceChildren();
    } else {
      const i = wave.labels.indexOf(picked.label);
      detail.replaceChildren(
        el("p", { className: "label", textContent: picked.label }),
        el("div", { className: "note", textContent: "absorption the export gave the solver, PFFDTD's bands" }),
        bandTable(materials.bands_hz || [], [
          { name: "α", absorption: materials.absorption ? materials.absorption[i] : null },
        ])
      );
    }
    $("#audit-note").textContent = wave.note || "";
  }

  function renderMirror() {
    $("#audit-lead").textContent =
      "What the mirror read: the derived scene, checked on the server against mirror/scene.npz triangle for triangle.";
    if (!mirror) {
      problems(["this run carries no mirror"]);
      return;
    }
    problems([...(mirror.problems || []).map((p) => `refused: ${p}`), ...(mirror.material_problems || [])]);
    const totals = (mirror.census && mirror.census.totals) || {};
    const kinds = totals.facets_by_kind || {};
    facts([
      ["scene", `${String(mirror.key).slice(0, 12)}`, mirror.key],
      ["facets", `${totals.facets} · ${kinds.shell_floor}F ${kinds.shell_wall}W ${kinds.shell_ceiling}C ${kinds.furniture}f`, "floor, wall, ceiling, furniture"],
      ["reflector tri", `${totals.reflector_triangles}`],
      ["reflecting", `${(totals.reflector_area_m2 || 0).toFixed(1)} of ${(totals.model_area_m2 || 0).toFixed(1)} m²`],
      ["diffuse", `${(totals.diffuse_area_m2 || 0).toFixed(1)} m²`],
      ["occluder tri", `${totals.occluder_triangles} of ${totals.model_triangles}`],
      ["kept open", `${(totals.labels_kept_open || []).length}`, (totals.labels_kept_open || []).join(", ")],
      ["over bound", `${(totals.labels_over_bound || []).join(", ") || "none"}`],
      ["rules", `${mirror.rules ? mirror.rules.reflector_area_m2 : "–"} m² · ${mirror.rules ? mirror.rules.edge_m : "–"} m edge`],
    ]);
    const perLabel = (mirror.census && mirror.census.labels) || {};
    if (colourBy === "kind") {
      legend(
        Object.entries(KIND_COLOURS).map(([kind, colour]) => ({
          name: kind.replace("_", " "),
          colour: hex(colour),
          cells: [`${kinds[kind] ?? 0}`],
        })),
        ["facet kind", "facets"]
      );
    } else {
      legend(
        mirror.labels.map((name, i) => {
          const row = perLabel[name] || {};
          return {
            key: name,
            name,
            colour: labelHex(i),
            cells: [`${row.facets ?? 0}`, fixed(row.reflector_area_m2, 1), `${row.triangles_after ?? "–"}`],
            title: `${name}: ${row.area_m2} m² in the model`,
          };
        }),
        ["label", "fac", "refl m²", "occ tri"]
      );
    }
    const detail = $("#audit-detail");
    const parts = [
      el("div", {
        className: "note",
        textContent:
          "Red stripes: the back of a one-sided facet (sides 2), where the engine reflects nothing. " +
          "Occluders are drawn two-sided whatever their sidedness.",
      }),
    ];
    if (picked) {
      const i = mirror.labels.indexOf(picked.label);
      const row = perLabel[picked.label] || {};
      const m = mirror.materials;
      const columns = [{ name: "cat", title: "catalogue: scene.npz", absorption: m.absorption[i], scattering: m.scattering[i] }];
      for (const variant of Object.values(mirror.variants || {})) {
        const tag = `${variant.variant}${Object.keys(mirror.variants).length > 2 ? ` ${variant.source}` : ""}`;
        columns.push({
          name: `${tag} rays`,
          title: `${variant.report}: ${variant.parameters.note}`,
          absorption: variant.rays.absorption[i],
          scattering: variant.rays.scattering[i],
        });
        columns.push({
          name: `${tag} img`,
          title: `${variant.report}: the image sources' gains`,
          absorption: variant.images.absorption[i],
          scattering: variant.images.scattering[i],
        });
      }
      parts.push(
        el("p", { className: "label", textContent: picked.label }),
        el("div", { className: "note", textContent: m.source[i] })
      );
      if (picked.facet !== undefined && picked.facet !== null) {
        const f = mirror.facets[picked.facet];
        parts.push(
          el(
            "div",
            { className: "note amber" },
            `facet ${picked.facet} · ${f.kind} · ${f.area_m2.toFixed(3)} m² · ${f.triangles} tri · ` +
              `${f.sides === 3 ? "both sides" : "front side"} · n ${f.normal.map((v) => v.toFixed(2)).join(" ")}`
          )
        );
      } else if (picked.layer === "occluders") {
        parts.push(el("div", { className: "note amber", textContent: "an occluder triangle: blocks and scatters, makes no image" }));
      }
      parts.push(
        el(
          "dl",
          { className: "facts" },
          ...[
            ["model area", `${fixed(row.area_m2, 3)} m²`],
            ["reflecting", `${fixed(row.reflector_area_m2, 3)} m² in ${row.facets ?? 0} facets`],
            ["diffuse", `${fixed(row.diffuse_area_m2, 3)} m²`],
            ["occluder tri", `${row.triangles_before} → ${row.triangles_after}`],
            ["strayed", `${fixed(row.distance_m, 4)} m${row.over_bound ? " · over bound" : ""}${row.kept_open ? " · kept open" : ""}`],
          ].flatMap(([a, b]) => [el("dt", { textContent: a }), el("dd", { textContent: b })])
        ),
        el("div", { className: "note", textContent: "absorption per octave band, and scattering" }),
        bandTable(m.bands_hz, columns)
      );
    }
    if (cell && cell.paths) {
      const orders = [];
      for (const [order] of cell.paths) orders[order] = (orders[order] || 0) + 1;
      parts.push(
        el("p", { className: "label", textContent: `Paths at cell ${cell.position}` }),
        el(
          "div",
          { className: "orders" },
          ...orders.map((n, order) =>
            n ? el("span", {}, el("i", { className: "swatch", style: `background:${hex(ORDER_COLOURS[Math.min(order, ORDER_COLOURS.length - 1)])}` }), `order ${order}: ${n}`) : null
          )
        ),
        el(
          "div",
          { className: "note", textContent: "one file for every mirror field: the paths do not depend on the materials" }
        )
      );
    }
    detail.replaceChildren(...parts);
    $("#audit-note").textContent = "";
  }

  function render() {
    root.dataset.mode = mode;
    for (const node of [$("#audit-facts"), $("#audit-legend"), $("#audit-detail")]) node.replaceChildren();
    problems([]);
    $("#audit-note").textContent = "";
    if (mode === "acoustic") renderWave();
    else if (mode === "mirror") renderMirror();
    else $("#audit-lead").textContent = "Colour: the furnished model as it looks. Wave and Mirror show what each solver read.";
  }

  return {
    setMode(next) {
      if (next === mode) return;
      mode = next;
      // A label picked in one view is looked for by name in the other.
      pick(picked && labels().includes(picked.label) ? { label: picked.label } : null);
    },
    setWave(record, fmax) {
      wave = record ? { ...record, fmax } : null;
      if (mode === "acoustic") pick(picked && labels().includes(picked.label) ? { label: picked.label } : null);
    },
    setMirror(check) {
      mirror = check;
      if (mode === "mirror") pick(null);
    },
    setCell(id, position, paths) {
      cell = paths ? { id, position, paths } : null;
      if (mode === "mirror") render();
    },
    pick,
    labels,
  };
}
