/** The Audit section of the left panel, shown with the mirror solver's view.
 *
 * The derived scene's census, a legend of its material labels (or facet
 * kinds), the absorption per band and the scattering of the label picked in
 * the legend, and the paths the image sources found at the listener's cell.
 */
import { materialColour } from "./grid.js";
import { KIND_COLOURS, ORDER_COLOURS } from "./mirror.js";

const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
};
const hex = (n) => `#${n.toString(16).padStart(6, "0")}`;
const fixed = (v, digits = 3) => (v === null || v === undefined ? "–" : Number(v).toFixed(digits));
const bandName = (hz) => (hz >= 1000 ? `${Number((hz / 1000).toFixed(2))}k` : `${hz}`);

export function createMirrorAudit(root, THREE, { onSwitch, onColourBy }) {
  const $ = (s) => root.querySelector(s);
  const rgb = new THREE.Color();
  let scene = null; // layers.json
  let picked = null; // a label
  let colourBy = "label";
  let cell = null; // { position, paths }

  for (const box of root.querySelectorAll("[data-mirror]")) {
    box.addEventListener("change", () => onSwitch(box.dataset.mirror, box.checked));
  }
  $("#audit-colour-by").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    colourBy = button.dataset.by;
    for (const b of $("#audit-colour-by").querySelectorAll("button")) b.classList.toggle("on", b === button);
    onColourBy(colourBy);
    render();
  });

  function facts(rows) {
    $("#audit-facts").replaceChildren(
      ...rows.flatMap(([name, value, title]) => [
        el("dt", { textContent: name }),
        el("dd", { textContent: value, title: title || "" }),
      ])
    );
  }

  function legend(rows, head) {
    $("#audit-legend").replaceChildren(
      el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { textContent: h })))),
      el(
        "tbody",
        {},
        ...rows.map((row) => {
          const tr = el(
            "tr",
            { className: `${row.key !== undefined ? "pick" : ""} ${row.key === picked ? "on" : ""}` },
            el("td", {}, el("i", { className: "swatch", style: `background:${row.colour}` }), row.name),
            ...row.cells.map((c) => el("td", { textContent: c }))
          );
          if (row.key !== undefined) {
            tr.addEventListener("click", () => {
              picked = picked === row.key ? null : row.key;
              render();
            });
          }
          return tr;
        })
      )
    );
  }

  function materials(index) {
    const m = scene.materials;
    return el(
      "table",
      { className: "bands" },
      el("thead", {}, el("tr", {}, el("th", { textContent: "Hz" }), el("th", { textContent: "α" }))),
      el(
        "tbody",
        {},
        ...m.bands_hz.map((hz, b) =>
          el("tr", {}, el("th", { textContent: bandName(hz) }), el("td", { textContent: fixed(m.absorption[index][b]) }))
        ),
        el(
          "tr",
          { className: "scatter" },
          el("th", { textContent: "s", title: "scattering coefficient, one for all bands" }),
          el("td", { textContent: fixed(m.scattering[index]) })
        )
      )
    );
  }

  function render() {
    if (!scene) {
      facts([["scene", "none for this simulation"]]);
      $("#audit-legend").replaceChildren();
      $("#audit-detail").replaceChildren();
      return;
    }
    const totals = (scene.census && scene.census.totals) || {};
    const kinds = totals.facets_by_kind || {};
    facts([
      ["scene", String(scene.key).slice(0, 12), scene.key],
      ["facets", `${totals.facets} · ${kinds.shell_floor}F ${kinds.shell_wall}W ${kinds.shell_ceiling}C ${kinds.furniture}f`, "floor, wall, ceiling, furniture"],
      ["reflecting", `${fixed(totals.reflector_area_m2, 1)} of ${fixed(totals.model_area_m2, 1)} m²`],
      ["diffuse", `${fixed(totals.diffuse_area_m2, 1)} m²`],
      ["occluder tri", `${totals.occluder_triangles} of ${totals.model_triangles}`],
    ]);
    const perLabel = (scene.census && scene.census.labels) || {};
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
        scene.labels.map((name, i) => ({
          key: name,
          name,
          colour: `#${materialColour(rgb, i).getHexString()}`,
          cells: [`${(perLabel[name] || {}).facets ?? 0}`, fixed((perLabel[name] || {}).reflector_area_m2, 1)],
        })),
        ["label", "fac", "refl m²"]
      );
    }
    const parts = [
      el("div", { className: "note", textContent: "Red stripes: the back of a one-sided facet, where nothing reflects." }),
    ];
    if (picked !== null) {
      const index = scene.labels.indexOf(picked);
      parts.push(
        el("p", { className: "label", textContent: picked }),
        el("div", { className: "note", textContent: scene.materials.source[index] }),
        materials(index)
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
            n
              ? el(
                  "span",
                  {},
                  el("i", { className: "swatch", style: `background:${hex(ORDER_COLOURS[Math.min(order, ORDER_COLOURS.length - 1)])}` }),
                  `order ${order}: ${n}`
                )
              : null
          )
        )
      );
    }
    $("#audit-detail").replaceChildren(...parts);
  }

  return {
    setScene(next) {
      scene = next;
      picked = null;
      render();
    },
    setCell(position, paths) {
      cell = paths ? { position, paths } : null;
      render();
    },
  };
}
