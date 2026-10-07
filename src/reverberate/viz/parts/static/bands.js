/** Numbers a third octave, drawn: one line a series over a logarithmic axis.
 *
 *   drawBands(element, { bandsHz, series: [{ label, values, colour }], rangeDb: 12, markHz: 1000 })
 *
 * A value that is `null` is a band that holds nothing: the line stops there.
 * The vertical scale is `rangeDb` either side of zero and does not follow the
 * data, so that two charts are read against each other; a value past it is
 * drawn at the edge with a tick. `markHz` draws a vertical line, the
 * crossover's.
 */
const SVG = "http://www.w3.org/2000/svg";
const make = (tag, attributes, text) => {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
};

export function drawBands(element, { bandsHz, series, rangeDb = 12, markHz = null, unit = "dB" }) {
  const width = 640;
  const height = 220;
  const left = 44;
  const right = 12;
  const top = 12;
  const bottom = 30;
  const low = bandsHz[0] / 2 ** (1 / 6);
  const high = bandsHz[bandsHz.length - 1] * 2 ** (1 / 6);
  const x = (f) => left + ((width - left - right) * Math.log(f / low)) / Math.log(high / low);
  const y = (db) => top + ((height - top - bottom) * (rangeDb - Math.max(-rangeDb, Math.min(rangeDb, db)))) / (2 * rangeDb);
  const svg = make("svg", { viewBox: `0 0 ${width} ${height}`, class: "p-bands", role: "img" });
  for (let db = -rangeDb; db <= rangeDb; db += rangeDb / 2) {
    svg.append(make("line", { x1: left, x2: width - right, y1: y(db), y2: y(db), class: db === 0 ? "p-zero" : "p-grid" }));
    svg.append(make("text", { x: left - 6, y: y(db) + 4, "text-anchor": "end", class: "p-tick" }, `${db > 0 ? "+" : ""}${db}`));
  }
  for (const f of [125, 250, 500, 1000, 2000, 4000, 8000, 16000]) {
    if (f < low || f > high) continue;
    svg.append(make("line", { x1: x(f), x2: x(f), y1: top, y2: height - bottom, class: "p-grid" }));
    svg.append(make("text", { x: x(f), y: height - bottom + 14, "text-anchor": "middle", class: "p-tick" }, f >= 1000 ? `${f / 1000} k` : `${f}`));
  }
  svg.append(make("text", { x: width - right, y: height - 2, "text-anchor": "end", class: "p-tick" }, "Hz, by third octave"));
  svg.append(make("text", { x: 4, y: top + 4, class: "p-tick" }, unit));
  if (markHz) {
    svg.append(make("line", { x1: x(markHz), x2: x(markHz), y1: top, y2: height - bottom, class: "p-mark" }));
  }
  for (const { values, colour } of series) {
    let path = "";
    let pen = false;
    values.forEach((value, index) => {
      if (value === null || value === undefined) {
        pen = false;
        return;
      }
      path += `${pen ? "L" : "M"}${x(bandsHz[index]).toFixed(1)} ${y(value).toFixed(1)}`;
      pen = true;
      svg.append(make("circle", { cx: x(bandsHz[index]), cy: y(value), r: Math.abs(value) > rangeDb ? 3.5 : 2, fill: colour }));
    });
    svg.append(make("path", { d: path, fill: "none", stroke: colour, "stroke-width": 1.6 }));
  }
  const key = document.createElement("div");
  key.className = "p-bands-key";
  for (const { label, colour } of series) {
    const item = document.createElement("span");
    const chip = document.createElement("i");
    chip.style.background = colour;
    item.append(chip, label);
    key.append(item);
  }
  element.textContent = "";
  element.append(svg, key);
}
