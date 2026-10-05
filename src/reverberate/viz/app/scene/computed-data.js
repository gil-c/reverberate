/** What the computation used, as numbers a page can draw: nothing here touches the page.
 *
 * `viz/computed_api.py` answers with arrays end to end and a header that
 * says what each is. This module reads them back (`unpack`) and turns them
 * into what three.js and a canvas take: a slice's bytes into pixels, merged
 * squares into triangles, rays and image paths into line segments. It holds
 * the colours too, so that a material is one colour in the walls, on the
 * slice and in the legend. `tests/js/computed.mjs` runs it under node.
 */

const TYPES = {
  "<f4": Float32Array,
  "<f8": Float64Array,
  "<i2": Int16Array,
  "<i4": Int32Array,
  "<i8": BigInt64Array,
  "|u1": Uint8Array,
  "|i1": Int8Array,
  "<u2": Uint16Array,
  "<u4": Uint32Array,
};

/** The arrays of a `packed` answer, by name, each a typed array over the buffer in place. */
export function unpack(buffer, spec) {
  const out = {};
  for (const { name, dtype, shape, offset } of spec.arrays) {
    const Type = TYPES[dtype];
    if (!Type) throw new Error(`an array of type ${dtype} is not read`);
    const count = shape.reduce((product, n) => product * n, 1);
    out[name] = new Type(buffer, offset, count);
  }
  return out;
}

// --- colours --------------------------------------------------------------------
export const CLASS = { notReached: 0, air: 1, rigid: 2, firstMaterial: 3 };
export const DIFF = { closed: 0, air: 1, onlyA: 2, onlyB: 3, outside: 4 };
export const COLOURS = {
  notReached: [38, 40, 46],
  air: [222, 226, 230],
  rigid: [250, 250, 250],
  onlyA: [255, 122, 26],
  onlyB: [38, 150, 255],
  outside: [18, 18, 20],
  wallA: [255, 190, 120],
  wallB: [130, 200, 255],
  sheet: [255, 60, 160],
};

function hsl(h, s, l) {
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}

/** One colour a material, the same wherever it is drawn; `-1` is a rigid node. */
export function materialColour(index) {
  if (index < 0) return COLOURS.rigid;
  // The golden angle keeps neighbours in the list apart on the wheel.
  return hsl((index * 137.508) % 360, 0.62, index % 2 ? 0.46 : 0.6);
}

/** Absorption 0 to 1 as a colour: a hard surface light and cold, a soft one dark and warm. */
export function absorptionColour(alpha) {
  const a = Math.max(0, Math.min(1, alpha));
  return hsl(210 - 190 * a, 0.75, 0.72 - 0.34 * a);
}

/** What is left of a ray's energy, on a scale of 60 dB, as a colour. */
export function energyColour(share) {
  const db = 10 * Math.log10(Math.max(share, 1e-12));
  const t = Math.max(0, Math.min(1, 1 + db / 60));
  return hsl(260 - 220 * t, 0.9, 0.25 + 0.4 * t);
}

function classColour(value) {
  if (value === CLASS.notReached) return COLOURS.notReached;
  if (value === CLASS.air) return COLOURS.air;
  if (value === CLASS.rigid) return COLOURS.rigid;
  return materialColour(value - CLASS.firstMaterial);
}

function diffColour(value) {
  if (value === DIFF.closed) return COLOURS.notReached;
  if (value === DIFF.air) return COLOURS.air;
  if (value === DIFF.onlyA) return COLOURS.onlyA;
  if (value === DIFF.onlyB) return COLOURS.onlyB;
  return COLOURS.outside;
}

/** A slice's bytes as RGBA pixels, row for row; `kind` is `grid` or `diff`. */
export function sliceRgba(classes, kind = "grid") {
  const out = new Uint8ClampedArray(4 * classes.length);
  const colourOf = kind === "diff" ? diffColour : classColour;
  const seen = new Map();
  for (let i = 0; i < classes.length; i++) {
    let colour = seen.get(classes[i]);
    if (!colour) {
      colour = colourOf(classes[i]);
      seen.set(classes[i], colour);
    }
    out[4 * i] = colour[0];
    out[4 * i + 1] = colour[1];
    out[4 * i + 2] = colour[2];
    out[4 * i + 3] = 255;
  }
  return out;
}

/** How many nodes of a slice are of each class. */
export function sliceCensus(classes) {
  const counts = new Map();
  for (let i = 0; i < classes.length; i++) counts.set(classes[i], (counts.get(classes[i]) || 0) + 1);
  return counts;
}

/** Where a slice lies: a pixel is a node's own cell, so the picture overhangs
 *  the first and the last node by half a step. `shape` is `[rows z, columns x]`. */
export function sliceRect({ origin_m: origin, step_m: step }, shape) {
  return {
    x0: origin[0] - step / 2,
    z0: origin[1] - step / 2,
    x1: origin[0] + step * (shape[1] - 0.5),
    z1: origin[1] + step * (shape[0] - 0.5),
  };
}

// --- surfaces -------------------------------------------------------------------
/** Merged squares `[quad, 4, 3]` as two triangles each, coloured by `colourOf(value)`. */
export function quadMesh(corners, values, colourOf) {
  const quads = values.length;
  const colours = new Float32Array(12 * quads);
  const index = new Uint32Array(6 * quads);
  for (let q = 0; q < quads; q++) {
    const [r, g, b] = colourOf(values[q]);
    for (let k = 0; k < 4; k++) {
      colours[12 * q + 3 * k] = r / 255;
      colours[12 * q + 3 * k + 1] = g / 255;
      colours[12 * q + 3 * k + 2] = b / 255;
    }
    index.set([4 * q, 4 * q + 1, 4 * q + 2, 4 * q, 4 * q + 2, 4 * q + 3], 6 * q);
  }
  return { positions: corners, colours, index };
}

/** The area of merged squares, m2: what the squares they came from add up to. */
export function quadArea(corners) {
  let total = 0;
  for (let q = 0; q < corners.length / 12; q++) {
    const at = 12 * q;
    const u = [corners[at + 3] - corners[at], corners[at + 4] - corners[at + 1], corners[at + 5] - corners[at + 2]];
    const v = [corners[at + 9] - corners[at], corners[at + 10] - corners[at + 1], corners[at + 11] - corners[at + 2]];
    total += Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
  }
  return total;
}

/** A colour per corner of triangles that each carry one id. */
export function triangleColours(ids, colourOf) {
  const out = new Float32Array(9 * ids.length);
  for (let t = 0; t < ids.length; t++) {
    const [r, g, b] = colourOf(ids[t]);
    for (let k = 0; k < 3; k++) {
      out[9 * t + 3 * k] = r / 255;
      out[9 * t + 3 * k + 1] = g / 255;
      out[9 * t + 3 * k + 2] = b / 255;
    }
  }
  return out;
}

/** How a facet is coloured: by its material, by its absorption in a band, or
 *  marked where it is one layer of a sheet the model holds twice. */
export function facetColourer(catalogue, mode, band = 3) {
  return (facet) => {
    const told = catalogue.facets[facet];
    if (!told) return COLOURS.rigid;
    if (mode === "sheets") return told.sheet_layer ? COLOURS.sheet : told.two_sided ? [255, 214, 90] : [120, 126, 136];
    if (mode === "absorption") return absorptionColour(catalogue.absorption[told.label][band]);
    return materialColour(told.label);
  };
}

// --- lines ----------------------------------------------------------------------
/** Rays as segments: each leg in the colour of what the ray still carries in `band`. */
export function raySegments({ offsets, points, energy }, bands, band) {
  const legs = points.length / 3 - (offsets.length - 1);
  const positions = new Float32Array(6 * Math.max(legs, 0));
  const colours = new Float32Array(6 * Math.max(legs, 0));
  let at = 0;
  for (let ray = 0; ray + 1 < offsets.length; ray++) {
    for (let v = offsets[ray]; v + 1 < offsets[ray + 1]; v++) {
      positions.set(points.subarray(3 * v, 3 * v + 6), 6 * at);
      const [r, g, b] = energyColour(energy[bands * v + band]);
      colours.set([r / 255, g / 255, b / 255, r / 255, g / 255, b / 255], 6 * at);
      at += 1;
    }
  }
  return { positions, colours, legs: at };
}

/** A step's image paths as segments through their corners, and the rows without corners. */
export function pathSegments(rows) {
  const positions = [];
  const orders = [];
  const corners = [];
  let missing = 0;
  for (const row of rows) {
    if (!row.points) {
      missing += 1;
      continue;
    }
    for (let k = 0; k + 1 < row.points.length; k++) {
      positions.push(...row.points[k], ...row.points[k + 1]);
      orders.push(row.order);
    }
    for (let k = 1; k + 1 < row.points.length; k++) corners.push({ at: row.points[k], facet: row.facets[k - 1] });
  }
  return { positions: new Float32Array(positions), orders, corners, missing };
}

/** The facets a step's paths bounce on, each with how many paths use it. */
export function facetsHit(rows) {
  const counts = new Map();
  for (const row of rows) for (const facet of row.facets || []) counts.set(facet, (counts.get(facet) || 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

/** A histogram's energy in one band as points of a plot `width` by `height`, in dB under its peak. */
export function decayPoints(energy, width, height, floorDb = -80) {
  let peak = 0;
  for (const value of energy) peak = Math.max(peak, value);
  const out = [];
  if (peak <= 0) return out;
  for (let i = 0; i < energy.length; i++) {
    if (energy[i] <= 0) continue;
    const db = Math.max(floorDb, 10 * Math.log10(energy[i] / peak));
    out.push([(i / Math.max(energy.length - 1, 1)) * width, (db / floorDb) * height]);
  }
  return out;
}

// --- words ----------------------------------------------------------------------
export const mm = (metres) => `${(metres * 1000).toFixed(1)} mm`;
export const m3 = (value) => `${value.toFixed(value < 10 ? 2 : 1)} m³`;
export const shortKey = (key) => (key && key.length > 14 ? `${key.slice(0, 12)}…` : key || "none");

/** An opening that changed, in a line. */
export function openingLine(row) {
  const side = (held) => (held ? `${held.width_nodes} links (${mm(held.width_m)})` : "none");
  const [x, z] = row.centre_m;
  return `at x ${x.toFixed(2)} z ${z.toFixed(2)}, across ${row.across}: A ${side(row.a)}, B ${side(row.b)}`;
}

/** The numbers of a comparison, as the lines the page shows. */
export function diffLines(numbers) {
  const lines = [
    `air: A ${m3(numbers.air_a_m3)}, B ${m3(numbers.air_b_m3)}`,
    `air only in A ${m3(numbers.air_only_in_a_m3)}, only in B ${m3(numbers.air_only_in_b_m3)}`,
    `largest wall displacement ${mm(numbers.largest_wall_displacement_m)}` +
      (numbers.largest_wall_displacement_at_m ? ` at ${numbers.largest_wall_displacement_at_m.join(", ")}` : ""),
    `walls with none of the other within ${mm(numbers.wall_search_m)}: A ${numbers.walls.a_to_b.without_m2.toFixed(2)} m², B ${numbers.walls.b_to_a.without_m2.toFixed(2)} m²`,
  ];
  for (const [name, regions] of [["A", numbers.regions_air_only_in_a], ["B", numbers.regions_air_only_in_b]]) {
    for (const region of regions.slice(0, 4)) {
      lines.push(`air only in ${name}: ${m3(region.volume_m3)} about ${region.centre_m.join(", ")}`);
    }
  }
  lines.push(`method: ${numbers.method}; resolution ${mm(numbers.resolution_m)}`);
  return lines;
}
