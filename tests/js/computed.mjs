/** What a pack was computed on, as the page turns it into pictures, checked without a browser.
 *
 * Run by `tests/test_computed_js.py`, which hands in the page's `app` folder.
 * Prints one JSON object.
 */
const app = process.argv[2];
const data = await import(`${app}/scene/computed-data.js`);

const out = {};

// --- an answer of arrays end to end is read back in place ------------------------
{
  // As `computed_api.packed` lays them: a byte array, then floats padded to four bytes.
  const buffer = new ArrayBuffer(3 + 1 + 8 + 4);
  new Uint8Array(buffer, 0, 3).set([1, 0, 5]);
  new Float32Array(buffer, 4, 2).set([1.5, -2.25]);
  new Int16Array(buffer, 12, 2).set([-1, 7]);
  const arrays = data.unpack(buffer, {
    arrays: [
      { name: "classes", dtype: "|u1", shape: [1, 3], offset: 0 },
      { name: "corners", dtype: "<f4", shape: [2], offset: 4 },
      { name: "material", dtype: "<i2", shape: [2], offset: 12 },
    ],
  });
  out.unpack = { classes: [...arrays.classes], corners: [...arrays.corners], material: [...arrays.material] };
  try {
    data.unpack(buffer, { arrays: [{ name: "x", dtype: "<c8", shape: [1], offset: 0 }] });
    out.unpack.refused = false;
  } catch {
    out.unpack.refused = true;
  }
}

// --- a slice: a byte a node, a pixel a node, a colour a material -------------------
{
  const classes = new Uint8Array([data.CLASS.notReached, data.CLASS.air, data.CLASS.rigid, data.CLASS.firstMaterial + 4, data.CLASS.firstMaterial + 4, data.CLASS.firstMaterial + 5]);
  const pixels = data.sliceRgba(classes);
  const pixel = (k) => [...pixels.subarray(4 * k, 4 * k + 3)];
  out.slice = {
    length: pixels.length,
    notReached: pixel(0),
    air: pixel(1),
    opaque: [...pixels].filter((_, k) => k % 4 === 3).every((alpha) => alpha === 255),
    sameMaterialSameColour: pixel(3).join() === pixel(4).join(),
    otherMaterialOtherColour: pixel(4).join() !== pixel(5).join(),
    isTheWallsColour: pixel(3).join() === data.materialColour(4).join(),
    census: [...data.sliceCensus(classes).entries()].sort((a, b) => a[0] - b[0]),
    rect: data.sliceRect({ origin_m: [-2, 10], step_m: 0.5 }, [4, 6]),
  };
  const diff = data.sliceRgba(new Uint8Array([data.DIFF.onlyA, data.DIFF.onlyB, data.DIFF.air]), "diff");
  out.slice.diff = [[...diff.subarray(0, 3)], [...diff.subarray(4, 7)], [...diff.subarray(8, 11)]];
  out.slice.diffNames = [data.COLOURS.onlyA, data.COLOURS.onlyB, data.COLOURS.air];
}

// --- merged squares: two triangles each, their area what they cover -------------------
{
  // A rectangle 2 by 3 in the plane x = 1, and a unit square in the plane y = 0.
  const corners = new Float32Array([1, 0, 0, 1, 2, 0, 1, 2, 3, 1, 0, 3, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]);
  const mesh = data.quadMesh(corners, new Int16Array([-1, 2]), data.materialColour);
  out.quads = {
    index: [...mesh.index],
    area: data.quadArea(corners),
    rigid: [...mesh.colours.subarray(0, 3)].map((v) => Math.round(v * 255)),
    everyCornerOneColour: [0, 1, 2, 3].every((k) => mesh.colours[12 + 3 * k] === mesh.colours[12]),
    triangleColours: [...data.triangleColours(new Int32Array([0, 1]), (id) => (id ? [255, 0, 0] : [0, 0, 255]))],
  };
  const catalogue = {
    facets: [
      { label: 0, two_sided: false, sheet_layer: false },
      { label: 1, two_sided: true, sheet_layer: true },
      { label: 1, two_sided: true, sheet_layer: false },
    ],
    absorption: [[0.02, 0.05], [0.9, 0.95]],
  };
  const sheets = data.facetColourer(catalogue, "sheets");
  out.facets = {
    sheetMarked: sheets(1).join() === data.COLOURS.sheet.join(),
    twoSidedApart: sheets(2).join() !== sheets(1).join() && sheets(2).join() !== sheets(0).join(),
    byMaterial: data.facetColourer(catalogue, "material")(1).join() === data.materialColour(1).join(),
    hardLighterThanSoft:
      data.facetColourer(catalogue, "absorption", 1)(0).reduce((a, b) => a + b) > data.facetColourer(catalogue, "absorption", 1)(1).reduce((a, b) => a + b),
  };
}

// --- rays and paths as segments ------------------------------------------------------
{
  // Two rays: three vertices and two; two bands.
  const rays = {
    offsets: new Int32Array([0, 3, 5]),
    points: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2]),
    energy: new Float32Array([1, 1, 0.5, 0.1, 0.25, 0.01, 1, 1, 1, 1]),
  };
  const held = data.raySegments(rays, 2, 1);
  out.rays = {
    legs: held.legs,
    positions: [...held.positions],
    firstLegFull: [...held.colours.subarray(0, 3)].map((v) => Math.round(v * 255)).join() === data.energyColour(1).join(),
    secondLegDimmer: [...held.colours.subarray(6, 9)].map((v) => Math.round(v * 255)).join() === data.energyColour(0.1).join(),
    dimmerIsDarker: data.energyColour(1e-5).reduce((a, b) => a + b) < data.energyColour(1).reduce((a, b) => a + b),
  };
  const rows = [
    { order: 0, points: [[0, 0, 0], [2, 0, 0]], facets: [] },
    { order: 2, points: [[0, 0, 0], [1, 1, 0], [1.5, 0, 0], [2, 0, 0]], facets: [7, 3] },
    { order: 1, points: null, facets: null },
    { order: 1, points: [[0, 0, 0], [1, -1, 0], [2, 0, 0]], facets: [7] },
  ];
  const paths = data.pathSegments(rows);
  out.paths = {
    segments: paths.positions.length / 6,
    orders: paths.orders,
    corners: paths.corners,
    missing: paths.missing,
    hit: data.facetsHit(rows),
  };
  out.decay = data.decayPoints([0, 1, 0.1, 1e-12], 300, 80);
}

// --- the words ---------------------------------------------------------------------
{
  out.words = {
    mm: data.mm(0.0218),
    key: data.shortKey("a6422ab46225c2ea0bdb6518a99a9e13"),
    opening: data.openingLine({ centre_m: [2, 1.45], across: "x", a: { width_nodes: 18, width_m: 0.9 }, b: null }),
    diff: data.diffLines({
      air_a_m3: 28.8,
      air_b_m3: 14.2,
      air_only_in_a_m3: 14.6,
      air_only_in_b_m3: 0,
      largest_wall_displacement_m: 0.04,
      largest_wall_displacement_at_m: [0.04, 0.04, 2.96],
      wall_search_m: 0.25,
      walls: { a_to_b: { without_m2: 29.24 }, b_to_a: { without_m2: 0.16 } },
      regions_air_only_in_a: [{ volume_m3: 14.4, centre_m: [3, 1.2, 1.5] }],
      regions_air_only_in_b: [],
      method: "B read at A's nodes",
      resolution_m: 0.05,
    }),
  };
}

console.log(JSON.stringify(out));
