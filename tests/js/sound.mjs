/** A scene's sound, checked without a browser: the clock, the fetch plan, the decode.
 *
 * Run by `tests/test_scene_sound_js.py`, which hands in the page's `app`
 * folder. Prints one JSON object.
 */
const app = process.argv[2];
const { createTransport } = await import(`${app}/scene/transport.js`);
const { createSoundClock, createFetchPlan, chunkSpans, commonRanges, meterFill, arrivalGlyphs, shellPoints } =
  await import(`${app}/scene/sound-plan.js`);
const { DEFAULT_LEVEL_DB, LEVEL_RANGE_DB, FULL_SCALE_SPL_DB, levelGain, peakOf, createClipMeter } = await import(
  `${app}/scene/sound-plan.js`
);
const { BLOCK, QUANTUM, BINS, headMatrix3, filterSpectra, rotateSpectra, createStreamDecoder } = await import(
  `${app}/scene/sound-decode.js`
);
const { createDecoder } = await import(`${app}/audio/decode.js`);
const { headMatrix, channelCount } = await import(`${app}/audio/sh.js`);

const out = {};
const RATE = 48000;

// A small generator, so the harness draws the same numbers every time.
let seed = 12345;
const random = () => {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 2 ** 32 - 0.5;
};

// --- the clock: scene time is the samples that sounded -------------------------
{
  let context = 100;
  const clock = createSoundClock({ sampleRate: RATE, contextTime: () => context, ahead: 0.03 });
  const transport = createTransport({ now: () => 0 });
  transport.load(60);
  transport.setClock(clock.now.bind(clock));
  const seen = {};

  transport.seek(10);
  transport.play();
  clock.hold();
  context += 0.4; // the first chunk is being rendered: nothing sounds
  seen.heldWhileWaiting = transport.time();

  // The audio thread takes the stream at its count 5000, then plays.
  clock.begin(5000);
  seen.atBegin = transport.time();
  clock.report(5000 + 4800, context, true);
  seen.afterTenthOfASecond = transport.time();
  context += 0.004;
  seen.betweenReports = transport.time();
  context += 10; // a report that never comes does not run the clock away
  seen.aheadIsBounded = transport.time();

  // Starved: the count stands still, whatever the context's time does.
  clock.report(5000 + 9600, context, false);
  const stalled = transport.time();
  context += 3;
  seen.stalled = [stalled, transport.time()];

  // A seek while playing: the old stream's samples do not move the new time.
  clock.report(5000 + 14400, context, true);
  transport.seek(40);
  clock.hold();
  clock.report(5000 + 19200, context, true); // the old stream, still sounding for a block
  seen.afterSeek = transport.time();
  clock.begin(5000 + 19712);
  clock.report(5000 + 19712 + 24000, context, false);
  seen.halfASecondOn = transport.time();

  // Never backwards: a report behind the clock's own run does not pull it back.
  const before = clock.now();
  clock.report(5000 + 19712 + 24000 - 100, context, false);
  seen.backwards = clock.now() - before;
  out.clock = seen;
}

// --- which chunks are asked for ------------------------------------------------
{
  const plan = createFetchPlan({ chunkSamples: 24000, chunks: 10, ahead: 4, parallel: 2 });
  const seen = {};
  const gen = plan.restart();
  seen.first = plan.next(30000); // in chunk 1
  seen.none = plan.next(30000); // both are flying
  seen.mayGoEmpty = plan.mayGo(30000);
  seen.accepted = plan.arrived(gen, 1);
  seen.mayGo = plan.mayGo(30000);
  seen.second = plan.next(30000);
  plan.arrived(gen, 2);
  plan.arrived(gen, 3);
  seen.third = plan.next(30000);
  seen.buffered = plan.buffered(30000);
  seen.starvedNeedsMore = [plan.mayGo(30000, { starved: true, resume: 4 }), plan.mayGo(30000, { starved: true, resume: 3 })];
  plan.failed(gen, 4);
  seen.retried = plan.next(30000);
  // A solo: a new generation, and the old one's answers are dropped.
  const later = plan.restart();
  seen.stale = plan.arrived(gen, 4);
  seen.again = plan.next(30000);
  seen.generations = [gen, later];
  // The scene's end is enough to go on.
  const end = createFetchPlan({ chunkSamples: 24000, chunks: 10 });
  const last = end.restart();
  end.next(9.5 * 24000);
  end.arrived(last, 9);
  seen.endGoes = end.mayGo(9.5 * 24000, { starved: true, resume: 4 });
  seen.pastEnd = end.next(9.5 * 24000);
  out.plan = seen;
}

// --- what the timeline is told ---------------------------------------------------
out.spans = {
  seconds: chunkSpans([[0, 3], [5, 7]], 24000, RATE, 3.2),
  common: commonRanges([[[0, 4], [6, 9]], [[2, 7]], [[0, 8]]]),
  none: commonRanges([]),
  meter: [meterFill(-200, -20), meterFill(-70, -20), meterFill(-45, -20), meterFill(-20, -20), meterFill(null, -20)],
};

// --- the decode: the same as the page's decode of a field ------------------------
const order = 2;
const channels = channelCount(order);
const taps = 512;
const filters = new Float32Array(2 * channels * taps);
for (let i = 0; i < filters.length; i++) {
  // Centred and decaying, as a designed decoder is.
  const tap = i % taps;
  filters[i] = random() * Math.exp(-Math.abs(tap - 256) / 40);
}
const base = filterSpectra({ channels, taps, filters });
const frames = 8 * BLOCK;

const signal = () => {
  const data = new Float32Array(frames * channels);
  for (let i = 0; i < data.length; i++) data[i] = random();
  return data;
};
const deinterleave = (data) =>
  Array.from({ length: channels }, (_, c) => Float32Array.from({ length: frames }, (_, n) => data[n * channels + c]));

/** The page's own decode of a field, at a head: the two ears, from sample zero. */
function reference(data, head) {
  const decoder = createDecoder();
  decoder.setDecoder({ order, channels, taps, filters });
  decoder.keep({ key: "x", channels: deinterleave(data) });
  const ears = decoder.renderEarly({ key: "x", head });
  // `renderEarly` cuts the decoder's lead: put it back.
  return ears.map((ear) => {
    const whole = new Float32Array(frames);
    for (let n = decoder.lead; n < frames; n++) whole[n] = ear[n - decoder.lead];
    return { whole, lead: decoder.lead };
  });
}

/** Run a decoder for `quanta` render quanta; what it wrote, and when. */
function run(core, quanta, each = () => {}) {
  const left = [];
  const right = [];
  const wrote = [];
  for (let q = 0; q < quanta; q++) {
    each(q);
    const l = new Float32Array(QUANTUM);
    const r = new Float32Array(QUANTUM);
    wrote.push(core.process(l, r));
    left.push(...l);
    right.push(...r);
  }
  return { left, right, wrote };
}

const worst = (got, want, from, to, shift = 0) => {
  let error = 0;
  let peak = 0;
  for (let n = from; n < to; n++) {
    error = Math.max(error, Math.abs(got[n + shift] - want[n]));
    peak = Math.max(peak, Math.abs(want[n]));
  }
  return error / peak;
};

const PREROLL = BLOCK; // four quanta of silence while the first block is made
const head = headMatrix3(0.7, -0.2, 0.15);
const x = signal();

{
  const want = reference(x, head);
  const core = createStreamDecoder({ channels });
  core.setTotal(frames);
  core.setFilters(rotateSpectra(base, order, head));
  core.start(0, 1);
  const idle = run(core, 3);
  core.push({ gen: 1, start: 0, frames, data: x, go: true });
  const made = run(core, (frames + PREROLL) / QUANTUM);
  const lead = want[0].lead;
  out.decode = {
    waitsForGo: idle.wrote.some(Boolean),
    preroll: made.wrote.indexOf(true) * QUANTUM,
    // After the faded first block, sample for sample the page's decode.
    left: worst(made.left, want[0].whole, Math.max(BLOCK, lead), frames, PREROLL),
    right: worst(made.right, want[1].whole, Math.max(BLOCK, lead), frames, PREROLL),
    played: core.state().played,
    ended: run(core, 8).wrote.some(Boolean),
    endState: core.state().ended,
  };

  // The same samples in three chunks, cut off the block's grid.
  const cuts = [0, 700, 1900, frames];
  const pieces = createStreamDecoder({ channels });
  pieces.setTotal(frames);
  pieces.setFilters(rotateSpectra(base, order, head));
  pieces.start(0, 1);
  for (let i = 0; i + 1 < cuts.length; i++) {
    pieces.push({
      gen: 1,
      start: cuts[i],
      frames: cuts[i + 1] - cuts[i],
      data: x.slice(cuts[i] * channels, cuts[i + 1] * channels),
      go: true,
    });
  }
  const cut = run(pieces, (frames + PREROLL) / QUANTUM);
  out.decode.chunking = Math.max(...cut.left.map((value, i) => Math.abs(value - made.left[i])));
}

// --- running dry: the count stops, and nothing is lost or said twice --------------
{
  const want = reference(x, head);
  const core = createStreamDecoder({ channels });
  core.setTotal(frames);
  core.setFilters(rotateSpectra(base, order, head));
  core.start(0, 1);
  const halfway = 3 * BLOCK;
  core.push({ gen: 1, start: 0, frames: halfway, data: x.slice(0, halfway * channels), go: true });
  const first = run(core, 40);
  const dry = core.state();
  // The rest arrives, but not the word to go: still silent.
  core.push({ gen: 1, start: halfway, frames: frames - halfway, data: x.slice(halfway * channels), go: false });
  const held = run(core, 8);
  core.push({ gen: 1, start: 0, frames: 0, data: new Float32Array(0), go: true });
  const rest = run(core, 30);
  const heard = [];
  for (const part of [first, held, rest]) {
    part.wrote.forEach((wrote, q) => {
      if (wrote) heard.push(...part.left.slice(q * QUANTUM, (q + 1) * QUANTUM));
    });
  }
  out.dry = {
    starved: dry.starved,
    playedWhenDry: dry.played,
    heldSilent: !held.wrote.some(Boolean),
    heard: heard.length,
    // Past the block that fades back in, the stream is where it would have been.
    after: worst(heard, want[0].whole, halfway + BLOCK, frames),
    before: worst(heard, want[0].whole, BLOCK, halfway),
  };
}

// --- a new head, a new mix: one block of crossfade, and no more --------------------
{
  const other = headMatrix3(-1.9, 0.3, 0);
  const y = signal();
  const core = createStreamDecoder({ channels });
  core.setTotal(frames);
  core.setFilters(rotateSpectra(base, order, head));
  core.start(0, 1);
  core.push({ gen: 1, start: 0, frames, data: x, go: true });
  // The head turns while block 2 is sounding: block 3 fades to it. The mix changes while
  // block 3 is, block 4 being made already: block 5 fades to it.
  const made = run(core, (frames + PREROLL) / QUANTUM, (q) => {
    if (q === 4 + 2 * 4 + 1) core.setFilters(rotateSpectra(base, order, other));
    if (q === 4 + 3 * 4 + 1) core.push({ gen: 2, start: 0, frames, data: y, go: true });
  });
  const a = reference(x, head)[0].whole;
  const b = reference(x, other)[0].whole;
  const c = reference(y, other)[0].whole;
  const at = (n) => made.left[n + PREROLL];
  const fade = (from, to, block) => {
    let error = 0;
    let peak = 0;
    for (let i = 0; i < BLOCK; i++) {
      const n = block * BLOCK + i;
      const w = (i + 1) / BLOCK;
      error = Math.max(error, Math.abs(at(n) - (from[n] + (to[n] - from[n]) * w)));
      peak = Math.max(peak, Math.abs(from[n]));
    }
    return error / peak;
  };
  out.changes = {
    before: worst(made.left, a, BLOCK, 3 * BLOCK, PREROLL),
    headFade: fade(a, b, 3),
    mixFade: fade(b, c, 5),
    between: worst(made.left, b, 4 * BLOCK, 5 * BLOCK, PREROLL),
    after: worst(made.left, c, 6 * BLOCK, frames, PREROLL),
    spent: core.takeSpent().length,
    staleRefused: core.push({ gen: 1, start: 0, frames: 1, data: new Float32Array(channels), go: false }),
  };
}

// --- the head's matrix -------------------------------------------------------------
{
  const flat = headMatrix3(0.4, -0.3, 0);
  const plain = headMatrix(0.4, -0.3);
  const rolled = headMatrix3(0.4, -0.3, 0.5);
  const dot = (m, i, j) => m[i] * m[j] + m[i + 3] * m[j + 3] + m[i + 6] * m[j + 6];
  out.head = {
    noRoll: Math.max(...flat.map((value, i) => Math.abs(value - plain[i]))),
    orthonormal: Math.max(
      Math.abs(dot(rolled, 0, 0) - 1),
      Math.abs(dot(rolled, 1, 1) - 1),
      Math.abs(dot(rolled, 2, 2) - 1),
      Math.abs(dot(rolled, 0, 1)),
      Math.abs(dot(rolled, 0, 2)),
      Math.abs(dot(rolled, 1, 2))
    ),
    // Facing the frame's front with a positive roll: the left ear (column 1) rises.
    leftEarUp: headMatrix3(-Math.PI / 2, 0, 0.5)[7],
    front: [0, 3, 6].map((i) => headMatrix3(-Math.PI / 2, 0, 0.5)[i]),
    bins: BINS,
  };
}

// --- what is drawn of a step ---------------------------------------------------------
{
  const step = {
    listener: [1, 1.5, 2],
    direction: [1, 0, 0, 0, 0, -1, 0, 1, 0],
    delay_s: [0.01, 0.02, 0.03],
    gain: [0.5, 0.05, 0.0005],
    order: [0, 1, 2],
    kind: [0, 1, 2],
  };
  const glyphs = arrivalGlyphs(step, 343.2);
  out.glyphs = {
    ends: glyphs.map((glyph) => glyph.end),
    radii: glyphs.map((glyph) => glyph.radius),
    orders: glyphs.map((glyph) => glyph.order),
    shell: Array.from(shellPoints([1, 1.5, 2], [1, 0, 0, 0, 1, 0], [0, 1])),
  };
}

// --- the level: its default, and the output held against full scale ---------------
{
  const meter = createClipMeter({ hold: 3 });
  const seen = {};
  const at = (db) => levelGain(db);
  // Three of fourteen sources peaked at -3.8 dB re full scale at a level of 0 dB.
  const three = 10 ** (-3.8 / 20);
  // Fourteen of them, their powers added: what a whole scene is expected to reach.
  const fourteen = three * Math.sqrt(14 / 3);
  meter.report(three, at(DEFAULT_LEVEL_DB), 0);
  meter.report(fourteen, at(DEFAULT_LEVEL_DB), 1);
  seen.wholeSceneAtDefault = meter.state(1);
  seen.headroomDb = -20 * Math.log10(fourteen * at(DEFAULT_LEVEL_DB));
  meter.reset();
  // The same scene at the old default: past full scale, said, and for three seconds.
  meter.report(fourteen, at(0), 10);
  seen.atZero = meter.state(10);
  meter.report(0.1, at(0), 10.5);
  seen.heldAfter = meter.state(12.9).clipping;
  seen.released = meter.state(13.1);
  // One excursion is one event however many reports it lasts; the next is another.
  meter.report(2, at(0), 20);
  meter.report(2, at(0), 20.01);
  meter.report(0.1, at(0), 20.02);
  meter.report(1.5, at(0), 21);
  seen.events = meter.state(21).events;
  seen.worstDb = meter.state(21).overDb;
  meter.reset();
  seen.afterReset = meter.state(21);
  // Exactly full scale is not past it; what is not a number is not a report.
  meter.report(1, 1, 30);
  meter.report(NaN, 1, 30);
  seen.atFullScale = meter.state(30).clipping;
  out.level = {
    default: DEFAULT_LEVEL_DB,
    range: LEVEL_RANGE_DB,
    fullScaleSpl: FULL_SCALE_SPL_DB,
    gains: [levelGain(DEFAULT_LEVEL_DB), levelGain("0"), levelGain(""), levelGain("abc"), levelGain(99), levelGain(-99)],
    peak: peakOf(new Float32Array([0.1, -0.5, 0.2]), new Float32Array([0.3, 0.4, -0.75]), 0.25),
    peakHeld: peakOf(new Float32Array([0.1]), new Float32Array([0.2]), 0.9),
    ...seen,
  };
}

// --- several packs of one scene: which, in which order, under which words -------------
{
  const { sameScene, scenePacks, variantLabels, variantName, variantCost, variantSaving, variantTitle, stepVariant } = await import(
    `${app}/scene/variants.js`
  );
  const told = (name, flags, scene, excerpt, measured = null) => ({
    name,
    says: `what ${name} is`,
    flags,
    predicted_scene_usd: scene,
    predicted_excerpt_usd: excerpt,
    measured,
  });
  const packs = [
    { id: "cccccc000000", name: "low-0.8s", recipe_sha256: "r1", scene_sha256: "s1", path: "/kit/low/pack.h5", variant: told("low-0.8s", { low_seconds: 0.8 }, 5.07, 0.57) },
    { id: "aaaaaa000000", name: "reference", recipe_sha256: "r1", scene_sha256: "s1", variant: told("reference", {}, 7.54, 0.84, { usd: 0.91 }) },
    // Rails at another pitch: another recipe, the same movements.
    { id: "bbbbbb000000", name: "all-cheap", recipe_sha256: "r2", scene_sha256: "s1", variant: told("all-cheap", { low_ppw: 7.2, rail_pitch_m: 0.12 }, 1.81, 0.18) },
    { id: "dddddd000000", name: "another", recipe_sha256: "r9", scene_sha256: "s9", variant: { name: null, flags: {} } },
    // A pack brought home twice under one name, and one with no variant.json beside it.
    { id: "eeeeee000000", name: "low-0.8s", recipe_sha256: "r1", scene_sha256: "s1", variant: told("low-0.8s", { low_seconds: 0.8 }, 5.07, 0.57) },
    { id: "ffffff000000", name: "pulled", recipe_sha256: "r1", scene_sha256: "s1", variant: { name: null, flags: {} } },
  ];
  const [low, reference, cheap, another] = packs;
  const offered = scenePacks(packs, low);
  out.variants = {
    sameSceneOtherRecipe: sameScene(reference, cheap),
    otherScene: sameScene(reference, another),
    // An older server gives no scene digest: the recipe decides, as it did.
    withoutDigest: [
      sameScene({ recipe_sha256: "r1" }, { recipe_sha256: "r1" }),
      sameScene({ recipe_sha256: "r1" }, { recipe_sha256: "r2", scene_sha256: "s1" }),
      sameScene(null, reference),
    ],
    order: offered.map((entry) => entry.id.slice(0, 6)),
    labels: variantLabels(offered),
    alone: scenePacks(packs, another),
    none: scenePacks(packs, null),
    names: [variantName(reference), variantName(packs[5])],
    costs: [variantCost(reference), variantCost(low), variantCost(another)],
    saving: [variantSaving(cheap, reference), variantSaving(reference, reference), variantSaving(packs[5], reference)],
    title: variantTitle(cheap, reference).split("\n"),
    referenceTitle: variantTitle(reference, reference).split("\n"),
    bareTitle: variantTitle(packs[5], reference).split("\n"),
    steps: [
      stepVariant(offered, "aaaaaa000000", 1).id.slice(0, 6),
      stepVariant(offered, "aaaaaa000000", -1).id.slice(0, 6),
      stepVariant(offered, "missing", 1).id.slice(0, 6),
      stepVariant([], "x", 1),
    ],
  };
}

console.log(JSON.stringify(out));
