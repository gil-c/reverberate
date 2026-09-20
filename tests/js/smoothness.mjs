/** How smooth a walk through a field sounds, for the page as it ships and
 * for the page with each of its decisions undone.
 *
 *     node tests/js/smoothness.mjs --field <dir> --decoders <dir> [--case turn|walk|both] [--variant name]
 *
 * `<dir>` is what `reverberate.viz.field_payload.build_site` and
 * `reverberate.viz.decoders.export_decoders` write. With no `--variant` it
 * runs them all and prints one row each; ADR 0013 records what they said.
 */
import { loadDecoder, loadFieldOnDisk } from "./field_on_disk.mjs";
import { measureTrace } from "./metrics.mjs";
import { renderFloor, renderWalk, SHIPPED, trajectory } from "./walk.mjs";

//: The app's own walk and turn speeds (`viz/app/viewport.js`).
const WALK_SPEED_M_S = 2.2;
const TURN_SPEED_RAD_S = 1.75;

export const VARIANTS = {
  //: The page as it ships.
  shipped: {},
  //: Standing still: the instrument's own floor, which must read zero.
  frozen: { minIntervalS: 1e9 },
  //: Each decision undone, one at a time.
  noDelayLine: { alignDirect: false },
  convolverNodes: { worklet: false },
  fadeShorterThanInterval: { engineFadeS: 0.035 },
  tailEveryCell: { tailEveryCell: true, tailFadeS: 0.08 },
  //: A shorter fade: the sound a degree nearer the head, walking less smooth.
  fade80: { engineFadeS: 0.08, convolvers: 4 },
  //: The page before any of it, as near as this harness can put it: two
  //: `ConvolverNode`s, a 50 ms fade every 60 ms, no delay line, the late part
  //: at every cell. The seam and the decoder's lead stay fixed.
  before: {
    worklet: false,
    convolvers: 2,
    engineFadeS: 0.05,
    minIntervalS: 0.06,
    alignDirect: false,
    tailEveryCell: true,
    tailFadeS: 0.05,
  },
  //: The lattice on its own, every frame decoded at the true pose.
  lattice: { floor: true },
};

export const CASES = {
  turn: { turnRate: TURN_SPEED_RAD_S, walk: 0 },
  walk: { turnRate: 0, walk: WALK_SPEED_M_S },
  both: { turnRate: TURN_SPEED_RAD_S, walk: WALK_SPEED_M_S },
};

/** The longest straight line of held cells, to walk along. */
export function longestRun(field) {
  const { grid_shape: shape, grid_step_m: step, grid_origin_m: origin, cell_index: cells } = field.index;
  const held = new Set(cells.map((cell) => cell.join(",")));
  let best = { length: 0 };
  for (const axis of [0, 2]) {
    const other = 2 - axis;
    for (let fixed = 0; fixed < shape[other]; fixed++) {
      let run = 0;
      for (let along = 0; along < shape[axis]; along++) {
        const at = [0, 0, 0];
        at[axis] = along;
        at[other] = fixed;
        run = held.has(at.join(",")) ? run + 1 : 0;
        if (run > best.length) best = { length: run, axis, from: along - run + 1, fixed, other };
      }
    }
  }
  const at = [0, 0, 0];
  at[best.axis] = best.from;
  at[best.other] = best.fixed;
  const [x, y, z] = [0, 1, 2].map((axis) => origin[axis] + at[axis] * step[axis]);
  return { x, y, z, axis: best.axis, metres: (best.length - 1) * step[best.axis] };
}

export async function run({ fieldDirectory, decoderDirectory, decoderName = "measured", seconds = 6, cases, variants }) {
  const field = await loadFieldOnDisk(fieldDirectory);
  const decoderFilters = await loadDecoder(decoderDirectory, decoderName);
  const sampleRate = field.index.sample_rate_hz;
  const line = longestRun(field);
  const rows = [];
  for (const [caseName, move] of Object.entries(cases)) {
    // No faster than the line is long, so the walk stays on solved air.
    const speed = Math.min(move.walk, line.metres / seconds);
    const walk = line.axis === 0 ? [speed, 0] : [0, speed];
    const from = { x: line.x, z: line.z, yaw: 0 };
    const poses = trajectory({ seconds, frameHz: SHIPPED.frameHz, from, y: line.y, walk, turnRate: move.turnRate });
    for (const [variantName, variant] of Object.entries(variants)) {
      const rendered = variant.floor
        ? await renderFloor({ field, decoderFilters, poses, sampleRate })
        : await renderWalk({ field, decoderFilters, poses, sampleRate, variant });
      const lag = rendered.lagDegrees ? [...rendered.lagDegrees].sort((a, b) => a - b) : [0];
      rows.push({
        case: caseName,
        variant: variantName,
        ...measureTrace(rendered.trace, { sampleRate }),
        lagP95Degrees: Number(lag[Math.round(0.95 * (lag.length - 1))].toFixed(1)),
        swaps: rendered.swaps ?? rendered.trace.length,
      });
    }
  }
  await field.close();
  return { line, rows };
}

const COLUMNS = [
  ["case", "case"],
  ["variant", "variant"],
  ["level jerk dB", "levelJerkDb"],
  ["timbre step dB", "timbreStepDb"],
  ["reverb jerk dB", "reverbJerkDb"],
  ["itd jerk smp", "itdJerkSamples"],
  ["ild jerk dB", "ildJerkDb"],
  ["arrival jerk smp", "arrivalJerkSamples"],
  ["lag p95 deg", "lagP95Degrees"],
  ["swaps", "swaps"],
];

export function table(rows) {
  const cells = [COLUMNS.map(([title]) => title), ...rows.map((row) => COLUMNS.map(([, key]) => String(row[key] ?? "-")))];
  const widths = cells[0].map((_, column) => Math.max(...cells.map((row) => row[column].length)));
  return cells.map((row) => row.map((value, column) => value.padStart(widths[column])).join("  ")).join("\n");
}

async function main(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i += 2) options[argv[i].replace(/^--/, "")] = argv[i + 1];
  if (!options.field || !options.decoders) {
    console.error("usage: smoothness.mjs --field <dir> --decoders <dir> [--case name] [--variant name] [--seconds n]");
    return 2;
  }
  const { rows } = await run({
    fieldDirectory: options.field,
    decoderDirectory: options.decoders,
    decoderName: options.decoder || "measured",
    seconds: Number(options.seconds || 6),
    cases: options.case ? { [options.case]: CASES[options.case] } : CASES,
    variants: options.variant ? { [options.variant]: VARIANTS[options.variant] } : VARIANTS,
  });
  console.log(table(rows));
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await main(process.argv.slice(2));
}
