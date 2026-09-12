/** One source's impulse response field, read straight out of its HDF5.
 *
 * `index.json` names the lattice, every cell it holds, and the byte range of
 * each cell's response in the file; a cell is one range request of its
 * `cell_bytes`, kept under a byte cap, oldest out first.
 */

//: Bytes of cells kept per field. At order 7 and 1.2 s a cell is 14.7 MB, so
//: this holds about twenty-five cells: a few metres of walk in every direction.
const CACHE_BYTES = 400e6;

export async function loadField(url) {
  const index = await fetch(`${url}/index.json`).then((r) => {
    if (!r.ok) throw new Error(`${url}/index.json: ${r.status}`);
    return r.json();
  });
  const file = `${url}/${index.file}`;
  const byCell = new Map();
  index.cell_index.forEach(([i, j, k], position) => byCell.set(`${i},${j},${k}`, position));
  const origin = index.grid_origin_m;
  const step = index.grid_step_m;
  const shape = index.grid_shape;
  const cells = new Map(); // position -> { promise, buffer, at }
  let held = 0;

  // An axis with one layer has no step to divide by; every point maps to it.
  const along = (value, axis) =>
    shape[axis] <= 1 || !step[axis] ? 0 : Math.round((value - origin[axis]) / step[axis]);
  const lattice = (x, y, z) => [along(x, 0), along(y, 1), along(z, 2)];

  /** The nearest held cell to a point, searching outward from its lattice cell. */
  function cellAt(x, y, z) {
    const [i, j, k] = lattice(x, y, z);
    for (let radius = 0; radius <= 2; radius++) {
      let best = null;
      let bestDistance = Infinity;
      for (let di = -radius; di <= radius; di++) {
        for (let dj = -radius; dj <= radius; dj++) {
          for (let dk = -radius; dk <= radius; dk++) {
            if (Math.max(Math.abs(di), Math.abs(dj), Math.abs(dk)) !== radius) continue;
            const ci = i + di;
            const cj = j + dj;
            const ck = k + dk;
            if (ci < 0 || cj < 0 || ck < 0 || ci >= shape[0] || cj >= shape[1] || ck >= shape[2]) {
              continue;
            }
            const position = byCell.get(`${ci},${cj},${ck}`);
            if (position === undefined) continue;
            const [px, py, pz] = index.positions[position];
            const distance = Math.hypot(px - x, py - y, pz - z);
            if (distance < bestDistance) {
              bestDistance = distance;
              best = position;
            }
          }
        }
      }
      if (best !== null) return best;
    }
    return null;
  }

  async function range(offset, bytes) {
    const response = await fetch(file, { headers: { Range: `bytes=${offset}-${offset + bytes - 1}` } });
    if (response.status !== 206) {
      throw new Error(`${index.file}: expected a partial response, got ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== bytes) {
      throw new Error(`${index.file}: asked ${bytes} bytes, got ${buffer.byteLength}`);
    }
    return buffer;
  }

  async function fetchCell(position) {
    let entry = cells.get(position);
    if (!entry) {
      entry = { promise: range(index.offsets[position], index.cell_bytes), buffer: null, at: performance.now() };
      cells.set(position, entry);
      entry.buffer = await entry.promise;
      held += entry.buffer.byteLength;
      while (held > CACHE_BYTES && cells.size > 1) {
        let oldest = null;
        for (const [key, other] of cells) {
          if (other.buffer && (oldest === null || other.at < cells.get(oldest).at)) oldest = key;
        }
        if (oldest === null || oldest === position) break;
        held -= cells.get(oldest).buffer.byteLength;
        cells.delete(oldest);
      }
    }
    entry.at = performance.now();
    return entry.promise;
  }

  /** The channels of one cell, as views into its bytes. */
  async function cell(position) {
    const buffer = await fetchCell(position);
    const samples = index.samples;
    const channels = [];
    for (let ch = 0; ch < index.channels; ch++) {
      channels.push(new Float32Array(buffer, ch * samples * 4, samples));
    }
    return channels;
  }

  return {
    index,
    cellAt,
    cell,
    /** What the producer says of a cell: its room and the band solved there. */
    describe: (position) => ({
      room: index.rooms ? index.rooms[position] : null,
      solvedToHz: index.solved_to_hz ? index.solved_to_hz[position] : null,
      hasHigh: index.has_high ? index.has_high[position] : null,
    }),
    /** Fetch a cell without waiting for it. */
    prefetch: (position) => {
      if (position !== null) fetchCell(position).catch(() => {});
    },
    /** Every cell's centre, for drawing the measurement points. */
    positions: () => index.positions,
  };
}
