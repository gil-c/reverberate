/** One source's impulse response field, read straight out of its HDF5.
 *
 * `index.json` names the lattice, every cell it holds, and the byte range of
 * each cell's response in the file; a cell is one range request of its
 * `cell_bytes`, kept under a byte cap, oldest out first.
 */
import { createLattice } from "./lattice.js";

//: Bytes of cells kept per field. At order 7 and 1.2 s a cell is 14.7 MB, so
//: this holds about twenty-five cells: a few metres of walk in every direction.
const CACHE_BYTES = 400e6;

export async function loadField(url) {
  const index = await fetch(`${url}/index.json`).then((r) => {
    if (!r.ok) throw new Error(`${url}/index.json: ${r.status}`);
    return r.json();
  });
  const file = `${url}/${index.file}`;
  const { cellAt } = createLattice(index);
  const cells = new Map(); // position -> { promise, buffer, at }
  let held = 0;

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
    }),
    /** Fetch a cell without waiting for it. */
    prefetch: (position) => {
      if (position !== null) fetchCell(position).catch(() => {});
    },
    /** Every cell's centre, for drawing the measurement points. */
    positions: () => index.positions,
  };
}
