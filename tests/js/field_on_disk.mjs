/** A field read from a directory on this machine, as the page reads one over
 * HTTP: the same `index.json`, the same byte ranges, the same lattice.
 *
 * `reverberate.viz.field_payload.build_site` writes the directory; nothing
 * here parses HDF5, because a cell is one contiguous block of float32 whose
 * offset the index already names.
 */
import { open, readFile } from "node:fs/promises";
import { join } from "node:path";

import { createLattice } from "../../src/reverberate/viz/app/audio/lattice.js";

export async function loadFieldOnDisk(directory) {
  const index = JSON.parse(await readFile(join(directory, "index.json"), "utf8"));
  const handle = await open(join(directory, index.file), "r");
  const { cellAt } = createLattice(index);
  const cells = new Map();

  async function cell(position) {
    if (!cells.has(position)) {
      const bytes = Buffer.alloc(index.cell_bytes);
      const { bytesRead } = await handle.read(bytes, 0, index.cell_bytes, index.offsets[position]);
      if (bytesRead !== index.cell_bytes) {
        throw new Error(`${index.file}: asked ${index.cell_bytes} bytes, got ${bytesRead}`);
      }
      const all = new Float32Array(bytes.buffer, bytes.byteOffset, index.cell_bytes / 4);
      cells.set(position, Array.from({ length: index.channels }, (_, ch) => all.subarray(ch * index.samples, (ch + 1) * index.samples)));
    }
    return cells.get(position);
  }

  return { index, cellAt, cell, close: () => handle.close() };
}

/** A decoder as `reverberate.viz.decoders.export_decoders` writes one. */
export async function loadDecoder(directory, name = "measured") {
  const record = JSON.parse(await readFile(join(directory, `${name}.json`), "utf8"));
  const bytes = await readFile(join(directory, record.url));
  const filters = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  return { order: record.order, channels: record.channels, taps: record.taps, filters };
}
