/** Where a point falls on a field's lattice: the one answer the page and the
 * offline harness must agree on, so neither can drift from the other.
 *
 * A field's `index.json` names a regular lattice, but not every cell of it is
 * held: the solver only wrote the ones over the flat's air. So a point maps
 * to its lattice cell when that cell exists, and otherwise to the nearest
 * held one the caller is willing to reach for.
 */

//: Shells the search will look through before giving up, whatever it is
//: asked for. A cap, not a distance: the distance is the caller's.
const MAX_RADIUS = 8;

export function createLattice(index) {
  const byCell = new Map();
  index.cell_index.forEach(([i, j, k], position) => byCell.set(`${i},${j},${k}`, position));
  const origin = index.grid_origin_m;
  const step = index.grid_step_m;
  const shape = index.grid_shape;

  // An axis with one layer has no step to divide by; every point maps to it.
  const along = (value, axis) =>
    shape[axis] <= 1 || !step[axis] ? 0 : (value - origin[axis]) / step[axis];

  /** The nearest held cell within `withinMetres`, searching outward.
   *
   * The lattice covers the flat's air, and the flat's air is holey: the
   * solver wrote nothing inside the furniture, against the walls, or wherever
   * a cell did not fit, and on `hssd_0076` there are more lattice points with
   * no cell than with one. A listener walking a room passes through those
   * holes, so how far this is willing to look decides whether they keep
   * hearing the room or fall off it.
   */
  function cellAt(x, y, z, withinMetres = Infinity) {
    const i = Math.round(along(x, 0));
    const j = Math.round(along(y, 1));
    const k = Math.round(along(z, 2));
    // A shell of radius r holds nothing nearer than `(r - 1)` steps, so the
    // search can stop as soon as even the nearest corner of the next shell
    // would be further than the caller will accept.
    const widest = Math.max(step[0] || 0, step[1] || 0, step[2] || 0) || 1;
    const limit = Math.min(MAX_RADIUS, Math.ceil(withinMetres / widest) + 1);
    for (let radius = 0; radius <= limit; radius++) {
      let best = null;
      let bestDistance = Infinity;
      for (let di = -radius; di <= radius; di++) {
        for (let dj = -radius; dj <= radius; dj++) {
          for (let dk = -radius; dk <= radius; dk++) {
            if (Math.max(Math.abs(di), Math.abs(dj), Math.abs(dk)) !== radius) continue;
            const position = held(i + di, j + dj, k + dk);
            if (position === null) continue;
            const [px, py, pz] = index.positions[position];
            const distance = Math.hypot(px - x, py - y, pz - z);
            if (distance < bestDistance) {
              bestDistance = distance;
              best = position;
            }
          }
        }
      }
      if (best !== null) return bestDistance <= withinMetres ? best : null;
    }
    return null;
  }

  function held(ci, cj, ck) {
    if (ci < 0 || cj < 0 || ck < 0 || ci >= shape[0] || cj >= shape[1] || ck >= shape[2]) return null;
    const position = byCell.get(`${ci},${cj},${ck}`);
    return position === undefined ? null : position;
  }

  return { cellAt };
}
