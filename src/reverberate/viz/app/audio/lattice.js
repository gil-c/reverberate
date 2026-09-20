/** Where a point falls on a field's lattice: the one answer the page and the
 * offline harness must agree on, so neither can drift from the other.
 *
 * A field's `index.json` names a regular lattice, but not every cell of it is
 * held: the solver only wrote the ones over the flat's air. So a point maps
 * to its lattice cell when that cell exists, and otherwise to the nearest
 * held one the caller is willing to reach for.
 */

//: Shells the search will look through before giving up, whatever it is
//: asked for: 3.2 m at the usual 40 cm lattice. A cap, not a distance: the
//: distance is the caller's.
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
   *
   * The search goes out a shell of the lattice at a time, and does not stop
   * at the first shell with a cell in it: a corner of one shell can be
   * further than the middle of a face of the next. It stops once no shell
   * further out can hold anything nearer than what it has, or anything within
   * reach at all.
   */
  function cellAt(x, y, z, withinMetres = Infinity) {
    const centre = [Math.round(along(x, 0)), Math.round(along(y, 1)), Math.round(along(z, 2))];
    // Everything in shell `r` is at least `r - 1/2` steps of the finest axis
    // away, the point being anywhere within half a step of its own node.
    const steps = [0, 1, 2].filter((axis) => shape[axis] > 1 && step[axis]).map((axis) => step[axis]);
    const unit = steps.length ? Math.min(...steps) : 1;
    let best = null;
    let bestDistance = Infinity;
    for (let radius = 0; radius <= MAX_RADIUS; radius++) {
      const nearest = Math.max(0, radius - 0.5) * unit;
      if (nearest > withinMetres || nearest > bestDistance) break;
      // The shell, clipped to the lattice.
      const lo = centre.map((c) => Math.max(-radius, -c));
      const hi = centre.map((c, axis) => Math.min(radius, shape[axis] - 1 - c));
      for (let di = lo[0]; di <= hi[0]; di++) {
        for (let dj = lo[1]; dj <= hi[1]; dj++) {
          for (let dk = lo[2]; dk <= hi[2]; dk++) {
            if (Math.max(Math.abs(di), Math.abs(dj), Math.abs(dk)) !== radius) continue;
            const position = held(centre[0] + di, centre[1] + dj, centre[2] + dk);
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
    }
    return best !== null && bestDistance <= withinMetres ? best : null;
  }

  function held(ci, cj, ck) {
    if (ci < 0 || cj < 0 || ck < 0 || ci >= shape[0] || cj >= shape[1] || ck >= shape[2]) return null;
    const position = byCell.get(`${ci},${cj},${ck}`);
    return position === undefined ? null : position;
  }

  return { cellAt };
}
