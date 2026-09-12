/** Spherical harmonic bookkeeping, mirrored from `reverberate.spatial.sh`:
 * ACN ordering, N3D normalisation, frame x front, y left, z up, and the
 * exact rotation of a field for a head orientation.
 */
export const acn = (n, m) => n * n + n + m;
export const channelCount = (order) => (order + 1) * (order + 1);

/** The head yaw, in the ambisonic frame, of a camera yaw in the scene frame.
 *
 * The camera looks along (-sin yaw, 0, -cos yaw) in the y-up scene, and the
 * ambisonic frame is (x, -z, y) of it: at camera yaw 0 the listener faces
 * ambisonic +y, the left of the frame's own front, so the head yaw is a
 * quarter turn ahead. Both angles turn counter clockwise seen from above.
 */
export function headYawOfCamera(cameraYaw) {
  return cameraYaw + Math.PI / 2;
}

// --- the general rotation, for pitch ------------------------------------------

/** N3D real spherical harmonics at one unit direction, ACN order, as
 * `reverberate.spatial.sh.real_sh`: no Condon-Shortley phase. */
export function realSH(order, x, y, z) {
  const norm = Math.hypot(x, y, z);
  const cosTheta = Math.max(-1, Math.min(1, z / norm));
  const phi = Math.atan2(y, x);
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const out = new Float64Array(channelCount(order));
  // P[m][n] by the standard recurrences, positive at the pole.
  const legendre = [];
  for (let m = 0; m <= order; m++) {
    const column = new Float64Array(order + 1);
    let pmm = 1;
    for (let k = 1; k <= m; k++) pmm *= (2 * k - 1) * sinTheta;
    column[m] = pmm;
    if (m < order) column[m + 1] = cosTheta * (2 * m + 1) * pmm;
    for (let n = m + 2; n <= order; n++) {
      column[n] = ((2 * n - 1) * cosTheta * column[n - 1] - (n + m - 1) * column[n - 2]) / (n - m);
    }
    legendre.push(column);
  }
  const factorial = [1];
  for (let i = 1; i <= 2 * order; i++) factorial.push(factorial[i - 1] * i);
  for (let n = 0; n <= order; n++) {
    for (let m = -n; m <= n; m++) {
      const k = Math.abs(m);
      const scale = Math.sqrt(((2 * n + 1) * factorial[n - k]) / factorial[n + k]) * legendre[k][n];
      const angular = m > 0 ? Math.SQRT2 * Math.cos(k * phi) : m < 0 ? Math.SQRT2 * Math.sin(k * phi) : 1;
      out[acn(n, m)] = scale * angular;
    }
  }
  return out;
}

/** Gauss-Legendre in cos(theta) times a uniform ring in azimuth, exact for
 * polynomials of `degree` on the sphere; weights sum to 4 pi. */
export function quadrature(degree) {
  const nTheta = Math.floor(degree / 2) + 1;
  const nPhi = degree + 1;
  // Gauss-Legendre nodes by Newton on the Legendre polynomial.
  const nodes = [];
  const weights = [];
  for (let i = 0; i < nTheta; i++) {
    let x = Math.cos((Math.PI * (i + 0.75)) / (nTheta + 0.5));
    let dp = 0;
    for (let iteration = 0; iteration < 100; iteration++) {
      let p0 = 1;
      let p1 = x;
      for (let k = 2; k <= nTheta; k++) {
        const p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = (nTheta * (x * p1 - p0)) / (x * x - 1);
      const dx = p1 / dp;
      x -= dx;
      if (Math.abs(dx) < 1e-15) break;
    }
    nodes.push(x);
    weights.push(2 / ((1 - x * x) * dp * dp));
  }
  const directions = [];
  const w = [];
  for (let i = 0; i < nTheta; i++) {
    const sinTheta = Math.sqrt(1 - nodes[i] * nodes[i]);
    for (let j = 0; j < nPhi; j++) {
      const phi = (2 * Math.PI * (j + 0.5)) / nPhi;
      directions.push([sinTheta * Math.cos(phi), sinTheta * Math.sin(phi), nodes[i]]);
      w.push((weights[i] * 2 * Math.PI) / nPhi);
    }
  }
  return { directions, weights: w };
}

/** The block-diagonal matrix R with `f'(d) = f(M d)` for a 3x3 rotation `M`
 * (row-major, nine numbers): `c' = R c`. One dense block per degree, since a
 * rotation never mixes degrees. Built by projection on a quadrature exact
 * for degree 2 * order, so it is exact to rounding for any rotation. */
export function rotationBlocks(order, M) {
  const { directions, weights } = quadrature(2 * order);
  const at = directions.map(([x, y, z]) => realSH(order, x, y, z));
  const rotated = directions.map(([x, y, z]) =>
    realSH(
      order,
      M[0] * x + M[1] * y + M[2] * z,
      M[3] * x + M[4] * y + M[5] * z,
      M[6] * x + M[7] * y + M[8] * z
    )
  );
  const blocks = [];
  for (let n = 0; n <= order; n++) {
    const size = 2 * n + 1;
    const offset = n * n;
    const matrix = new Float64Array(size * size);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        let sum = 0;
        for (let q = 0; q < directions.length; q++) {
          sum += weights[q] * rotated[q][offset + j] * at[q][offset + i];
        }
        // N3D harmonics integrate to 4 pi times the identity.
        matrix[i * size + j] = sum / (4 * Math.PI);
      }
    }
    blocks.push({ offset, size, matrix });
  }
  return blocks;
}

/** The head's orientation in the ambisonic frame as the matrix `M` of
 * `rotationBlocks`, for a camera yaw and pitch in the scene: `f'(d) = f(M d)`
 * gives the field as the head hears it. Yaw about z, then pitch about the
 * head's own left axis; looking up is a negative turn about +y. */
export function headMatrix(cameraYaw, cameraPitch) {
  const psi = headYawOfCamera(cameraYaw);
  const theta = -cameraPitch;
  const cz = Math.cos(psi);
  const sz = Math.sin(psi);
  const cy = Math.cos(theta);
  const sy = Math.sin(theta);
  // Rz(psi) * Ry(theta), row-major.
  return [cz * cy, -sz, cz * sy, sz * cy, cz, sz * sy, -sy, 0, cy];
}
