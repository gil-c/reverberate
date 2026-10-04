/** Reading the tracks the server sampled: where everything is at a time `t`.
 *
 * The page never places a source from a recipe. The server does, with the
 * recipe's own kinematics, on a grid of times (`/api/scene/tracks`); between
 * two of its samples the page draws a straight line. That is a drawing, to
 * within half a step of travel, and nothing computed downstream may read it.
 */

/** The samples around `t` and the weight of the second: `t = (1 - w) t[i] + w t[i + 1]`. */
export function bracket(times, step, t) {
  const last = times.length - 1;
  if (last <= 0 || t <= times[0]) return { i: 0, w: 0 };
  if (t >= times[last]) return { i: last - 1, w: 1 };
  // The grid is even but for its last interval, which stops at the scene's end.
  let i = Math.min(last - 1, Math.max(0, Math.floor((t - times[0]) / step)));
  while (i > 0 && times[i] > t) i--;
  while (i < last - 1 && times[i + 1] <= t) i++;
  const span = times[i + 1] - times[i];
  return { i, w: span > 0 ? (t - times[i]) / span : 0 };
}

const mix = (column, i, w) => column[i] + (column[i + 1] - column[i]) * w;

/** A source's mouth at `t`: position in metres, yaw in degrees, not wrapped. */
export function sourceAt(tracks, track, t) {
  const { i, w } = bracket(tracks.t, tracks.step_s, t);
  return { x: mix(track.x, i, w), y: mix(track.y, i, w), z: mix(track.z, i, w), yaw_deg: mix(track.yaw_deg, i, w) };
}

/** The listener's head at `t`. */
export function listenerAt(tracks, t) {
  const track = tracks.listener;
  const { i, w } = bracket(tracks.t, tracks.step_s, t);
  return {
    x: mix(track.x, i, w),
    y: mix(track.y, i, w),
    z: mix(track.z, i, w),
    yaw_deg: mix(track.yaw_deg, i, w),
    pitch_deg: mix(track.pitch_deg, i, w),
    roll_deg: mix(track.roll_deg, i, w),
  };
}

/** Whether `t` falls in one of `intervals`, `[start, end]` pairs in time order. */
export function activeAt(intervals, t) {
  let low = 0;
  let high = intervals.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (t < intervals[middle][0]) high = middle - 1;
    else if (t > intervals[middle][1]) low = middle + 1;
    else return true;
  }
  return false;
}

/** The movement span holding `t`: the last that has started. */
export function spanAt(spans, t) {
  let low = 0;
  let high = spans.length - 1;
  let found = spans[0] || null;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (spans[middle].start_s <= t) {
      found = spans[middle];
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return found;
}

/** Where a track has been over the `seconds` before `t`: `[x, y, z, ...]`, oldest first,
 *  ending where the track is at `t`. Written into `into` when given, which
 *  must hold `3 * trailCapacity(...)` numbers; the count of points is returned. */
export function trail(tracks, track, t, seconds, into = null) {
  const from = Math.max(tracks.t[0], t - seconds);
  const first = bracket(tracks.t, tracks.step_s, from);
  const last = bracket(tracks.t, tracks.step_s, t);
  const out = into || [];
  let count = 0;
  const put = (x, y, z) => {
    out[3 * count] = x;
    out[3 * count + 1] = y;
    out[3 * count + 2] = z;
    count += 1;
  };
  put(mix(track.x, first.i, first.w), mix(track.y, first.i, first.w), mix(track.z, first.i, first.w));
  for (let i = first.i + 1; i <= last.i; i++) put(track.x[i], track.y[i], track.z[i]);
  if (last.w > 0) put(mix(track.x, last.i, last.w), mix(track.y, last.i, last.w), mix(track.z, last.i, last.w));
  if (!into) out.length = 3 * count;
  return into ? count : out;
}

/** The most points a trail of `seconds` can have on a grid of `step`. */
export const trailCapacity = (seconds, step) => Math.ceil(seconds / step) + 3;

const RAD = Math.PI / 180;

/** The viewport's yaw, in radians, for a recipe's: the recipe's zero faces
 *  `+x` and the camera's faces `-z`, both counter clockwise seen from above. */
export const viewYaw = (yawDeg) => (yawDeg - 90) * RAD;

/** The unit `(x, z)` a recipe's yaw faces: 0 is `+x`, +90 is `-z`. */
export const heading = (yawDeg) => [Math.cos(yawDeg * RAD), -Math.sin(yawDeg * RAD)];
