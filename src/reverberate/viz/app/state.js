/** The one shared state object.
 *
 * The viewport owns the listener's pose and announces it through `onMove`;
 * every other module reads this object and asks the viewport to move rather
 * than writing the pose itself.
 */
export const state = {
  // Scene frame, metres, y up; yaw and pitch in radians, fov in degrees.
  listener: { x: 0, y: 1.6, z: 0, yaw: 0, pitch: 0, fov: 70 },
  // "colour" or "acoustic".
  view: "colour",
  // Band limit of the mesh drawn in the acoustic view, as the key of
  // run.meshes ("4000"), or null when none is drawable here.
  fmax: null,
  // The apartment open, by short name, and its manifest once fetched.
  apartment: null,
  manifest: null,
  run: null,
  // The room the listener stands in: the plan's entry, or null outside.
  room: null,
  // Sources of the run with the page's own per-source state.
  sources: [],
  selected: null,
};
