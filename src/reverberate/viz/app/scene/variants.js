/** Several packs of one scene: which they are, what each is called and what it cost.
 *
 * A scene traced with other options is another pack: a coarser wave grid, a
 * shorter low response, fewer rays, rails solved further apart. The owner
 * judges them by ear, switching the pack heard at one instant. This module
 * is what the page knows of that, and no DOM: which packs are one scene,
 * in which order they are offered, and the words beside each.
 *
 * **One scene, not one recipe.** A pack traced on rails of another pitch is
 * of another recipe, since a rail's pitch is the recipe's, and of the same
 * movements. The server gives every pack a `scene_sha256`, the digest of
 * where the head and every source are at every step: two packs with the
 * same one are heard at the same instant on the same picture. Without it
 * (an older server) the recipe's digest decides, as it did.
 *
 * What a variant is called and what it cost come from the `variant.json`
 * `python -m reverberate.trace variants` wrote beside the pack, which the
 * server reads into the pack's `variant`; a pack without one is called by
 * its own name and shows no cost.
 */

/** Whether two packs are one scene: the same movements, whatever their recipes. */
export function sameScene(a, b) {
  if (!a || !b) return false;
  if (a.scene_sha256 && b.scene_sha256) return a.scene_sha256 === b.scene_sha256;
  return a.recipe_sha256 === b.recipe_sha256;
}

const flagsOf = (entry) => (entry.variant && entry.variant.flags) || {};
const isReference = (entry) => Boolean(entry.variant && entry.variant.name) && Object.keys(flagsOf(entry)).length === 0;

/** A pack's name among its scene's: its variant's, or its own. */
export function variantName(entry) {
  return (entry.variant && entry.variant.name) || entry.name;
}

/** The packs that are the scene of `current`, the reference first, then by name; a
 * scene with one pack has no choice to offer and gives none. */
export function scenePacks(packs, current) {
  if (!current) return [];
  const found = packs.filter((entry) => sameScene(entry, current));
  if (found.length < 2) return [];
  return found.sort(
    (a, b) => Number(isReference(b)) - Number(isReference(a)) || variantName(a).localeCompare(variantName(b)) || a.id.localeCompare(b.id)
  );
}

/** Buttons' labels: the names, and where two packs share one, six characters of each id. */
export function variantLabels(entries) {
  const count = new Map();
  for (const entry of entries) count.set(variantName(entry), (count.get(variantName(entry)) || 0) + 1);
  return entries.map((entry) => (count.get(variantName(entry)) > 1 ? `${variantName(entry)} · ${entry.id.slice(0, 6)}` : variantName(entry)));
}

const usd = (value) => `${Number(value).toFixed(2)} USD`;

/** What a variant cost, in the words of the bar: predicted for the whole scene, and what
 * this pack's rental was billed. Empty for a pack that says neither. */
export function variantCost(entry) {
  const told = entry.variant || {};
  const parts = [];
  if (told.predicted_scene_usd != null) parts.push(`whole scene ${usd(told.predicted_scene_usd)} predicted`);
  if (told.measured && told.measured.usd != null) parts.push(`this pack ${usd(told.measured.usd)} billed`);
  else if (told.predicted_excerpt_usd != null) parts.push(`this pack ${usd(told.predicted_excerpt_usd)} predicted`);
  return parts.join(", ");
}

/** What a variant saves on the whole scene against `reference`, per cent; null when unknown. */
export function variantSaving(entry, reference) {
  const mine = entry.variant && entry.variant.predicted_scene_usd;
  const theirs = reference && reference.variant && reference.variant.predicted_scene_usd;
  if (mine == null || theirs == null || !(theirs > 0)) return null;
  return Math.round(100 * (1 - mine / theirs));
}

/** A button's tooltip: what the variant is, its options, its cost. */
export function variantTitle(entry, reference = null) {
  const told = entry.variant || {};
  const flags = Object.entries(flagsOf(entry)).map(([name, value]) => `${name} = ${value}`);
  const lines = [];
  if (told.says) lines.push(told.says);
  lines.push(flags.length ? flags.join(", ") : told.name ? "the reference: no option" : "no variant.json beside this pack");
  const cost = variantCost(entry);
  if (cost) lines.push(cost);
  const saving = variantSaving(entry, reference);
  if (saving !== null && reference && reference.id !== entry.id) lines.push(`${saving} % less than ${variantName(reference)} on the whole scene`);
  lines.push(entry.path || "");
  return lines.filter(Boolean).join("\n");
}

/** The pack heard next or before in the row, wrapping: for a key that steps through them. */
export function stepVariant(entries, currentId, by) {
  if (!entries.length) return null;
  const at = entries.findIndex((entry) => entry.id === currentId);
  return entries[(((at < 0 ? 0 : at) + by) % entries.length + entries.length) % entries.length];
}
