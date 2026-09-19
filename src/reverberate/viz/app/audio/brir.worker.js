/** The decode off the main thread: the message shell around `decode.js`.
 *
 * Two instances of this script run, in two roles the messages give them:
 *
 * - **early**: a cell's first `earlyMs` are kept as spectra and re-decoded
 *   for every head orientation, in one go, and handed back already cut into
 *   the blocks the early convolver works in (`partitioned.js`).
 * - **late**: the rest of the response. Decoded at the head orientation of
 *   the moment when a cell is entered, then again, exactly, once the head has
 *   been still. A late decode is ten times the early one, so it runs in steps
 *   that yield between them and can be cancelled when the head moves again.
 *
 * The arithmetic is all in `decode.js`, which the offline harness drives with
 * no browser; nothing here does more than order the work and answer.
 */
import { createDecoder } from "./decode.js";
import { partitionFilter } from "./partitioned.js";

const decoder = createDecoder();
const cancelled = new Set();
let current = null; // the late render in progress, if any

function renderEarly(message) {
  const started = performance.now();
  const early = decoder.renderEarly({ key: message.cell, head: message.head });
  if (!early) {
    self.postMessage({ type: "brir", id: message.id, missing: message.cell });
    return;
  }
  // Cut and transformed here, for the early convolver, so the audio thread
  // only multiplies and adds. The two ears go too, for the page's readout.
  const filter = partitionFilter(early);
  self.postMessage({ type: "brir", id: message.id, early, filter, ms: performance.now() - started }, [
    early[0].buffer,
    early[1].buffer,
    filter.re.buffer,
    filter.im.buffer,
  ]);
}

/** The late decode, in steps that yield so a cancel can land between them. */
function renderLate(message) {
  const started = performance.now();
  const plan = decoder.lateSteps({ key: message.key, head: message.head });
  if (!plan) {
    self.postMessage({ type: "late", id: message.id, missing: message.key });
    return;
  }
  let at = 0;
  const step = () => {
    if (cancelled.has(message.id)) {
      cancelled.delete(message.id);
      current = null;
      self.postMessage({ type: "late", id: message.id, cancelled: true });
      return;
    }
    plan.steps[at++]();
    if (at < plan.steps.length) {
      setTimeout(step, 0);
      return;
    }
    current = null;
    const ears = plan.ears;
    self.postMessage(
      { type: "late", id: message.id, key: message.key, brir: ears, ms: performance.now() - started },
      [ears[0].buffer, ears[1].buffer]
    );
  };
  current = message.id;
  setTimeout(step, 0);
}

self.onmessage = (event) => {
  const message = event.data;
  switch (message.type) {
    case "decoder":
      decoder.setDecoder(message);
      return;
    case "cell":
    case "keep":
      decoder.keep(message);
      return;
    case "forget":
      decoder.forget(message.key);
      return;
    case "cancel":
      if (current === message.id) cancelled.add(message.id);
      return;
    case "render":
      if (message.cell !== undefined) renderEarly(message);
      else renderLate(message);
      return;
    default:
      return;
  }
};
