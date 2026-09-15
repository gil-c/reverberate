/** The plots' analysis of one cell, off the main thread: sixty-four channel
 * covariances over tens of thousands of samples take longer than a frame. */
import { analyse } from "./analysis.js";

self.onmessage = (event) => {
  const { id, channels, order, earlySamples } = event.data;
  const result = analyse(channels, order, earlySamples);
  self.postMessage({ id, ...result });
};
