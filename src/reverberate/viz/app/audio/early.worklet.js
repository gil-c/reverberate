/** The early part's convolver, on the audio thread: the shell around
 * `partitioned.js`, which holds all the arithmetic.
 *
 * One mono input, the voice after the propagation delay; one stereo output,
 * the two ears. Filters arrive already partitioned and transformed, from the
 * decode worker through the engine, and are timed by this thread's own clock,
 * so a response takes over at the start of the block after it arrives.
 */
import { BLOCK, createPartitionedConvolver } from "./partitioned.js";

class EarlyConvolver extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { slots, fade } = options.processorOptions || {};
    // `sampleRate` and `currentTime` are the worklet scope's own.
    this.core = createPartitionedConvolver({ slots, fade, sampleRate });
    this.silent = new Float32Array(BLOCK);
    this.port.onmessage = (event) => {
      const message = event.data;
      if (message.type === "filter") this.core.setFilter(message.filter, currentTime);
      else if (message.type === "silence") this.core.silence(currentTime);
    };
  }

  process(inputs, outputs) {
    const channels = inputs[0];
    const input = channels && channels.length ? channels[0] : this.silent;
    const [left, right] = outputs[0];
    if (input.length !== BLOCK || left.length !== BLOCK) {
      throw new Error(`early-convolver works in blocks of ${BLOCK}, was given ${left.length}`);
    }
    this.core.process(input, left, right, currentTime);
    return true;
  }
}

registerProcessor("early-convolver", EarlyConvolver);
