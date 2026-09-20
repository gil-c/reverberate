/** The early part's convolver, on the audio thread: the shell around
 * `partitioned.js`, which holds all the arithmetic.
 *
 * One mono input, the voice after the propagation delay; one stereo output,
 * the two ears. Filters arrive already partitioned and transformed, from the
 * decode worker through the engine. Only the newest to arrive before a block
 * is used, at the start of that block, by this thread's own clock; each is
 * copied in and its buffers are sent straight back, so the thread that
 * allocated them is the one that collects them.
 */
import { BLOCK, createPartitionedConvolver } from "./partitioned.js";

class EarlyConvolver extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { slots, fade } = options.processorOptions;
    // `sampleRate` and `currentTime` are the worklet scope's own.
    this.core = createPartitionedConvolver({ slots, fade, sampleRate });
    this.silent = new Float32Array(BLOCK);
    this.pending = null;
    this.silence = false;
    this.broken = false;
    this.port.onmessage = (event) => {
      const message = event.data;
      if (message.type === "filter") {
        if (this.pending) this.giveBack(this.pending);
        this.pending = message.filter;
        this.silence = false;
      } else if (message.type === "silence") {
        if (this.pending) this.giveBack(this.pending);
        this.pending = null;
        this.silence = true;
      }
    };
  }

  giveBack(filter) {
    this.port.postMessage({ type: "spent" }, [filter.re.buffer, filter.im.buffer]);
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (this.broken) return true;
    if (output[0].length !== BLOCK) {
      // A browser rendering in other blocks than 128: say so once, and stay
      // silent rather than stop, which would take the node's messages with it.
      this.broken = true;
      this.port.postMessage({ type: "error", message: `early convolver needs blocks of ${BLOCK}, got ${output[0].length}` });
      return true;
    }
    if (this.pending) {
      this.core.setFilter(this.pending, currentTime);
      this.giveBack(this.pending);
      this.pending = null;
    } else if (this.silence) {
      this.core.silence(currentTime);
      this.silence = false;
    }
    const channels = inputs[0];
    this.core.process(channels.length ? channels[0] : this.silent, output[0], output[1], currentTime);
    return true;
  }
}

registerProcessor("early-convolver", EarlyConvolver);
