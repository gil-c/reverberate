/** A scene's sound on the audio thread: the shell around `sound-decode.js`.
 *
 * No input; one stereo output, the two ears. Chunks of the engine's order 7
 * frames and the head's filters arrive as messages, their buffers
 * transferred, never copied. It answers with what the page's clock needs:
 * how many samples of the scene have sounded, and when.
 */
import { BLOCK, QUANTUM, createStreamDecoder } from "./sound-decode.js";

class SceneStream extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { channels } = options.processorOptions;
    this.core = createStreamDecoder({ channels });
    this.quanta = 0;
    this.was = "";
    this.broken = false;
    this.port.onmessage = (event) => {
      const message = event.data;
      if (message.type === "filters") {
        this.core.setFilters({ re: message.re, im: message.im });
      } else if (message.type === "start") {
        this.core.setTotal(message.total);
        this.core.start(message.sample, message.gen);
        // The count the page's clock restarts from: nothing of this stream has sounded.
        this.port.postMessage({ type: "started", gen: message.gen, played: this.core.state().played });
      } else if (message.type === "chunk") {
        this.core.push(message);
      } else if (message.type === "stop") {
        this.core.stop();
        this.say(true);
      }
    };
  }

  say(force) {
    const state = this.core.state();
    const now = `${state.running}${state.waiting}${state.starved}${state.ended}`;
    if (!force && now === this.was && this.quanta % (BLOCK / QUANTUM) !== 0) return;
    this.was = now;
    this.port.postMessage({ type: "state", ...state, at: currentTime });
    const spent = this.core.takeSpent();
    for (const filter of spent) {
      this.port.postMessage({ type: "spent", re: filter.re, im: filter.im }, [filter.re.buffer, filter.im.buffer]);
    }
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (this.broken) return true;
    if (output[0].length !== QUANTUM) {
      this.broken = true;
      this.port.postMessage({ type: "error", message: `the scene stream needs quanta of ${QUANTUM}, got ${output[0].length}` });
      return true;
    }
    this.core.process(output[0], output[1] || output[0]);
    this.quanta += 1;
    this.say(false);
    return true;
  }
}

registerProcessor("scene-stream", SceneStream);
