/** The Web Audio graph: a voice per source, convolved with the listener's
 * binaural response, summed to the output.
 *
 * Each source has two convolver pairs. The `early` pair carries the first
 * part of the cell's response, rotated to the head at every update; the
 * `tail` pair carries the rest, delayed to where the early part hands over. Each pair is two `ConvolverNode`s, and a new response goes to the one
 * not playing, which then takes over by an equal-power crossfade: a response
 * swapped on a live convolver clicks, and a crossfade over a few tens of
 * milliseconds does not. The convolvers are the browser's own, which run the
 * partitioned FFT convolution on their own thread.
 */
const FADE_S = 0.05;
//: Raised-cosine crossfade: the incoming gain follows sin^2, the outgoing
//: cos^2, so two versions of one response sum to a constant, and both start
//: and end with zero slope, which is what keeps a swap inaudible.
const RISE = Float32Array.from({ length: 64 }, (_, i) => Math.sin((Math.PI / 2) * (i / 63)) ** 2);
const curve = (from, to) => RISE.map((r) => from + (to - from) * r);

export function createEngine() {
  let context = null;
  let master = null;
  let masterGain = 1;
  const sources = new Map();
  const voices = new Map(); // url -> Promise<AudioBuffer>

  function ensureContext() {
    if (!context) {
      context = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
      master = context.createGain();
      master.gain.value = masterGain;
      master.connect(context.destination);
    }
    if (context.state === "suspended") context.resume();
    return context;
  }

  function pair() {
    const ctx = ensureContext();
    const make = () => {
      const convolver = ctx.createConvolver();
      convolver.normalize = false;
      const gain = ctx.createGain();
      gain.gain.value = 0;
      convolver.connect(gain);
      return { convolver, gain };
    };
    return { a: make(), b: make(), live: null };
  }

  function entry(id) {
    let source = sources.get(id);
    if (!source) {
      const ctx = ensureContext();
      const input = ctx.createGain();
      const volume = ctx.createGain();
      volume.gain.value = 0.8;
      const delay = ctx.createDelay(2.0);
      const early = pair();
      const tail = pair();
      for (const slot of [early.a, early.b]) {
        input.connect(slot.convolver);
        slot.gain.connect(volume);
      }
      input.connect(delay);
      for (const slot of [tail.a, tail.b]) {
        delay.connect(slot.convolver);
        slot.gain.connect(volume);
      }
      volume.connect(master);
      source = { input, volume, delay, early, tail, player: null, voice: null, loop: true, playing: false };
      sources.set(id, source);
    }
    return source;
  }

  async function voiceBuffer(url) {
    const ctx = ensureContext();
    if (!voices.has(url)) {
      voices.set(
        url,
        fetch(url)
          .then((r) => r.arrayBuffer())
          .then((bytes) => ctx.decodeAudioData(bytes))
      );
    }
    return voices.get(url);
  }

  function swap(target, response) {
    const ctx = ensureContext();
    const buffer = ctx.createBuffer(2, response[0].length, ctx.sampleRate);
    buffer.copyToChannel(response[0], 0);
    buffer.copyToChannel(response[1], 1);
    const next = target.live === target.a ? target.b : target.a;
    const previous = target.live;
    next.convolver.buffer = buffer;
    const now = ctx.currentTime;
    next.gain.gain.cancelScheduledValues(now);
    next.gain.gain.setValueCurveAtTime(curve(next.gain.gain.value, 1), now, FADE_S);
    if (previous) {
      previous.gain.gain.cancelScheduledValues(now);
      previous.gain.gain.setValueCurveAtTime(curve(previous.gain.gain.value, 0), now, FADE_S);
      // A convolver with a buffer keeps convolving at gain zero; without one
      // it does nothing. Freed once the fade is over, unless it is live again.
      setTimeout(() => {
        if (target.live !== previous) previous.convolver.buffer = null;
      }, FADE_S * 1000 + 50);
    }
    target.live = next;
  }

  function stopPlayer(source) {
    if (source.player) {
      try {
        source.player.stop();
      } catch {
        // Already stopped.
      }
      source.player.disconnect();
      source.player = null;
    }
    source.playing = false;
  }

  return {
    get context() {
      return context;
    },
    get sampleRate() {
      return context ? context.sampleRate : 48000;
    },
    /** Point a source at a voice; restarts it if it was playing. */
    async setVoice(id, url) {
      const source = entry(id);
      source.voice = url;
      if (source.playing) {
        stopPlayer(source);
        await this.play(id);
      }
    },
    async play(id) {
      const source = entry(id);
      if (!source.voice) return false;
      const buffer = await voiceBuffer(source.voice);
      stopPlayer(source);
      const ctx = ensureContext();
      const player = ctx.createBufferSource();
      player.buffer = buffer;
      player.loop = source.loop;
      player.connect(source.input);
      player.onended = () => {
        if (source.player === player) source.playing = false;
      };
      player.start();
      source.player = player;
      source.playing = true;
      return true;
    },
    stop(id) {
      const source = sources.get(id);
      if (source) stopPlayer(source);
    },
    isPlaying: (id) => Boolean(sources.get(id) && sources.get(id).playing),
    setVolume(id, value) {
      entry(id).volume.gain.value = value;
    },
    /** Output level for every source at once, in decibels. */
    setMasterDb(db) {
      masterGain = Math.pow(10, db / 20);
      if (master) master.gain.value = masterGain;
    },
    setLoop(id, loop) {
      const source = entry(id);
      source.loop = loop;
      if (source.player) source.player.loop = loop;
    },
    /** A new early response for a source, two ears. */
    setEarly(id, response) {
      swap(entry(id).early, response);
    },
    /** A new tail for a source, delayed to `startSamples` after the early part begins. */
    setTail(id, response, startSamples) {
      const source = entry(id);
      source.delay.delayTime.value = startSamples / this.sampleRate;
      swap(source.tail, response);
    },
    silence(id) {
      const source = sources.get(id);
      if (!source) return;
      const ctx = ensureContext();
      for (const target of [source.early, source.tail]) {
        if (target.live) {
          target.live.gain.gain.cancelScheduledValues(ctx.currentTime);
          target.live.gain.gain.setValueCurveAtTime(
            curve(target.live.gain.gain.value, 0),
            ctx.currentTime,
            FADE_S
          );
          target.live = null;
        }
      }
    },
  };
}
