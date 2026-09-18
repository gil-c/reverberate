/** The Web Audio graph: a voice per source, convolved with the listener's
 * binaural response, summed to the output.
 *
 * Each source has two convolutions. The `early` one carries the first part of
 * the cell's response, rotated to the head at every update: it is an
 * `AudioWorklet` of this page's own (`early.worklet.js`, `partitioned.js`),
 * which keeps one history of the input and applies every response to it, so
 * a new response sounds from its first sample as though it had always been
 * there. The `tail` carries the rest, delayed to where the early part hands
 * over, on a ring of the browser's `ConvolverNode`s.
 *
 * **Why the early part is not on `ConvolverNode`s any more.** A convolver
 * given a buffer has no past: it convolves only what arrives after, as if the
 * voice had been switched on at that instant, and each tap of the response
 * adds a small step as the input reaches it, heard at whatever gain the fade
 * in has reached. Replaced twenty-five times a second while the listener
 * moves, that was a faint crackle no shape or length of crossfade removed
 * (ADR 0013). The tail is replaced every few seconds at most, its taps are
 * a second of diffuse reverberation, and it stays where it was.
 *
 * **A new response takes over by a crossfade longer than the interval
 * between responses**, so one is always fading into the next and the filter
 * the listener hears never stops moving; `ring.js` schedules the gains, in
 * the worklet for the early part and here for the tail.
 *
 * **The time it takes the sound to reach the listener is not in either
 * convolution.** It is a delay line in front of them, which follows the
 * listener continuously, and every response is handed over already aligned
 * on its own direct arrival (`spatial.js`). Without that, walking a lattice
 * of cells forty centimetres apart steps the arrival by up to a millisecond
 * and a half at every cell, and two responses a millisecond apart comb
 * filter each other for the whole crossfade: measured at fourteen decibels
 * of level jerk while walking, against three with the delay line
 * (`tests/js/smoothness.mjs`). With it, the walk shifts the delay smoothly
 * instead, which is also what gives a walk towards a source its Doppler.
 */
import { createRing } from "./ring.js";

//: The crossfade between one early response and the next. Longer than the
//: 40 ms interval of `spatial.js`, so fades overlap rather than meet. Longer
//: still is smoother, but a crossfade is an average over the poses it spans:
//: 120 ms of it is seven and a half degrees behind a head turning at the
//: speed the arrow keys turn it, which is where the page already was.
export const FADE_S = 0.12;
//: The tail's own crossfade, longer than the early one. Giving a convolver a
//: buffer clears it, so the response coming in has to build its own
//: reverberation up from nothing while the one going out is cut off, and the
//: reverberation dips in between. A second of reverberation cannot be built
//: up in eighty milliseconds; spreading the handover over a fifth of a
//: second makes the dip shallow enough not to be heard. The tail is diffuse,
//: so nothing is lost by taking longer over it.
export const TAIL_FADE_S = 0.25;
//: Responses the early convolution holds at once, and convolvers in the tail
//: ring. Six early, because a slot is given a new response only once its
//: fade is over, and a 120 ms fade at the 40 ms interval keeps four or five
//: busy; see `ring.js`. The tail keeps two: it is swapped only when the cell
//: changes or the head settles, so its fades never overlap, and a convolver a
//: second and a half long is not free.
export const EARLY_SLOTS = 6;
export const TAIL_SLOTS = 2;
//: How long after a pose was taken the delay is asked to match it. Every
//: ramp lands at its own pose's time plus this, so the delay traces the
//: listener's real distance, late by a constant and nothing else; see
//: `setPropagation`. It has to cover the worst age a pose reaches on its way
//: here, which is one update interval of `spatial.js`.
const PROPAGATION_LEAD_S = 0.07;
//: The shortest and longest a ramp may take, whatever the arithmetic says.
const PROPAGATION_RAMP_S = [0.01, 0.25];
//: A change of delay larger than this was not walked: it is 60 cm of sound
//: path, and the delay line is never more than a lead's worth of walking
//: behind the listener, which is under 20 cm. It used to be a slope times
//: the ramp's length, and a pose that reached here late was given a ramp so
//: short that an ordinary step crossed it, and the delay line jumped -- a
//: click, once or twice every thirty seconds of walking, measured.
const TELEPORT_S = 0.6 / 343.2;
//: The longest propagation the delay line allows, seconds: a hundred metres,
//: well past any flat.
const MAX_PROPAGATION_S = 0.3;

/** How the propagation delay goes from `current` to `target` seconds, for a
 *  pose `age` seconds old: `{ jump: true }` to set it at once, or the
 *  `seconds` to ramp over. Pure, so it can be tested; see `setPropagation`. */
export function propagationRamp(current, target, age) {
  if (Math.abs(target - current) > TELEPORT_S) return { jump: true };
  return { seconds: Math.max(PROPAGATION_RAMP_S[0], Math.min(PROPAGATION_RAMP_S[1], PROPAGATION_LEAD_S - age)) };
}

export function createEngine() {
  let context = null;
  let master = null;
  let masterGain = 1;
  let worklet = null; // the early convolver's module, once added
  const sources = new Map();
  const voices = new Map(); // url -> Promise<AudioBuffer>

  function ensureContext() {
    if (!context) {
      context = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
      master = context.createGain();
      master.gain.value = masterGain;
      master.connect(context.destination);
      worklet = context.audioWorklet.addModule(new URL("./early.worklet.js", import.meta.url));
    }
    if (context.state === "suspended") context.resume();
    return context;
  }

  function ring(count, fade) {
    const ctx = ensureContext();
    const slots = Array.from({ length: count }, () => {
      const convolver = ctx.createConvolver();
      convolver.normalize = false;
      const gain = ctx.createGain();
      gain.gain.value = 0;
      convolver.connect(gain);
      return { convolver, gain, timer: null };
    });
    return { slots, fade, order: createRing(count, fade) };
  }

  function entry(id) {
    let source = sources.get(id);
    if (!source) {
      const ctx = ensureContext();
      const input = ctx.createGain();
      const volume = ctx.createGain();
      volume.gain.value = 0.8;
      // Everything the listener hears is this far behind the source, and
      // the responses behind it are aligned on their own direct arrival.
      const propagation = ctx.createDelay(MAX_PROPAGATION_S);
      const delay = ctx.createDelay(2.0);
      // The early convolution is a worklet node, which exists only once its
      // module has loaded; a response that arrives first waits for it.
      const early = { node: null, waiting: null };
      const tail = ring(TAIL_SLOTS, TAIL_FADE_S);
      input.connect(propagation);
      worklet.then(() => {
        early.node = new AudioWorkletNode(ctx, "early-convolver", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [2],
          channelCount: 1,
          channelCountMode: "explicit",
          processorOptions: { slots: EARLY_SLOTS, fade: FADE_S },
        });
        propagation.connect(early.node);
        early.node.connect(volume);
        if (early.waiting) send(early, early.waiting);
        early.waiting = null;
      });
      propagation.connect(delay);
      for (const slot of tail.slots) {
        delay.connect(slot.convolver);
        slot.gain.connect(volume);
      }
      volume.connect(master);
      source = { input, volume, propagation, delay, early, tail, player: null, voice: null, loop: true, playing: false };
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
    const now = ctx.currentTime;
    // The ring says which slot, and what every slot whose gain changes is to
    // do; it has scheduled them all, so nothing here reads a gain back.
    const { slot: index, curves } = target.order.take(now);
    const next = target.slots[index];
    const buffer = ctx.createBuffer(2, response[0].length, ctx.sampleRate);
    buffer.copyToChannel(response[0], 0);
    buffer.copyToChannel(response[1], 1);
    next.convolver.buffer = buffer;
    clearTimeout(next.timer);
    play(target, curves, index);
  }

  /** A message to a source's early convolver, or kept for it if it is not
   *  there yet: only the latest matters, so a later one replaces it. */
  function send(early, message) {
    if (!early.node) {
      early.waiting = message;
      return;
    }
    const transfer = message.filter ? [message.filter.re.buffer, message.filter.im.buffer] : [];
    early.node.port.postMessage(message, transfer);
  }

  /** Put the ring's curves on the gains, replacing what each was doing. */
  function play(target, curves, live = null) {
    for (const { slot: index, start, duration, samples } of curves) {
      const gain = target.slots[index].gain.gain;
      gain.cancelScheduledValues(start);
      gain.setValueCurveAtTime(samples, start, duration);
      if (index !== live) release(target, target.slots[index]);
    }
  }

  /** A convolver with a buffer keeps convolving at gain zero; without one it
   *  does nothing. Freed once its fade is over, unless it is live again. */
  function release(target, slot) {
    clearTimeout(slot.timer);
    slot.timer = setTimeout(() => {
      if (target.slots[target.order.live] !== slot) slot.convolver.buffer = null;
    }, target.fade * 1000 + 50);
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
    /** How long the sound takes to reach the listener at the pose taken at
     *  `poseAt`, in seconds on `performance.now`'s clock.
     *
     * Ramped rather than set, because a delay line that jumps clicks and one
     * that slides changes pitch -- which is what walking towards a source
     * does to a sound, and is wanted.
     *
     * **Each ramp lands at its own pose's time plus a constant, not a
     * constant after the call.** How fast this delay slides *is* the pitch
     * shift: it is the distance covered divided by the time the ramp is
     * given. The two have to be the same stretch of time, and they are not
     * unless the ramp is anchored to the pose. A pose reaches here late, by
     * anything from nothing to a whole update interval -- the scheduler holds
     * one while a decode is in flight and releases it when the worker answers
     * -- so a ramp always given the same length divides a varying distance by
     * a fixed time and the pitch warbles by more than the Doppler it is meant
     * to carry: plus or minus thirteen cents, measured on a straight walk,
     * against the eleven the walk earns. Anchored to the pose, consecutive
     * ramps end one pose interval apart, the slope is the walking speed
     * whatever the scheduler did, and the delay is simply late by the lead.
     */
    setPropagation(id, seconds, poseAt) {
      const source = entry(id);
      const ctx = ensureContext();
      const now = ctx.currentTime;
      const target = Math.max(0, Math.min(MAX_PROPAGATION_S, seconds));
      const time = source.propagation.delayTime;
      const current = time.value;
      const age = poseAt === undefined ? 0 : performance.now() / 1000 - poseAt;
      const ramp = propagationRamp(current, target, age);
      time.cancelScheduledValues(now);
      if (ramp.jump) {
        // Nobody walked that far. Somebody was moved, and sliding a delay
        // line across a room is a siren rather than a Doppler.
        time.setValueAtTime(target, now);
        return;
      }
      time.setValueAtTime(current, now);
      time.linearRampToValueAtTime(target, now + ramp.seconds);
    },
    /** A new early response for a source, as `partitioned.partitionFilter`
     *  returns it; its buffers are handed over, not copied. */
    setEarly(id, filter) {
      send(entry(id).early, { type: "filter", filter });
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
      send(source.early, { type: "silence" });
      play(source.tail, source.tail.order.silence(ctx.currentTime));
    },
  };
}
