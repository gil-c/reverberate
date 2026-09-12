/** The two plots of the selected source's response at the listener: a
 * spectrogram and the Schroeder decay. Both come from the worker with the
 * response and are only drawn here.
 */
export function createPlots(spectrogramCanvas, decayCanvas, captions) {
  const sg = spectrogramCanvas.getContext("2d");
  const dg = decayCanvas.getContext("2d");

  function clear() {
    sg.fillStyle = "#0e1217";
    sg.fillRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
    dg.fillStyle = "#0e1217";
    dg.fillRect(0, 0, decayCanvas.width, decayCanvas.height);
    captions.spectrogram.textContent = "";
    captions.decay.textContent = "";
  }

  function drawSpectrogram(spec, sampleRate, seconds) {
    const { frames, bins, levels } = spec;
    const image = new ImageData(frames, bins);
    for (let f = 0; f < frames; f++) {
      for (let b = 0; b < bins; b++) {
        const v = levels[f * bins + b] / 255;
        const at = ((bins - 1 - b) * frames + f) * 4;
        image.data[at] = Math.round(20 + 235 * Math.pow(v, 1.4));
        image.data[at + 1] = Math.round(30 + 200 * Math.pow(v, 2.2));
        image.data[at + 2] = Math.round(60 + 120 * v);
        image.data[at + 3] = 255;
      }
    }
    const off = new OffscreenCanvas(frames, bins);
    off.getContext("2d").putImageData(image, 0, 0);
    sg.imageSmoothingEnabled = true;
    sg.drawImage(off, 0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
    // Gridlines at 4, 8 and 16 kHz.
    const nyquist = sampleRate / 2;
    sg.strokeStyle = "rgba(216,222,230,.3)";
    sg.lineWidth = 1;
    for (const hz of [4000, 8000, 16000]) {
      const y = spectrogramCanvas.height * (1 - hz / nyquist);
      sg.beginPath();
      sg.moveTo(0, y);
      sg.lineTo(spectrogramCanvas.width, y);
      sg.stroke();
    }
    captions.spectrogram.textContent = `0 – ${seconds.toFixed(2)} s · 0 – ${(nyquist / 1000).toFixed(0)} kHz · 70 dB`;
  }

  function drawDecay(curve, seconds) {
    const w = decayCanvas.width;
    const h = decayCanvas.height;
    const range = 70;
    dg.fillStyle = "#0e1217";
    dg.fillRect(0, 0, w, h);
    dg.strokeStyle = "#262e38";
    dg.lineWidth = 1;
    for (const db of [10, 30, 60]) {
      const y = (h * db) / range;
      dg.beginPath();
      dg.moveTo(0, y);
      dg.lineTo(w, y);
      dg.stroke();
    }
    dg.strokeStyle = "#6fd18a";
    dg.lineWidth = 2;
    dg.beginPath();
    for (let i = 0; i < curve.length; i++) {
      const x = (i / (curve.length - 1)) * w;
      const y = Math.min(h, (-curve[i] / range) * h);
      i ? dg.lineTo(x, y) : dg.moveTo(x, y);
    }
    dg.stroke();
    // T30 from the -5 to -35 dB stretch, doubled, as the decay time.
    const at = (db) => {
      for (let i = 0; i < curve.length; i++) if (curve[i] <= db) return (i / curve.length) * seconds;
      return null;
    };
    const t5 = at(-5);
    const t35 = at(-35);
    const t30 = t5 !== null && t35 !== null ? 2 * (t35 - t5) : null;
    captions.decay.textContent =
      `0 – ${seconds.toFixed(2)} s · ${range} dB` + (t30 !== null ? ` · T30 ${t30.toFixed(2)} s` : "");
  }

  clear();
  return { clear, drawSpectrogram, drawDecay };
}
