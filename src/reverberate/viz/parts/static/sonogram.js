/** Sonograms one under the other on one time axis, or one less another.
 *
 *   const view = createSonogram(element, { player });
 *   view.show([{ id, label }, ...]);            // one panel an item, the same scale
 *   view.show([{ id, label }, ...], { less: 0 }); // each item less the first, in dB
 *
 * The table is the server's (`api/sonogram`, `reverberate.viz.parts.sonogram`):
 * a column every 10 ms, twelve bands an octave from 50 Hz to 16 kHz, dB re
 * full scale. **The scale does not move**: `LEVEL_DB` for a level and
 * `DIFFERENCE_DB` for a difference, whatever the items hold, so that a colour
 * is a level on every panel and from one day to the next. A cell of a
 * difference that holds nothing in either item is left grey.
 *
 * With a `player` the cursor follows it; a click moves it and a drag sets the
 * region its loop turns in.
 */
import { attachRegion } from "./transport.js";

//: The scale of a level, dB re full scale at the page's level: from black to the brightest.
export const LEVEL_DB = [-110, -30];
//: The scale of a difference, dB: blue where the item is under the other, red where over.
export const DIFFERENCE_DB = 12;

// Dark to bright through violet and orange: a level's colour, in sixteen stops.
const STOPS = [
  [0, 0, 4], [12, 8, 38], [35, 12, 76], [66, 10, 104], [96, 19, 110], [125, 30, 109], [155, 41, 100], [184, 55, 84],
  [210, 72, 66], [231, 96, 44], [245, 125, 21], [252, 157, 8], [250, 193, 39], [243, 229, 92], [249, 249, 160], [252, 255, 220],
];
const levelColour = (u) => {
  const at = Math.max(0, Math.min(1, u)) * (STOPS.length - 1);
  const k = Math.min(Math.floor(at), STOPS.length - 2);
  const f = at - k;
  return STOPS[k].map((value, i) => value + (STOPS[k + 1][i] - value) * f);
};
// Blue under, white at nothing, red over.
const differenceColour = (u) => {
  const v = Math.max(-1, Math.min(1, u));
  const far = v < 0 ? [33, 82, 172] : [190, 38, 38];
  const a = Math.abs(v) ** 0.8;
  return [246, 246, 246].map((value, i) => value + (far[i] - value) * a);
};

const hz = (f) => (f >= 1000 ? `${f / 1000} k` : `${f}`);

export function createSonogram(element, { api = "api", player = null } = {}) {
  element.classList.add("p-sonogram");
  let panels = [];
  let shown = 0;
  let duration = 0;

  async function table(query) {
    const response = await fetch(`${api}/sonogram?${new URLSearchParams(query)}`);
    if (!response.ok) throw new Error((await response.json()).error || `${response.status}`);
    const said = JSON.parse(response.headers.get("X-Sonogram"));
    return { ...said, values: new Float32Array(await response.arrayBuffer()) };
  }

  function paint(canvas, data, difference) {
    canvas.width = data.columns;
    canvas.height = data.bands;
    const context = canvas.getContext("2d");
    const image = context.createImageData(data.columns, data.bands);
    for (let column = 0; column < data.columns; column++) {
      for (let band = 0; band < data.bands; band++) {
        const value = data.values[column * data.bands + band];
        let colour;
        if (Number.isNaN(value)) colour = [70, 74, 82];
        else if (difference) colour = differenceColour(value / DIFFERENCE_DB);
        else colour = levelColour((value - LEVEL_DB[0]) / (LEVEL_DB[1] - LEVEL_DB[0]));
        // The lowest band at the bottom.
        const at = ((data.bands - 1 - band) * data.columns + column) * 4;
        image.data[at] = colour[0];
        image.data[at + 1] = colour[1];
        image.data[at + 2] = colour[2];
        image.data[at + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }

  function legend(difference) {
    const bar = document.createElement("canvas");
    bar.width = 256;
    bar.height = 1;
    const context = bar.getContext("2d");
    for (let x = 0; x < 256; x++) {
      const colour = difference ? differenceColour((x / 255) * 2 - 1) : levelColour(x / 255);
      context.fillStyle = `rgb(${colour.map(Math.round).join(",")})`;
      context.fillRect(x, 0, 1, 1);
    }
    const row = document.createElement("div");
    row.className = "p-sono-legend";
    const low = difference ? `-${DIFFERENCE_DB} dB` : `${LEVEL_DB[0]} dB`;
    const high = difference ? `+${DIFFERENCE_DB} dB` : `${LEVEL_DB[1]} dB re full scale`;
    row.append(low, bar, high);
    if (difference) row.append("  grey: nothing in either");
    return row;
  }

  function axis(data) {
    const scale = document.createElement("div");
    scale.className = "p-sono-axis";
    const octaves = Math.log2(data.high_hz / data.low_hz);
    for (const f of [100, 250, 500, 1000, 2000, 4000, 8000]) {
      const tick = document.createElement("span");
      tick.textContent = hz(f);
      tick.style.bottom = `${(100 * Math.log2(f / data.low_hz)) / octaves}%`;
      scale.append(tick);
    }
    return scale;
  }

  function setCursor(seconds) {
    const left = duration ? `${(100 * seconds) / duration}%` : "0";
    for (const panel of panels) panel.cursor.style.left = left;
  }

  function setRegion(region) {
    for (const panel of panels) {
      panel.region.hidden = !region || !duration;
      if (region && duration) {
        panel.region.style.left = `${(100 * region[0]) / duration}%`;
        panel.region.style.width = `${(100 * (region[1] - region[0])) / duration}%`;
      }
    }
  }

  if (player) {
    player.on("time", setCursor);
    player.on("state", (state) => setRegion(state.loop));
  }

  return {
    /** Draw `items` (`{ id, label }`); with `less`, each less the item at that index. */
    async show(items, { less = null } = {}) {
      const mine = ++shown;
      const wanted = less === null ? items : items.filter((_, index) => index !== less);
      let made;
      try {
        made = await Promise.all(
          wanted.map((item) => table(less === null ? { item: item.id } : { item: item.id, less: items[less].id }))
        );
      } catch (error) {
        if (mine === shown) element.textContent = error.message;
        return;
      }
      if (mine !== shown) return;
      element.textContent = "";
      panels = [];
      duration = made.length ? made[0].columns * made[0].hop_s : 0;
      wanted.forEach((item, index) => {
        const data = made[index];
        const panel = document.createElement("div");
        panel.className = "p-sono-panel";
        const label = document.createElement("div");
        label.className = "p-sono-label";
        label.textContent = less === null ? item.label : `${item.label} less ${items[less].label}`;
        const plot = document.createElement("div");
        plot.className = "p-sono-plot";
        const canvas = document.createElement("canvas");
        const cursor = document.createElement("div");
        cursor.className = "p-cursor";
        const region = document.createElement("div");
        region.className = "p-region";
        region.hidden = true;
        plot.append(canvas, region, cursor);
        paint(canvas, data, less !== null);
        if (player) {
          const secondsAt = (event) => {
            const box = plot.getBoundingClientRect();
            return Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)) * duration;
          };
          attachRegion(plot, secondsAt, player);
        }
        panel.append(label, axis(data), plot);
        element.append(panel);
        panels.push({ cursor, region });
      });
      element.append(legend(less !== null));
      if (player) {
        setCursor(player.time());
        setRegion(player.state().loop);
      }
    },
    clear() {
      shown += 1;
      element.textContent = "";
      panels = [];
    },
    setCursor,
  };
}
