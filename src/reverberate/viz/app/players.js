/** The sources tab and the audio panel's player rows.
 *
 * A row per source that is on: its voice, play, volume, loop, and how far
 * the listener stands from it. A source without a field is drawn and listed
 * but cannot play, and its button says so by being disabled.
 */
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

export function createSourceList(root, { onToggle, onSelect }) {
  return {
    render(sources, selected) {
      root.replaceChildren(
        ...sources.map((source) => {
          const row = document.createElement("label");
          row.className = `srcrow${source.on ? "" : " off"}${source.id === selected ? " sel" : ""}`;
          const box = document.createElement("input");
          box.type = "checkbox";
          box.checked = source.on;
          box.addEventListener("change", () => onToggle(source.id, box.checked));
          const text = document.createElement("span");
          const name = document.createElement("span");
          name.className = "nm";
          name.textContent = source.name ? `${source.id} · ${source.name}` : source.id;
          const pos = document.createElement("span");
          pos.className = "pos";
          pos.textContent = source.position.map((v) => v.toFixed(2)).join(" ");
          text.append(name, document.createElement("br"), pos);
          const kind = document.createElement("span");
          kind.className = "kind";
          kind.textContent = source.directivity;
          row.append(box, text, kind);
          row.addEventListener("click", (event) => {
            if (event.target !== box) onSelect(source.id);
          });
          return row;
        })
      );
    },
  };
}

export function createPlayers(root, { onSelect, onPlay, onVolume, onLoop, onVoice, isPlaying }) {
  let voices = [];
  return {
    setVoices(list) {
      voices = list;
    },
    /** The listener moved: only the distances change, so only they are touched. */
    updateDistances(sources, listener) {
      const at = [listener.x, listener.y, listener.z];
      for (const row of root.querySelectorAll(".player")) {
        const source = sources.find((s) => s.id === row.dataset.id);
        if (source) row.querySelector(".ttl span").textContent = `${dist(source.position, at).toFixed(2)} m`;
      }
    },
    render(sources, selected, listener) {
      const at = [listener.x, listener.y, listener.z];
      root.replaceChildren(
        ...sources
          .filter((source) => source.on)
          .map((source) => {
            const row = document.createElement("div");
            row.className = `player${source.id === selected ? " sel" : ""}`;
            const playing = isPlaying(source.id);
            const audible = Boolean(source.field);
            row.innerHTML =
              `<button type="button" class="pb${playing ? " playing" : ""}" ${audible ? "" : "disabled"}>${playing ? "❚❚" : "▶"}</button>` +
              `<div class="ttl"><b></b><span></span></div>` +
              `<select class="voice"></select>` +
              `<div class="vol-box"><input class="vol" type="range" min="0" max="1" step="0.01" value="${source.volume}">` +
              `<div class="meter"><i></i></div></div>` +
              `<button type="button" class="loop${source.loop ? " on" : ""}">⟳</button>`;
            row.querySelector("b").textContent = source.id;
            row.querySelector(".ttl span").textContent = `${dist(source.position, at).toFixed(2)} m`;
            row.dataset.id = source.id;
            const select = row.querySelector(".voice");
            select.replaceChildren(
              ...voices.map((voice) => {
                const option = document.createElement("option");
                option.value = voice.url;
                option.textContent = voice.id;
                option.selected = voice.url === source.voice;
                return option;
              })
            );
            if (!voices.length) {
              const option = document.createElement("option");
              option.textContent = "no voice";
              select.append(option);
              select.disabled = true;
            }
            select.addEventListener("change", () => onVoice(source.id, select.value));
            select.addEventListener("click", (event) => event.stopPropagation());
            row.querySelector(".pb").addEventListener("click", (event) => {
              event.stopPropagation();
              onPlay(source.id, !playing);
            });
            row.querySelector(".loop").addEventListener("click", (event) => {
              event.stopPropagation();
              onLoop(source.id, !source.loop);
            });
            const volume = row.querySelector(".vol");
            volume.addEventListener("input", () => onVolume(source.id, Number(volume.value)));
            volume.addEventListener("click", (event) => event.stopPropagation());
            row.addEventListener("click", () => onSelect(source.id));
            return row;
          })
      );
    },
  };
}
