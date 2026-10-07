/** A track list: for each source a solo, a mute and a fader, and the balance kept.
 *
 *   const tracks = createTrackList(element, { onBalance: (version) => player.setBalance(version) });
 *   tracks.setLevels(levels);   // the answer of api/levels, for the meters
 *   tracks.setTime(seconds);
 *
 * The balance is the server's (`api/balance`, `reverberate.viz.parts.balance`):
 * every change is sent there, saved at once in `balance.json`, and answered
 * with a version that a player asks its frames under, so the mix heard is the
 * one the file holds. "Export" hands the same document to the browser as a
 * download; "Reset" puts every fader back to 0 dB.
 *
 * A fader runs from -60 dB, where the source is silent, to +12 dB.
 */
const el = (tag, attributes = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
};

export function createTrackList(element, { api = "api", onBalance = () => {} } = {}) {
  element.classList.add("p-tracks");
  let tracks = [];
  let rows = {};
  let levels = null;
  let timer = 0;
  let saved = null;
  const listeners = [];

  const body = () => ({ sources: Object.fromEntries(tracks.map((t) => [t.id, { gain_db: t.gain_db, mute: t.mute, solo: t.solo }])) });

  function took(answer) {
    // The rows hold their track, and a fader may have moved again since this was asked: a
    // track the list has already is kept as it stands here.
    const known = Object.fromEntries(tracks.map((track) => [track.id, track]));
    tracks = answer.tracks.map((track) => known[track.id] || track);
    saved = answer.saved;
    onBalance(answer.version);
    listeners.forEach((fn) => fn(tracks));
  }

  function send() {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const response = await fetch(`${api}/balance`, { method: "POST", body: JSON.stringify(body()) });
      if (response.ok) took(await response.json());
      show();
    }, 120);
  }

  function show() {
    const soloed = tracks.some((t) => t.solo);
    for (const track of tracks) {
      const row = rows[track.id];
      if (!row) continue;
      row.solo.classList.toggle("on", track.solo);
      row.mute.classList.toggle("on", track.mute);
      row.fader.value = track.gain_db;
      row.said.textContent = track.gain_db <= -60 ? "off" : `${track.gain_db > 0 ? "+" : ""}${track.gain_db} dB`;
      row.line.classList.toggle("silent", track.mute || (soloed && !track.solo) || track.gain_db <= -60);
    }
    file.textContent = saved ? `saved in ${saved}` : "";
  }

  const file = el("span", { class: "p-note" });
  const exportButton = el("button", { type: "button", class: "p-button p-small", text: "Export", title: "Download the balance as JSON" });
  const reset = el("button", { type: "button", class: "p-button p-small", text: "Reset", title: "Every fader back to 0 dB, nothing muted" });
  const list = el("div", { class: "p-track-rows" });
  element.append(list, el("div", { class: "p-row" }, exportButton, reset, file));

  exportButton.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify({ schema: "reverberate.apps.balance", ...body() }, null, 1)], { type: "application/json" });
    const link = el("a", { href: URL.createObjectURL(blob), download: "balance.json" });
    link.click();
    URL.revokeObjectURL(link.href);
  });
  reset.addEventListener("click", () => {
    for (const track of tracks) Object.assign(track, { gain_db: 0, mute: false, solo: false });
    show();
    send();
  });

  function build() {
    list.textContent = "";
    rows = {};
    for (const track of tracks) {
      const chip = el("i", { class: `p-chip ${track.kind}` });
      chip.style.background = track.colour;
      const solo = el("button", { type: "button", class: "p-button p-small p-solo", text: "S", title: "Solo: only the soloed sources sound" });
      const mute = el("button", { type: "button", class: "p-button p-small p-mute", text: "M", title: "Mute" });
      const fader = el("input", { type: "range", min: "-60", max: "12", step: "1", class: "p-fader", title: "Volume, dB; at the bottom the source is silent" });
      const said = el("span", { class: "p-said" });
      const meter = el("i", { class: "p-meter" });
      const lane = el("canvas", { class: "p-lane", width: "1000", height: "24" });
      const cursor = el("i", { class: "p-cursor" });
      const line = el("div", { class: "p-track" }, chip, el("span", { class: "p-name", text: track.id }), solo, mute, fader, said, el("span", { class: "p-meter-box" }, meter), el("span", { class: "p-lane-box" }, lane, cursor));
      solo.addEventListener("click", () => {
        track.solo = !track.solo;
        show();
        send();
      });
      mute.addEventListener("click", () => {
        track.mute = !track.mute;
        show();
        send();
      });
      fader.addEventListener("input", () => {
        track.gain_db = Number(fader.value);
        show();
        send();
      });
      fader.addEventListener("dblclick", () => {
        track.gain_db = 0;
        show();
        send();
      });
      rows[track.id] = { line, solo, mute, fader, said, meter, lane, cursor };
      list.append(line);
    }
    show();
  }

  const ready = fetch(`${api}/balance`)
    .then((response) => response.json())
    .then((answer) => {
      took(answer);
      build();
    });

  return {
    ready,
    tracks: () => tracks,
    /** Called with the tracks whenever the balance changes. */
    onChange: (fn) => listeners.push(fn),
    /** The gain a track sounds at now, 0 when it is silent. */
    gainOf(id) {
      const track = tracks.find((t) => t.id === id);
      if (!track) return 1;
      const soloed = tracks.some((t) => t.solo);
      if (track.mute || (soloed && !track.solo) || track.gain_db <= -60) return 0;
      return 10 ** (track.gain_db / 20);
    },
    /** The answer of `api/levels`: the meters follow it, and each track's lane is drawn from it. */
    setLevels(answer) {
      levels = answer;
      element.classList.add("with-lanes");
      for (const track of tracks) {
        const values = levels.stems[track.id];
        const row = rows[track.id];
        if (!values || !row) continue;
        const context = row.lane.getContext("2d");
        const { width, height } = row.lane;
        context.clearRect(0, 0, width, height);
        context.fillStyle = track.colour;
        for (let x = 0; x < width; x++) {
          const from = Math.floor((x * values.length) / width);
          const to = Math.max(from + 1, Math.floor(((x + 1) * values.length) / width));
          let top = -160;
          for (let k = from; k < to; k++) top = Math.max(top, values[k]);
          // The meter's own scale: silent at -80 dB, full at -20.
          const tall = Math.max(0, Math.min(1, (top + 80) / 60)) * height;
          if (tall > 0) context.fillRect(x, height - tall, 1, tall);
        }
      }
    },
    /** Move the meters to `seconds`: each source's level there, under its fader. */
    setTime(seconds) {
      if (!levels) return;
      const at = Math.floor(seconds / levels.hop_s);
      for (const track of tracks) {
        const values = levels.stems[track.id];
        const row = rows[track.id];
        if (!values || !row) continue;
        const db = values[Math.min(at, values.length - 1)] + 20 * Math.log10(Math.max(this.gainOf(track.id), 1e-6));
        row.meter.style.width = `${Math.max(0, Math.min(100, ((db + 80) / 60) * 100))}%`;
        row.cursor.style.left = `${Math.min(100, (100 * seconds) / (values.length * levels.hop_s))}%`;
      }
    },
  };
}
