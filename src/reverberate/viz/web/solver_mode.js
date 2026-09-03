/**
 * The solver mode of the apartment viewer: what entered and left the wave run.
 *
 * The other two modes render the authored HSSD scene. This one renders the
 * serialised surface list the solver actually read, so a picture and a response
 * can be shown to be of the same object. It is a separate module because it is
 * the only mode that needs signal plots and audio, not because it is a separate
 * application: it shares the apartment selector, the camera and the canvas.
 *
 * No signal processing happens here. The envelopes, spectrograms and decay
 * curves arrive precomputed from the tested Python path; this file draws them.
 */

const escapeHtml = (value) =>
  String(value).replace(/[<&>]/g, (ch) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[ch]);

export async function loadRun(url) {
  const response = await fetch(`${url}/run.json`);
  if (!response.ok) throw new Error(`run payload ${response.status}`);
  const data = await response.json();
  data.baseUrl = url;
  return data;
}

/** Build the solver's surfaces, plus a marker per source and receiver. */
export function buildRunGroup(THREE, data) {
  const group = new THREE.Group();

  for (const entry of data.groups) {
    if (!entry.indices.length) continue;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(entry.positions), 3)
    );
    geometry.setIndex(entry.indices);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(...entry.colour.map((v) => v / 255)),
      // The solver's shell keeps its outward normals, so standing inside means
      // looking at the back of every face.
      side: THREE.DoubleSide,
      roughness: 1,
      metalness: 0,
      transparent: true,
      opacity: 0.92,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = entry.label;
    group.add(mesh);
  }

  const bounds = new THREE.Box3().setFromObject(group);
  const size = bounds.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z) || 1;
  const radius = Math.max(span * 0.012, 0.03);

  const SOURCE = 0xff6b4a;
  const RECEIVER = 0x4ac1ff;
  const PICKED = 0xffe066;
  const marker = (position, colour, isSource) => {
    const geometry = isSource
      ? new THREE.SphereGeometry(radius, 20, 14)
      : new THREE.BoxGeometry(radius * 1.6, radius * 1.6, radius * 1.6);
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: colour }));
    mesh.position.fromArray(position);
    group.add(mesh);
    return mesh;
  };
  const sources = data.sources.map((s) => marker(s.position, SOURCE, true));
  const receivers = data.receivers.map((r) => marker(r.position, RECEIVER, false));

  const highlight = (sourceIndex, receiverIndex) => {
    sources.forEach((m, i) => m.material.color.setHex(i === sourceIndex ? PICKED : SOURCE));
    receivers.forEach((m, i) => m.material.color.setHex(i === receiverIndex ? PICKED : RECEIVER));
  };

  return { group, highlight, bounds, receivers, sources };
}

// ------------------------------------------------------------------- plots
function fit(canvas) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, canvas.clientWidth * ratio);
  canvas.height = Math.max(1, canvas.clientHeight * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return [context, canvas.clientWidth, canvas.clientHeight];
}

function drawEnvelope(canvas, pairs) {
  const [c, w, h] = fit(canvas);
  c.clearRect(0, 0, w, h);
  const peak = Math.max(1e-9, ...pairs.map(([lo, hi]) => Math.max(Math.abs(lo), Math.abs(hi))));
  c.strokeStyle = "#2b3038";
  c.beginPath();
  c.moveTo(0, h / 2);
  c.lineTo(w, h / 2);
  c.stroke();
  c.strokeStyle = "#4ac1ff";
  c.beginPath();
  pairs.forEach(([lo, hi], i) => {
    const x = (i / Math.max(1, pairs.length - 1)) * w;
    c.moveTo(x, h / 2 - (hi / peak) * (h / 2 - 2));
    c.lineTo(x, h / 2 - (lo / peak) * (h / 2 - 2));
  });
  c.stroke();
}

function drawDecay(canvas, decay) {
  const [c, w, h] = fit(canvas);
  c.clearRect(0, 0, w, h);
  const { seconds, db } = decay;
  if (!seconds.length) return;
  const tMax = seconds[seconds.length - 1] || 1;
  const floor = -70;
  const x = (t) => (t / tMax) * w;
  const y = (v) => (1 - (Math.max(v, floor) - floor) / -floor) * (h - 4) + 2;
  c.strokeStyle = "#2b3038";
  c.fillStyle = "#6b7280";
  c.font = "10px system-ui";
  for (const level of [-10, -30, -60]) {
    c.beginPath();
    c.moveTo(0, y(level));
    c.lineTo(w, y(level));
    c.stroke();
    c.fillText(`${level} dB`, 3, y(level) - 2);
  }
  c.strokeStyle = "#8be08b";
  c.lineWidth = 1.5;
  c.beginPath();
  seconds.forEach((t, i) => (i ? c.lineTo(x(t), y(db[i])) : c.moveTo(x(t), y(db[i]))));
  c.stroke();
  c.lineWidth = 1;
}

function drawSpectrogram(canvas, spec) {
  const [c, w, h] = fit(canvas);
  c.clearRect(0, 0, w, h);
  const raw = atob(spec.data);
  const { bins, frames } = spec;
  const image = new ImageData(frames, bins);
  for (let row = 0; row < bins; row += 1) {
    for (let column = 0; column < frames; column += 1) {
      // Rows arrive low frequency first; flip so low sits at the bottom.
      const level = raw.charCodeAt(row * frames + column);
      const target = ((bins - 1 - row) * frames + column) * 4;
      // Blue through green to white: dark where there is nothing, bright where
      // the energy is, and monotonic in level throughout.
      image.data[target] = level < 128 ? level * 0.5 : (level - 128) * 2;
      image.data[target + 1] = level < 128 ? level * 1.6 : 205 + (level - 128) * 0.4;
      image.data[target + 2] = level < 128 ? 40 + level : 255 - (level - 128) * 0.2;
      image.data[target + 3] = 255;
    }
  }
  // Drawn at native size into an offscreen buffer, then scaled to the canvas,
  // so the browser does the interpolation rather than us.
  const buffer = new OffscreenCanvas(frames, bins);
  buffer.getContext("2d").putImageData(image, 0, 0);
  c.imageSmoothingEnabled = true;
  c.drawImage(buffer, 0, 0, w, h);
}

// ------------------------------------------------------------------- panel
/**
 * Fill `element` with the run's panel and return `show(index)`.
 *
 * `onSelect` is called with the chosen sample so the host can highlight the
 * markers, and `onStand` is offered as a button so the listener can put the
 * camera exactly where the receiver they are hearing was.
 */
/** The air the solver sealed off, as a table a reader can challenge.
 *
 * Sealing stops the simulation carrying sound through a region, so it is shown
 * rather than assumed. The frequency is what makes the row actionable: a cavity
 * of side L would have rung at c/2L had it been left rigid, and that is how a
 * 125 Hz boom announces itself before anyone pays for a solve.
 */
function sealedSection(sealed) {
  if (!sealed) return "";
  const rows = (sealed.interiors || [])
    .slice(0, 8)
    .map(
      (r) =>
        `<tr><td>${escapeHtml(r.owner)}</td><td>${r.volume_m3.toFixed(3)} m³</td>` +
        `<td>${r.first_mode_hz.toFixed(0)} Hz</td></tr>`
    )
    .join("");
  const unclosed = sealed.unclosed_bodies || [];
  return `
    <h2>Sealed air</h2>
    <p class="caption">There is no air inside a solid object, so the solver is
    made to carry none there. Left coupled it is a cavity with rigid walls and no
    absorption at all, and it rings: that is what put 3.6 s of decay into the
    125 Hz band of an earlier run. Total ${sealed.sealed_volume_m3.toFixed(3)} m³
    in ${(sealed.interiors || []).length} closed bodies.</p>
    <table><tr><th>body</th><th>volume</th><th>would have rung at</th></tr>${rows}</table>
    ${
      unclosed.length
        ? `<p class="note">${unclosed.length} bodies are not closed, so their
           inside cannot be told from their outside and nothing was sealed for
           them: ${escapeHtml(unclosed.slice(0, 6).join(", "))}</p>`
        : `<p class="caption">Every body is closed, so no interior was left
           undecided.</p>`
    }`;
}

export function renderRunPanel(element, data, { onSelect, onStand }) {
  const room = data.room;
  const theory = data.theory;
  const shell = data.theory_shell_only;

  const legend = data.groups
    .filter((g) => g.indices.length)
    .map(
      (g) =>
        `<span><i class="swatch" style="background: rgb(${g.colour.join(",")})"></i>` +
        // The number is on the label because four of the thirteen materials in
        // this room share an absorption, so the colour alone cannot separate
        // them and should not pretend to.
        `${escapeHtml(g.label)} ${g.absorption === null ? "α?" : "α " + g.absorption}</span>`
    )
    .join("");

  element.innerHTML = `
    <h1>${escapeHtml(data.run)}</h1>
    <p class="sub">scene ${escapeHtml(data.scene_id)}, room ${escapeHtml(data.room_name)}<br>
    mesh ${escapeHtml(data.scene_sha256.slice(0, 16))}…, grid ${escapeHtml(data.cache_key.slice(0, 12))}…</p>
    <p class="caption">The geometry drawn here is the serialised surface list the
    solver read, not the authored scene, so the picture and the responses are
    provably of the same object. That is why it looks blockier than the other
    two modes: this is the room the wave equation was solved in.</p>

    <h2>Room, measured on the solver's boundary</h2>
    <table>
      <tr><td>volume</td><td>${room.volume_m3.toFixed(2)} m³</td></tr>
      <tr><td>boundary the field meets</td><td>${room.surface_area_m2.toFixed(1)} m²</td></tr>
      <tr><td>shell alone</td><td>${room.shell_area_m2.toFixed(1)} m²</td></tr>
      <tr><td>mean absorption</td><td>${room.mean_absorption.toFixed(3)}</td></tr>
      <tr><td>first axial mode</td><td>${room.first_axial_mode_hz.toFixed(1)} Hz</td></tr>
      <tr><td>Sabine, full boundary</td><td>${theory.sabine_rt60_s.toFixed(2)} s</td></tr>
      ${shell ? `<tr><td>Sabine, shell only</td><td>${shell.sabine_rt60_s.toFixed(2)} s</td></tr>` : ""}
    </table>
    <p class="caption">${escapeHtml(data.theory_note || "")}</p>

    <h2>Surfaces</h2>
    <p class="caption">Coloured by absorption at 1 kHz, red reflective through to
    blue absorbent. Grey means the material carried no measured coefficient.</p>
    <div class="legend">${legend}</div>

    ${sealedSection(data.sealed)}

    <h2>Sample</h2>
    <select id="run-pick">${data.samples
      .map((s, i) => `<option value="${i}">${escapeHtml(s.label)}</option>`)
      .join("")}</select>
    <button id="run-stand" style="margin-top:6px">Stand at this receiver</button>

    <h2>Impulse response</h2>
    <canvas class="plot" id="run-wave"></canvas>
    <p class="caption" id="run-wave-caption"></p>

    <h2>Spectrogram</h2>
    <canvas class="plot" id="run-spec" style="height: 132px"></canvas>
    <p class="caption" id="run-spec-caption"></p>

    <h2>Energy decay</h2>
    <canvas class="plot" id="run-edc"></canvas>
    <p class="caption">Schroeder backward integration, broadband, low cut at
    ${data.low_cut_hz ?? "?"} Hz.</p>

    <h2>Per band</h2>
    <table id="run-bands"></table>
    <p class="caption">${escapeHtml(data.band_note || "")}</p>

    <h2>Listen</h2>
    <p class="caption">dry: ${escapeHtml(data.dry_voice.member_name)},
    ${escapeHtml(data.dry_voice.licence)}${data.dry_voice.anechoic ? ", anechoic" : ""}</p>
    <audio id="run-dry" controls preload="none"></audio>
    <p class="caption" style="margin-top:.6rem">wet: the same voice through this response</p>
    <audio id="run-wet" controls preload="none"></audio>

    <h2>What this is not</h2>
    <p class="note">${escapeHtml(data.binaural_note)}</p>
    <ul class="caption">${(data.omissions || [])
      .map((o) => `<li>${escapeHtml(o)}</li>`)
      .join("")}</ul>
    ${
      data.measured_anomaly
        // The heading follows the record rather than asserting either state:
        // it said "Open question" for as long as the anomaly was closed, which
        // is the way a page quietly goes stale.
        ? `<h2>${
            String(data.measured_anomaly.status || "").startsWith("closed")
              ? "A defect that was found and fixed"
              : "Open question"
          }</h2>
      <p class="note">${escapeHtml(data.measured_anomaly.what)}</p>
      <p class="caption">${escapeHtml(data.measured_anomaly.status)}</p>
      ${
        data.measured_anomaly.cause
          ? `<p class="caption"><b>Cause.</b> ${escapeHtml(data.measured_anomaly.cause)}</p>`
          : ""
      }
      ${
        data.measured_anomaly.fix
          ? `<p class="caption"><b>Fix.</b> ${escapeHtml(data.measured_anomaly.fix)}</p>`
          : ""
      }`
        : ""
    }
  `;

  const byId = (id) => element.querySelector(`#${id}`);
  const dry = byId("run-dry");
  if (data.dry_audio) dry.src = `${data.baseUrl}/${data.dry_audio}`;
  else dry.hidden = true;
  const wet = byId("run-wet");
  const pick = byId("run-pick");

  function show(index) {
    const sample = data.samples[index];
    if (!sample) return;
    onSelect?.(sample);
    drawEnvelope(byId("run-wave"), sample.envelope);
    drawSpectrogram(byId("run-spec"), sample.spectrogram);
    drawDecay(byId("run-edc"), sample.decay);
    byId("run-spec-caption").textContent =
      `0 to ${(sample.spectrogram.max_hz / 1000).toFixed(1)} kHz, ` +
      `${sample.spectrogram.range_db} dB range, scaled to this response's own peak ` +
      `so two samples are not comparable by eye`;
    byId("run-wave-caption").textContent =
      `${sample.seconds.toFixed(2)} s at ${(sample.sample_rate_hz / 1000).toFixed(1)} kHz, ` +
      `peak ${sample.peak}`;
    const m = sample.measures;
    byId("run-bands").innerHTML = m
      ? `<tr><th>band</th><th>RT60</th><th>EDT</th><th>C50</th><th>DRR</th></tr>` +
        m.bands_hz
          .map((band, i) => {
            const out = m.in_band && !m.in_band[i] ? ' class="out"' : "";
            const cell = (v, unit) => `<td${out}>${v === null ? "n/a" : v.toFixed(2) + unit}</td>`;
            return (
              `<tr><td${out}>${band} Hz${out ? " ✗" : ""}</td>` +
              cell(m.rt60_s[i], " s") +
              cell(m.edt_s[i], " s") +
              cell(m.c50_db[i], " dB") +
              cell(m.drr_db[i], " dB") +
              "</tr>"
            );
          })
          .join("")
      : "";
    if (sample.wet_audio) {
      wet.src = `${data.baseUrl}/${sample.wet_audio}`;
      wet.hidden = false;
    } else {
      wet.removeAttribute("src");
      wet.hidden = true;
    }
  }

  pick.addEventListener("change", () => show(Number(pick.value)));
  byId("run-stand").addEventListener("click", () => {
    const sample = data.samples[Number(pick.value)];
    if (sample) onStand?.(sample);
  });
  show(0);
  return show;
}
