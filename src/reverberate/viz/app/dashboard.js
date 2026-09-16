/** The mirror's dashboard: the criteria at the listener's cell and over the storey.
 *
 * A run whose sources carry a mirror field also carry `metrics/<id>.json`,
 * written by `reverberate.mirror.stage`: the settings and targets of the
 * criteria, a summary over every point (medians, pass fractions), the
 * solver's own floor, and one record per point with the errors and the
 * verdicts. This panel shows the point the listener stands at, against the
 * targets, and the storey's summary beside it. Nothing is computed here;
 * the numbers are the machine's.
 */

//: The criteria in the order they are shown, with how their value is printed.
const ROWS = [
  ["recall", "recall", (v) => v.toFixed(2), "≥"],
  ["precision", "precision", (v) => v.toFixed(2), "≥"],
  ["time_error_s", "time", (v) => `${(v * 1000).toFixed(3)} ms`, "≤"],
  ["direction_error_deg", "direction", (v) => `${v.toFixed(1)}°`, "≤"],
  ["level_error_db", "level", (v) => `${v.toFixed(2)} dB`, "≤"],
  ["itd_error_s", "ITD", (v) => `${(v * 1e6).toFixed(0)} µs`, "≤"],
  ["ild_error_db", "ILD", (v) => `${v.toFixed(2)} dB`, "≤"],
  ["early_coherence_error", "early coherence", (v) => v.toFixed(3), "≤"],
  ["mixing_time_relative", "mixing time", (v) => `${(100 * v).toFixed(0)} %`, "≤"],
  ["t30_relative", "T30", (v) => `${(100 * v).toFixed(0)} %`, "≤"],
  ["edt_relative", "EDT", (v) => `${(100 * v).toFixed(0)} %`, "≤"],
  ["tail_colour_db", "tail colour", (v) => `${v.toFixed(1)} dB`, "≤"],
  ["order_energy_db", "order energy", (v) => `${v.toFixed(1)} dB`, "≤"],
  ["sector_energy_db", "sector energy", (v) => `${v.toFixed(1)} dB`, "≤"],
  ["late_coherence_error", "late coherence", (v) => v.toFixed(3), "≤"],
  ["seam_db", "seam", (v) => `${v.toFixed(1)} dB`, "≤"],
];

const TARGET_KEYS = {
  recall: "recall",
  precision: "precision",
  time_error_s: "time_error_s",
  direction_error_deg: "direction_error_deg",
  level_error_db: "level_error_db",
  itd_error_s: "itd_error_s",
  ild_error_db: "ild_error_db",
  early_coherence_error: "early_coherence_error",
  mixing_time_relative: "mixing_time_relative",
  t30_relative: "t30_relative",
  edt_relative: "edt_relative",
  tail_colour_db: "tail_colour_db",
  order_energy_db: "order_energy_db",
  sector_energy_db: "sector_energy_db",
  late_coherence_error: "late_coherence_error",
  seam_db: "seam_db",
};

export function createDashboard(root, { table, summary, caption }) {
  const metrics = new Map(); // source id -> parsed metrics json
  let shown = { id: null, point: null };

  function targetsOf(record) {
    return (record && record.criteria && record.criteria.targets) || {};
  }

  function renderSummary(record) {
    if (!record || !record.summary || !record.summary.points) {
      summary.textContent = "";
      return;
    }
    const s = record.summary;
    const passed = Object.values(s.pass_fraction || {});
    const mean = passed.length ? passed.reduce((a, b) => a + b, 0) / passed.length : 0;
    const floor = record.floor ? ` · solver floor recall ${record.floor.recall_median.toFixed(2)}` : "";
    summary.textContent =
      `${s.points} points · recall ${fmt(s.errors.recall)} · precision ${fmt(s.errors.precision)}` +
      ` · ${(100 * mean).toFixed(0)} % of criteria met${floor}`;
  }

  const fmt = (entry) => (entry && entry.median !== null && entry.median !== undefined ? entry.median.toFixed(2) : "—");

  function renderPoint(record, point) {
    table.replaceChildren();
    const row = record && record.points ? record.points[String(point)] : null;
    if (!row) {
      caption.textContent = record ? `no judgement at point ${point}` : "no mirror metrics for this source";
      return;
    }
    const targets = targetsOf(record);
    caption.textContent =
      `point ${point} · ${row.reference_reflections} reference, ${row.candidate_reflections} mirror reflections` +
      ` · ${row.passed}/${row.of} criteria` + (row.silent ? " · mirror silent here" : "");
    for (const [key, label, format, sense] of ROWS) {
      const value = row.errors ? row.errors[key] : null;
      const tr = document.createElement("tr");
      const ok = row.verdicts ? row.verdicts[key] : null;
      tr.className = ok === true ? "ok" : ok === false ? "bad" : "";
      const target = targets[TARGET_KEYS[key]];
      tr.innerHTML = `<td>${label}</td><td>${value === null || value === undefined ? "—" : format(value)}</td><td>${sense} ${target === undefined ? "" : format(target)}</td>`;
      table.append(tr);
    }
  }

  return {
    /** The metrics of a source, fetched once from the run's site. */
    async load(id, url) {
      if (!url) {
        metrics.set(id, null);
        return;
      }
      try {
        const record = await fetch(url).then((r) => (r.ok ? r.json() : null));
        metrics.set(id, record);
      } catch {
        metrics.set(id, null);
      }
      if (shown.id === id) this.show(id, shown.point);
    },
    /** The judgement at a point of a source's field; `point` is the field's position index. */
    show(id, point) {
      shown = { id, point };
      const record = metrics.get(id) || null;
      renderSummary(record);
      renderPoint(record, point);
      root.classList.toggle("hidden", !record);
    },
    has: (id) => Boolean(metrics.get(id)),
    clear() {
      shown = { id: null, point: null };
      table.replaceChildren();
      summary.textContent = "";
      caption.textContent = "";
    },
  };
}
