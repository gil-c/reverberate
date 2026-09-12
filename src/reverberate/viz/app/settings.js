/** The page's adjustable numbers and switches, kept in localStorage.
 *
 * Every value has a default here; a stored one is taken only when it is a
 * finite number or a boolean, so a stale or hand-edited store cannot leave
 * the page with a setting it cannot use.
 */
const DEFAULTS = {
  // Where a whole response is split; the part before turns with the head.
  earlyMs: 150,
  // How long the head must be still before the late part is decoded for it.
  stillMs: 200,
  // Crossfade between the two parts of a response.
  fadeMs: 10,
  // Exact mode: after this long without moving, glide to the nearest cell.
  settleMs: 1000,
  // How long the glide takes.
  glideMs: 600,
  exact: false,
  showPoints: false,
  // Drag the picture with the cursor, as paper, instead of turning the view.
  grabDrag: false,
};
const KEY = "reverberate.settings";

export function loadSettings() {
  const out = { ...DEFAULTS };
  try {
    const stored = JSON.parse(localStorage.getItem(KEY) || "{}");
    for (const key of Object.keys(DEFAULTS)) {
      const value = stored[key];
      if (typeof DEFAULTS[key] === "boolean" ? typeof value === "boolean" : Number.isFinite(value) && value > 0) {
        out[key] = value;
      }
    }
  } catch {
    // No store, or not ours to read.
  }
  return out;
}

export function saveSettings(values) {
  try {
    localStorage.setItem(KEY, JSON.stringify(values));
  } catch {
    // Not worth telling anyone about.
  }
}

/** Bind the settings tab's inputs to a settings object; `onChange(key, value)`. */
export function bindSettings(root, values, onChange) {
  for (const input of root.querySelectorAll("[data-setting]")) {
    const key = input.dataset.setting;
    if (input.type === "checkbox") {
      input.checked = values[key];
      input.addEventListener("change", () => {
        values[key] = input.checked;
        saveSettings(values);
        onChange(key, values[key]);
      });
      continue;
    }
    input.value = values[key];
    input.addEventListener("change", () => {
      const value = Number.parseFloat(input.value);
      if (!Number.isFinite(value) || value <= 0) {
        input.value = values[key];
        return;
      }
      values[key] = value;
      saveSettings(values);
      onChange(key, value);
    });
  }
}
