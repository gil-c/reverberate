/** The listener tab: the pose as six bounded numbers, each with a slider.
 *
 * Position bounds follow the apartment (its walkable box and its floor and
 * ceiling), angles and zoom are fixed. Heights are shown above the floor.
 */
const DEG = 180 / Math.PI;

const SPEC = [
  { key: "x", label: "x", unit: "m", step: 0.01, min: -10, max: 10, digits: 2 },
  { key: "z", label: "z", unit: "m", step: 0.01, min: -10, max: 10, digits: 2 },
  { key: "height", label: "height", unit: "m", step: 0.01, min: 0.3, max: 2.5, digits: 2 },
  { key: "yaw", label: "yaw", unit: "°", step: 0.5, min: 0, max: 360, digits: 1 },
  { key: "pitch", label: "pitch", unit: "°", step: 0.5, min: -85, max: 85, digits: 1 },
  { key: "fov", label: "zoom", unit: "fov", step: 1, min: 12, max: 70, digits: 0 },
];

export function createListenerTab(root, { onEdit }) {
  const rows = new Map();
  let floor = 0;
  for (const spec of SPEC) {
    const label = document.createElement("label");
    label.className = "field";
    const name = document.createElement("span");
    name.textContent = spec.label;
    const number = document.createElement("input");
    number.type = "number";
    number.step = spec.step;
    number.min = spec.min;
    number.max = spec.max;
    const unit = document.createElement("em");
    unit.textContent = spec.unit;
    const slide = document.createElement("input");
    slide.type = "range";
    slide.className = "slide";
    slide.step = spec.step;
    slide.min = spec.min;
    slide.max = spec.max;
    label.append(name, number, unit, slide);
    root.append(label);
    rows.set(spec.key, { spec, number, slide });
    const apply = (raw) => {
      const value = Number.parseFloat(raw);
      if (!Number.isFinite(value)) return;
      const clamped = Math.min(Number(number.max), Math.max(Number(number.min), value));
      if (spec.key === "height") onEdit({ y: floor + clamped });
      else if (spec.key === "yaw" || spec.key === "pitch") onEdit({ [spec.key]: clamped / DEG });
      else onEdit({ [spec.key]: clamped });
    };
    number.addEventListener("change", () => apply(number.value));
    slide.addEventListener("input", () => apply(slide.value));
  }

  const put = (key, value) => {
    const row = rows.get(key);
    if (document.activeElement === row.number) return;
    row.number.value = value.toFixed(row.spec.digits);
    if (document.activeElement !== row.slide) row.slide.value = value;
  };

  return {
    /** The apartment's box and floor: what x, z and height may range over. */
    setBounds({ x, z, height, floorHeight }) {
      floor = floorHeight;
      for (const [key, [min, max]] of Object.entries({ x, z, height })) {
        const row = rows.get(key);
        row.number.min = min.toFixed(2);
        row.number.max = max.toFixed(2);
        row.slide.min = min.toFixed(2);
        row.slide.max = max.toFixed(2);
      }
    },
    update(listener) {
      put("x", listener.x);
      put("z", listener.z);
      put("height", listener.y - floor);
      put("yaw", (((listener.yaw * DEG) % 360) + 360) % 360);
      put("pitch", listener.pitch * DEG);
      put("fov", listener.fov);
    },
  };
}
