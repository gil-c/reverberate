/** The two side panels: collapse to a rail, resize by the inner edge.
 *
 * Widths are kept in localStorage as a convenience for one browser; nothing
 * else depends on them and a cleared store just gives the defaults back.
 */
const MIN = 220;
const MAX = 560;
const RAIL = 28;
const DEFAULTS = { left: 300, right: 320 };

export function setupPanels(app, onResize) {
  const widths = { ...DEFAULTS };
  const collapsed = { left: false, right: false };
  try {
    const saved = JSON.parse(localStorage.getItem("reverberate.panels") || "{}");
    for (const side of ["left", "right"]) {
      if (Number.isFinite(saved[side])) widths[side] = Math.min(MAX, Math.max(MIN, saved[side]));
      if (typeof saved[`${side}Collapsed`] === "boolean") collapsed[side] = saved[`${side}Collapsed`];
    }
  } catch {
    // No storage, or none we may read: defaults are fine.
  }

  const apply = (side) => {
    const panel = document.getElementById(side);
    panel.classList.toggle("collapsed", collapsed[side]);
    app.style.setProperty(`--${side}`, `${collapsed[side] ? RAIL : widths[side]}px`);
    const button = panel.querySelector("[data-collapse]");
    const open = side === "left" ? "‹" : "›";
    const shut = side === "left" ? "›" : "‹";
    button.textContent = collapsed[side] ? shut : open;
    onResize();
  };
  const save = () => {
    try {
      localStorage.setItem(
        "reverberate.panels",
        JSON.stringify({
          left: widths.left,
          right: widths.right,
          leftCollapsed: collapsed.left,
          rightCollapsed: collapsed.right,
        })
      );
    } catch {
      // Not worth telling anyone about.
    }
  };

  for (const button of app.querySelectorAll("[data-collapse]")) {
    button.addEventListener("click", () => {
      const side = button.dataset.collapse;
      collapsed[side] = !collapsed[side];
      apply(side);
      save();
    });
  }
  for (const grip of app.querySelectorAll("[data-grip]")) {
    const side = grip.dataset.grip;
    grip.addEventListener("pointerdown", (event) => {
      grip.setPointerCapture(event.pointerId);
      grip.classList.add("active");
      const move = (ev) => {
        const raw = side === "left" ? ev.clientX : innerWidth - ev.clientX;
        widths[side] = Math.min(MAX, Math.max(MIN, raw));
        apply(side);
      };
      const up = () => {
        grip.removeEventListener("pointermove", move);
        grip.removeEventListener("pointerup", up);
        grip.classList.remove("active");
        save();
      };
      grip.addEventListener("pointermove", move);
      grip.addEventListener("pointerup", up);
    });
  }
  apply("left");
  apply("right");
}
