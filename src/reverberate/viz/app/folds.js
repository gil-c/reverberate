/** Foldable sections of the left panel, their state kept per browser. */
const KEY = "reverberate.folds";

export function setupFolds(root) {
  let closed = {};
  try {
    closed = JSON.parse(localStorage.getItem(KEY) || "{}");
  } catch {
    closed = {};
  }
  for (const fold of root.querySelectorAll(".fold")) {
    const name = fold.dataset.fold;
    fold.classList.toggle("closed", Boolean(closed[name]));
    const head = fold.querySelector(".fold-head");
    if (!head) continue;
    head.addEventListener("click", () => {
      fold.classList.toggle("closed");
      closed[name] = fold.classList.contains("closed");
      try {
        localStorage.setItem(KEY, JSON.stringify(closed));
      } catch {
        // Fine without.
      }
    });
  }
}
