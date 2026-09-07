"""W37: the source-receiver ellipsoid, priced on real geometry without a solve.

Roadmap 4.1 and 5.1. Keep only the geometry whose shortest source to surface to
receiver path is under a budget ``L``, and no artificial face can reach the
receiver before ``L / c``. With the foci on the source and the receiver this is
the tightest domain the criterion allows, and it beats the box B0 measured:
every boundary point satisfies ``|PS| + |PR| = L`` exactly, so the first
artefact arrives at ``L / c`` with no corner penalty.

**Two things this module gets right that are easy to get wrong.**

``bmin`` and ``bmax`` on a ``SceneSpec`` cannot shrink a domain. PFFDTD's
``RoomGeo`` takes the minimum of the mesh points and the supplied bound, so a
supplied box only ever grows. The ellipsoid must therefore come from culling
triangles, which ``scene_export.truncate`` already does, and the bounds do a
different job: they pin the culled grid onto the reference grid's lattice so
the two are comparable.

**The domain is the box of the triangles that survive, not the box of the
ellipsoid.** Wherever the spheroid pokes out of the room there is no geometry
for it to enclose, and the surviving-geometry box is smaller. Taking the
ellipsoid's own extent would over-price the trick and, worse, would suggest a
domain larger than the one the solver is actually given.

**The mesh must be refined before it can be culled.**
``scene_export.path_length_bound`` subtracts a triangle's longest edge, so it
is a genuine lower bound and is biased towards keeping. An apartment shell
arrives as a handful of triangles that can each span a whole storey, and
culling those removes nothing at all: one corner near the receiver keeps a
floor that reaches the far wall. ``scene_export.refine`` at a 0.25 m maximum
edge comes first, and a model exported without it will price as though the
trick did not work.

**The path budget is ``c T``, and that is measured rather than assumed.** B0's
own comparison puts the machine-precision departure at 1.004 and 0.992 of
``cut / c`` on its two cuts, against 1.739 and 1.718 of ``cut / (c sqrt 3)``.
The conservative stencil rule costs a factor of 1.73 in window for the same
domain, which section 5.1 already calls a bad trade outside that experiment.

This run spends nothing. It reports how much domain the trick removes and what
that is worth in cards and in money; whether the truncated response is exact
over its window is a separate claim that needs two solves, and it is not made
here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from reverberate import cost as cost_model
from reverberate.acoustics import SPEED_OF_SOUND
from reverberate.experiments.scene_export import path_length_bound

__all__ = [
    "DEFAULT_WINDOWS_MS",
    "Restriction",
    "build",
    "load_scene",
    "path_budget_m",
    "restrict",
]

#: Windows priced. The short end is around a small room's mixing time, where
#: the synthetic tail is meant to take over anyway; the long end is where the
#: ellipsoid has swallowed the whole room and buys nothing.
DEFAULT_WINDOWS_MS = (10.0, 15.0, 20.0, 30.0, 50.0, 100.0)


def path_budget_m(
    window_s: float,
    *,
    sound_speed_m_s: float = SPEED_OF_SOUND,
    conservative: bool = False,
) -> float:
    """Path length a domain must contain to be exact over ``window_s``.

    ``c T`` by default. ``conservative=True`` gives ``c sqrt(3) T``, the
    stencil-corner rule, which buys bit-exactness rather than machine
    precision at a factor of 1.73 in window for the same domain.

    **B0 measured which one applies.** Its two cuts departed from the
    untruncated reference at 1.004 and 0.992 of ``cut / c``, and the residual
    out to that point is float32 rounding at -135 dB peak. The stencil rule is
    a factor of 1.74 early, so using it as the planning rule pays 1.73 times
    the domain for 135 decibels nobody can hear.
    """
    if window_s < 0.0:
        raise ValueError(f"window must not be negative, got {window_s} s")
    if sound_speed_m_s <= 0.0:
        raise ValueError(f"sound speed must be positive, got {sound_speed_m_s} m/s")
    return float(sound_speed_m_s * window_s * (np.sqrt(3.0) if conservative else 1.0))


@dataclass(frozen=True)
class Restriction:
    """What the ellipsoid leaves behind, at one window."""

    window_ms: float
    cut_m: float
    span_m: tuple[float, float, float]
    triangles: int
    grid_points: int
    #: The whole scene, for comparison. Same fields, no culling.
    full_span_m: tuple[float, float, float]
    full_triangles: int
    full_grid_points: int
    cost: cost_model.SolveCost
    full_cost: cost_model.SolveCost

    @property
    def box_share(self) -> float:
        """Fraction of the untruncated bounding box the domain still needs."""
        full = float(np.prod(self.full_span_m))
        return float(np.prod(self.span_m) / full) if full > 0.0 else 1.0

    @property
    def saturated(self) -> bool:
        """True once the ellipsoid contains the whole scene and buys nothing."""
        return self.grid_points >= self.full_grid_points

    def record(self) -> dict[str, Any]:
        return {
            "window_ms": self.window_ms,
            "cut_m": self.cut_m,
            "span_m": list(self.span_m),
            "box_m3": float(np.prod(self.span_m)),
            "box_share": self.box_share,
            "triangles": self.triangles,
            "triangle_share": self.triangles / max(self.full_triangles, 1),
            "grid_points": self.grid_points,
            "point_share": self.grid_points / max(self.full_grid_points, 1),
            "saturated": self.saturated,
            "cards": self.cost.cards,
            "cards_full": self.full_cost.cards,
            "usd": self.cost.usd,
            "usd_full": self.full_cost.usd,
            "usd_per_hour_per_card": self.cost.usd_per_hour_per_card,
        }


def load_scene(model_json: Path) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Meshes, source and first receiver, from an exported PFFDTD model."""
    document = json.loads(model_json.read_text())
    meshes = [
        trimesh.Trimesh(
            vertices=np.asarray(group["pts"], dtype=float),
            faces=np.asarray(group["tris"], dtype=int),
            process=False,
        )
        for group in document["mats_hash"].values()
    ]
    sources = document.get("sources") or []
    receivers = document.get("receivers") or []
    if not sources or not receivers:
        raise ValueError(f"{model_json} carries no source and receiver to place the foci on")
    return (
        meshes,
        np.asarray(sources[0]["xyz"], dtype=float),
        np.asarray(receivers[0]["xyz"], dtype=float),
    )


def restrict(
    meshes: list[trimesh.Trimesh],
    src: np.ndarray,
    rec: np.ndarray,
    cut_m: float,
) -> tuple[np.ndarray, int]:
    """The bounding box and triangle count of the geometry inside the ellipsoid.

    Returns the box of what *survives*, which is what the solver would be
    given, and not the box of the ellipsoid itself.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    triangles = 0
    for mesh in meshes:
        if len(mesh.faces) == 0:
            continue
        keep = path_length_bound(mesh, src, rec) <= cut_m
        if not keep.any():
            continue
        vertices = np.asarray(mesh.vertices, dtype=float)[
            np.unique(np.asarray(mesh.faces, dtype=int)[keep])
        ]
        lo = np.minimum(lo, vertices.min(axis=0))
        hi = np.maximum(hi, vertices.max(axis=0))
        triangles += int(keep.sum())
    if not np.isfinite(lo).all():
        raise ValueError(
            f"a {cut_m:.3f} m path budget keeps no geometry at all: the source and the "
            "receiver are further apart than the budget allows"
        )
    return hi - lo, triangles


def build(
    model_json: Path,
    out: Path,
    *,
    fmax_hz: float,
    usd_per_hour_per_card: float,
    windows_ms: tuple[float, ...] = DEFAULT_WINDOWS_MS,
    ppw: float = 10.5,
    sound_speed_m_s: float = SPEED_OF_SOUND,
    conservative: bool = False,
) -> dict[str, Any]:
    """Price the ellipsoid over a sweep of windows, and write ``report.json``."""
    out.mkdir(parents=True, exist_ok=True)
    meshes, src, rec = load_scene(model_json)

    full_vertices = np.concatenate([np.asarray(mesh.vertices, dtype=float) for mesh in meshes])
    full_span = full_vertices.max(axis=0) - full_vertices.min(axis=0)
    full_triangles = sum(len(mesh.faces) for mesh in meshes)
    full_points = cost_model.grid_points_for(full_span.tolist(), fmax_hz, ppw=ppw)
    rate = cost_model.solver_rate_hz(fmax_hz, ppw=ppw)

    results: list[Restriction] = []
    for window_ms in windows_ms:
        window_s = window_ms / 1000.0
        cut = path_budget_m(window_s, sound_speed_m_s=sound_speed_m_s, conservative=conservative)
        span, triangles = restrict(meshes, src, rec, cut)
        points = min(cost_model.grid_points_for(span.tolist(), fmax_hz, ppw=ppw), full_points)
        results.append(
            Restriction(
                window_ms=window_ms,
                cut_m=cut,
                span_m=(float(span[0]), float(span[1]), float(span[2])),
                triangles=triangles,
                grid_points=points,
                full_span_m=(float(full_span[0]), float(full_span[1]), float(full_span[2])),
                full_triangles=full_triangles,
                full_grid_points=full_points,
                cost=cost_model.estimate(
                    points, window_s, rate, usd_per_hour_per_card=usd_per_hour_per_card
                ),
                full_cost=cost_model.estimate(
                    full_points, window_s, rate, usd_per_hour_per_card=usd_per_hour_per_card
                ),
            )
        )

    report: dict[str, Any] = {
        "run": out.name,
        "reference_run": None,
        "trick": (
            "Keep only the geometry whose shortest source to surface to receiver path is "
            "under c times the window, so no artificial face can reach the receiver inside "
            "it. It should be free because the wave equation is hyperbolic: geometry the "
            "sound cannot reach and return from within the window cannot influence the "
            "response over it."
        ),
        "not_claimed": (
            "This run prices the domain. Whether the truncated response is exact over its "
            "window is a separate claim that needs the full scene and the restricted scene "
            "solved on one grid, and it is not made here."
        ),
        "model_json": str(model_json),
        "scene_sha256": None,
        "fmax_hz": fmax_hz,
        "points_per_wavelength": ppw,
        "sound_speed_m_s": sound_speed_m_s,
        "path_budget_rule": "c sqrt(3) T" if conservative else "c T",
        "path_budget_evidence": (
            "B0's departure lands at 1.004 and 0.992 of cut/c on its two cuts, against "
            "1.739 and 1.718 of cut/(c sqrt 3)."
        ),
        "source_xyz": list(map(float, src)),
        "receiver_xyz": list(map(float, rec)),
        "direct_path_m": float(np.linalg.norm(rec - src)),
        "full_span_m": list(map(float, full_span)),
        "full_box_m3": float(np.prod(full_span)),
        "full_triangles": full_triangles,
        "full_grid_points": full_points,
        "windows": [result.record() for result in results],
        "omissions": [
            "one source and receiver pair: the saving depends on where the pair "
            "sits, and a pair near one wall restricts more than a pair on the "
            "diagonal",
            "no solve stands behind this, so the exactness of the truncation is "
            "cited from B0 rather than measured on this geometry",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Price the source-receiver ellipsoid.")
    parser.add_argument("--model", type=Path, required=True, help="an exported PFFDTD model JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fmax", type=float, default=16000.0)
    parser.add_argument("--usd-per-hour", type=float, required=True)
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="use the stencil-corner budget c sqrt(3) T instead of c T",
    )
    args = parser.parse_args(argv)

    report = build(
        args.model,
        args.out,
        fmax_hz=args.fmax,
        usd_per_hour_per_card=args.usd_per_hour,
        conservative=args.conservative,
    )

    print(
        f"{Path(report['model_json']).name} at {report['fmax_hz'] / 1000:g} kHz: "
        f"{report['full_box_m3']:.1f} m3 box, {report['full_triangles']:,} triangles, "
        f"{report['full_grid_points']:.3e} points"
    )
    print(
        f"source to receiver {report['direct_path_m']:.2f} m, "
        f"budget rule {report['path_budget_rule']}\n"
    )
    print(
        f"{'window':>7} {'cut':>7} {'box':>10} {'share':>7} {'points':>10} {'cards':>7} {'USD':>8}"
    )
    for window in report["windows"]:
        mark = "  saturated" if window["saturated"] else ""
        print(
            f"{window['window_ms']:>5.0f}ms {window['cut_m']:>6.2f}m {window['box_m3']:>9.1f} "
            f"{100 * window['box_share']:>6.1f}% {window['grid_points']:>10.3e} "
            f"{window['cards']:>3d}/{window['cards_full']:<3d} {window['usd']:>7.2f}{mark}"
        )
    print(f"\n{report['not_claimed']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
