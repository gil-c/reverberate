"""The receiver array the field is expanded from, snapped to the solver's own nodes.

Roadmap section 7.1 sizes an array from the microphone rule ``N <~ k r``, which
is a statement about a *microphone*: below ``k r = N`` the radial term
``j_n(k r)`` falls off so fast that the order disappears into the microphone's
noise. This solver has no microphone. Its floor is numerical, about -100 dB
(B0 measured the truncation residual at -106 dB in float32), so an order sitting
40 dB down is still there to be recovered. The array is therefore sized against
a *measured* conditioning limit rather than against the rule, and
:func:`reverberate.spatial.encode.conditioning` is what measures it.

**Two things do bind, and they pull in opposite directions.**

Small ``k r`` costs dynamic range: at 1 kHz and 16 cm, ``k r`` is 2.9 and order
7 sits 63 dB below order 0. That is affordable here and is why the outer shells
exist at all.

Large ``k r`` costs correctness: the expansion truncated at the fitted order no
longer describes the field on that shell, and what is left over folds into the
kept orders. At 16 kHz and 16 cm, ``k r`` is 47 and no affordable order
describes it. The answer is not a bigger array but **shell gating**: each
frequency is fitted from the shells whose ``k r`` it can still describe, so the
outer shells serve the bottom of the band and the inner cloud the top.

**Receivers are grid nodes, not points.** PFFDTD interpolates a receiver from
the eight nodes around it, which is exact for a field linear across the cell and
is not one at the top of the band: at 10.5 points per wavelength a wave read
halfway between two nodes loses ``cos(k h / 2)`` per axis, up to -1.2 dB in
three. That error is a smooth function of position, so it does not average out
over an array; it becomes a radius dependent gain, which is exactly the quantity
the encoder is fitting. So the array is built in node index space and every
receiver is a node, with the node's own coordinate carried into the fit. The
cost is that a shell is not exactly spherical, and the fit is told the true
radii rather than the nominal ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.wave.comms import Grid, engine_indices, nearest_node

__all__ = [
    "ArrayDesign",
    "check_clearance",
    "default_shells",
    "design_array",
    "fibonacci_directions",
]


@dataclass(frozen=True)
class ArrayDesign:
    """A set of grid nodes about one centre, and what each of them is.

    ``positions`` are the nodes' own coordinates in scene metres, not the
    positions asked for. ``offsets`` are those minus the centre, ``radii`` their
    lengths, and ``shell`` the index of the shell each was drawn for, kept so a
    report can say which shell a frequency was fitted from.
    """

    centre: np.ndarray
    positions: np.ndarray
    offsets: np.ndarray
    radii: np.ndarray
    shell: np.ndarray
    nominal_radii: tuple[float, ...]
    grid_step_m: float
    #: The listening position that was asked for. ``centre`` is the grid node
    #: nearest it, which is what the field is actually expanded about.
    requested_centre: np.ndarray | None = None

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def record(self) -> dict[str, Any]:
        """What the run's JSON has to carry to make the array reproducible."""
        shells = []
        for index, nominal in enumerate(self.nominal_radii):
            on_shell = self.radii[self.shell == index]
            if on_shell.size == 0:
                continue
            shells.append(
                {
                    "shell": index,
                    "nominal_radius_m": round(float(nominal), 5),
                    "nodes": int(on_shell.size),
                    "radius_min_m": round(float(on_shell.min()), 5),
                    "radius_max_m": round(float(on_shell.max()), 5),
                }
            )
        offset = (
            None
            if self.requested_centre is None
            else round(float(np.linalg.norm(self.centre - self.requested_centre)), 6)
        )
        return {
            "centre": [round(float(v), 5) for v in self.centre],
            "requested_centre": (
                None
                if self.requested_centre is None
                else [round(float(v), 5) for v in self.requested_centre]
            ),
            "centre_offset_m": offset,
            "nodes": self.count,
            "grid_step_m": self.grid_step_m,
            "outer_radius_m": round(float(self.radii.max()), 5),
            "shells": shells,
            "note": (
                "every receiver is a grid node, so there is no interpolation "
                "error; the radii are the nodes' own, not the nominal ones"
            ),
        }


def fibonacci_directions(count: int) -> np.ndarray:
    """``count`` near uniform directions on the sphere, deterministic.

    The spherical Fibonacci lattice. It is not a quadrature and carries no
    weights, which is all this needs: the encoder solves a least squares
    problem over the array and never integrates over it. Deterministic, so an
    array is reproducible from its parameters without a seed.
    """
    if count < 1:
        raise ValueError("a shell needs at least one direction")
    if count == 1:
        return np.array([[0.0, 0.0, 1.0]])
    index = np.arange(count) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    golden = np.pi * (1.0 + 5.0**0.5)
    phi = golden * index
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)


def default_shells(
    grid_step_m: float, *, outer_radius_m: float = 0.16, fit_order: int = 9
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Radii and node counts covering the band, from the grid step alone.

    The radii are geometric from about six grid steps to ``outer_radius_m``, so
    that ``k r`` is covered evenly on a log axis and every frequency has a shell
    whose ``k r`` sits in its usable range. The innermost is a few cells across
    because two nodes closer than a cell are the same measurement.

    Each shell carries ``1.4 (N + 1)^2`` nodes for the fitted order ``N``, the
    usual oversampling for a least squares fit on a lattice that is not a
    quadrature, with the shell's own ``k r`` never consulted: which orders a
    shell can support is a property of the frequency, not of the array, and it
    is decided per frequency by the gate rather than baked into the geometry.
    """
    inner = max(6.0 * grid_step_m, 0.01)
    if outer_radius_m <= inner:
        raise ValueError(f"outer radius {outer_radius_m} m is inside the innermost shell")
    count = 6
    radii = tuple(np.geomspace(inner, outer_radius_m, count))
    per_shell = int(np.ceil(1.4 * (fit_order + 1) ** 2))
    return radii, (per_shell,) * count


def design_array(
    centre: np.ndarray,
    grid: Grid,
    *,
    radii_m: tuple[float, ...] | None = None,
    counts: tuple[int, ...] | None = None,
    fit_order: int = 9,
    outer_radius_m: float = 0.16,
) -> ArrayDesign:
    """Snap a set of shells about ``centre`` onto grid nodes, deduplicated.

    A node drawn twice is one measurement, so duplicates are dropped and the
    shell that asked for it first keeps it. The centre node itself is always
    included: it is the only receiver whose ``j_0`` is one at every frequency.
    """
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (3,):
        raise ValueError(f"centre must be one xyz point, got shape {centre.shape}")
    if radii_m is None or counts is None:
        radii_m, counts = default_shells(grid.h, outer_radius_m=outer_radius_m, fit_order=fit_order)
    if len(radii_m) != len(counts):
        raise ValueError(f"{len(radii_m)} radii but {len(counts)} counts")

    seen: dict[int, tuple[np.ndarray, int]] = {}
    # **The expansion centre is a node, not the point that was asked for.** The
    # requested centre is a listening position and lands wherever it lands; the
    # node nearest it is up to half a cell diagonal away, 1.8 mm at the 16 kHz
    # step. Expanding about the node instead makes the centre receiver's radius
    # exactly zero, which turns the W channel into the pressure measured there
    # rather than something 18 dB away from it, and gives the whole encode a
    # free exact check. The offset is recorded, not swallowed.
    requested = centre
    centre_position, centre_index = nearest_node(centre, grid)
    centre = centre_position
    seen[centre_index] = (centre_position, 0)
    for shell, (radius, count) in enumerate(zip(radii_m, counts, strict=True)):
        for direction in fibonacci_directions(count):
            position, index = nearest_node(centre + radius * direction, grid)
            if index not in seen:
                seen[index] = (position, shell)

    indices = sorted(seen)
    positions = np.array([seen[i][0] for i in indices], dtype=float)
    shells = np.array([seen[i][1] for i in indices], dtype=int)
    offsets = positions - centre
    return ArrayDesign(
        centre=centre,
        positions=positions,
        offsets=offsets,
        radii=np.linalg.norm(offsets, axis=1),
        shell=shells,
        nominal_radii=tuple(float(r) for r in radii_m),
        grid_step_m=float(grid.h),
        requested_centre=np.asarray(requested, dtype=float),
    )


def check_clearance(
    design: ArrayDesign,
    entry_dir: Path | str,
    grid: Grid,
    *,
    margin_m: float = 0.02,
    chunk: int = 4_000_000,
) -> dict[str, Any]:
    """Refuse an array whose ball is not entirely in free air.

    ``write_comms`` already refuses a receiver *on* a boundary node, which is
    PFFDTD's own rule. This is the stronger statement the expansion needs: the
    interior expansion ``sum a_nm j_n(k r) Y_nm`` is a solution of the
    homogeneous Helmholtz equation on the whole ball, so a boundary anywhere
    inside it makes the model wrong rather than noisy, and the fit would return
    a plausible field that is not the one in the room.

    The ball's node indices are enumerated in the grid's own index space and
    intersected with ``bn_ixyz`` in chunks, because that array is a gigabyte at
    16 kHz.
    """
    entry_dir = Path(entry_dir)
    radius = float(design.radii.max()) + float(margin_m)
    axes = (grid.xv, grid.yv, grid.zv)
    ranges = []
    for j in range(3):
        low = int(np.searchsorted(axes[j], design.centre[j] - radius, side="left"))
        high = int(np.searchsorted(axes[j], design.centre[j] + radius, side="right"))
        if low <= 0 or high >= len(axes[j]):
            raise ValueError(
                f"the ball of {radius:.3f} m about {design.centre.round(3).tolist()} "
                f"reaches the edge of the grid on axis {j}"
            )
        ranges.append(np.arange(low, high, dtype=np.int64))

    nx, ny, nz = grid.shape
    # Built one x slice at a time. At the 16 kHz step a ball of 18 cm spans
    # 355 cells a side, and the three coordinate arrays of that box together
    # are a gigabyte, which is a great deal of memory to spend on a check.
    slices = []
    iy, iz = np.meshgrid(ranges[1], ranges[2], indexing="ij")
    plane = (axes[1][iy] - design.centre[1]) ** 2 + (axes[2][iz] - design.centre[2]) ** 2
    flat_plane = (iy * nz + iz).ravel()
    plane = plane.ravel()
    for index in ranges[0]:
        budget = radius**2 - (axes[0][index] - design.centre[0]) ** 2
        if budget < 0.0:
            continue
        keep = flat_plane[plane <= budget]
        if keep.size:
            slices.append(engine_indices(keep + index * (nz * ny), grid))
    ball = np.unique(np.concatenate(slices)) if slices else np.zeros(0, dtype=np.int64)

    hits = 0
    with h5py.File(entry_dir / "vox_out.h5", "r") as handle:
        boundary = handle["bn_ixyz"]
        total = int(boundary.shape[0])
        for start in range(0, total, chunk):
            block = np.asarray(boundary[start : start + chunk])
            # Searched the cheap way round: the boundary node list is tens of
            # millions of entries and the ball is fixed, so each block is
            # looked up in the ball rather than the other way about.
            position = np.searchsorted(ball, block)
            position = np.clip(position, 0, max(ball.size - 1, 0))
            hits += int(np.count_nonzero(ball[position] == block)) if ball.size else 0
    record = {
        "centre": [round(float(v), 5) for v in design.centre],
        "ball_radius_m": round(radius, 5),
        "ball_nodes": int(ball.size),
        "boundary_nodes_in_ball": hits,
        "boundary_nodes_total": total,
    }
    if hits:
        raise ValueError(
            f"{hits} boundary nodes lie inside the {radius:.3f} m ball about "
            f"{design.centre.round(3).tolist()}; the interior expansion needs free air, "
            "so move the centre rather than shrinking the array"
        )
    return record
