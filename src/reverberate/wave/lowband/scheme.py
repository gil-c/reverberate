"""The two stencils the low band may be solved with, and what each costs on paper.

A wave solve's cost is its node updates: nodes times steps. Both go as the
points per wavelength a scheme needs to hold its dispersion at the top of
the band, so the scheme is the first thing that sets the price.

- **Cartesian, 7 points**, the present engine's: stable to a Courant number
  of ``1 / sqrt(3)``, and worst along the axes.
- **Face centred cubic, 13 points**: the nodes of a cubic grid whose index
  sum is even, each read with its twelve neighbours at ``h sqrt(2)``. Stable
  to a Courant number of 1, where it is exact along the cube's axes, and
  worst along the body diagonals. Half the nodes of the cubic grid it lives
  on, and a time step ``sqrt(3)`` times longer for the same ``h``.

Both are leapfrog schemes of the same two fields, so they share the engine's
boundary model, and PFFDTD carries both. The relations are

    Cartesian:  sin^2(w T / 2) = l^2 sum_axes sin^2(k_axis h / 2)
    FCC:        sin^2(w T / 2) = (l^2 / 4) (3 - cx cy - cy cz - cz cx),  c_a = cos(k_a h)

with ``T = l h / c``. :func:`velocity_error` solves either for ``|k|`` along
a set of directions; :func:`points_per_wavelength` turns an error budget
into the points a scheme needs; :func:`numbers` prices a solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import roots_legendre

__all__ = [
    "CARTESIAN",
    "FCC",
    "SCHEMES",
    "Scheme",
    "numbers",
    "points_per_wavelength",
    "scheme_of",
    "velocity_error",
    "wavenumber",
]

#: PFFDTD backs the Courant number off its limit to remove the Nyquist mode.
BACK_OFF = 0.999


@dataclass(frozen=True)
class Scheme:
    """One stencil: its neighbours, its stability limit and what a node stands for."""

    name: str
    #: Neighbours read by an update, in the engine's order: ``(dx, dy, dz)`` in cells.
    neighbours: tuple[tuple[int, int, int], ...]
    #: The Courant number at the stability limit.
    courant_limit: float
    #: The stencil's weight is ``laplacian * l^2``.
    laplacian: float
    #: Nodes per cell of the cubic grid the scheme's ``h`` names.
    nodes_per_cell: float
    #: Points per wavelength PFFDTD runs the scheme at.
    ppw: float

    @property
    def courant(self) -> float:
        return self.courant_limit * BACK_OFF

    @property
    def fcc(self) -> bool:
        return self.name == "fcc"

    def relation(self, k_vector: np.ndarray, grid_step_m: float, courant: float) -> np.ndarray:
        """``sin^2(w T / 2)`` for wave vectors ``[..., 3]``: the scheme's dispersion relation."""
        phase = np.asarray(k_vector, dtype=float) * grid_step_m
        if self.fcc:
            cx, cy, cz = (np.cos(phase[..., a]) for a in range(3))
            return np.asarray(courant**2 / 4.0 * (3.0 - cx * cy - cy * cz - cz * cx))
        return np.asarray(courant**2 * np.sum(np.sin(phase / 2.0) ** 2, axis=-1))


CARTESIAN = Scheme(
    name="cartesian",
    neighbours=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
    courant_limit=math.sqrt(1.0 / 3.0),
    laplacian=1.0,
    nodes_per_cell=1.0,
    ppw=10.5,
)

FCC = Scheme(
    name="fcc",
    neighbours=(
        (1, 1, 0),
        (-1, -1, 0),
        (0, 1, 1),
        (0, -1, -1),
        (1, 0, 1),
        (-1, 0, -1),
        (1, -1, 0),
        (-1, 1, 0),
        (0, 1, -1),
        (0, -1, 1),
        (1, 0, -1),
        (-1, 0, 1),
    ),
    courant_limit=1.0,
    laplacian=0.25,
    nodes_per_cell=0.5,
    ppw=7.7,
)

SCHEMES = {CARTESIAN.name: CARTESIAN, FCC.name: FCC}


def scheme_of(fcc_flag: int | bool) -> Scheme:
    """The scheme of an engine's ``fcc_flag``: 0 is Cartesian, 1 and 2 are FCC."""
    return FCC if int(fcc_flag) > 0 else CARTESIAN


def _directions(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors and quadrature weights: the encoder's own grid, plus the lattice's axes."""
    nodes, weights = roots_legendre(count)
    cos_theta = np.asarray(nodes, dtype=float)
    phi = 2.0 * np.pi * (np.arange(count) + 0.5) / count
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    unit = np.stack(
        [
            np.outer(sin_theta, np.cos(phi)).ravel(),
            np.outer(sin_theta, np.sin(phi)).ravel(),
            np.repeat(cos_theta, count),
        ],
        axis=1,
    )
    weight = np.repeat(np.asarray(weights, dtype=float), count) * (2.0 * np.pi / count)
    return unit, weight / weight.sum()


#: The directions a cubic lattice is extreme along: an axis, a face diagonal, a body diagonal.
_EXTREMES = np.array(
    [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0] / np.sqrt(2.0), [1.0, 1.0, 1.0] / np.sqrt(3.0)]
)


def _solve(
    scheme: Scheme,
    frequency: np.ndarray,
    unit: np.ndarray,
    grid_step_m: float,
    sound_speed_m_s: float,
    courant: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``|k|`` per frequency and direction by bisection, and which bins the scheme reaches."""
    ts = courant * grid_step_m / sound_speed_m_s
    target = np.sin(np.pi * frequency * ts) ** 2
    low = np.zeros((frequency.size, unit.shape[0]))
    # The relation is monotone in |k| up to the first axis reaching pi / h.
    high = np.full_like(low, np.pi / grid_step_m)
    top = scheme.relation(high[:, :, None] * unit[None, :, :], grid_step_m, courant)
    for _ in range(60):
        mid = 0.5 * (low + high)
        value = scheme.relation(mid[:, :, None] * unit[None, :, :], grid_step_m, courant)
        too_small = value < target[:, None]
        low = np.where(too_small, mid, low)
        high = np.where(too_small, high, mid)
    return 0.5 * (low + high), top > target[:, None]


def velocity_error(
    scheme: Scheme,
    ppw: float,
    *,
    courant: float | None = None,
    fraction: float = 1.0,
    directions: int = 24,
) -> dict[str, float]:
    """The phase velocity's relative error at ``fraction`` of ``fmax``: worst, and along what.

    ``ppw`` is points per wavelength at ``fmax`` on the cubic grid's ``h``,
    as PFFDTD counts them. Negative is slow. Read on the encoder's direction
    grid and on the lattice's three extreme directions.
    """
    lam = scheme.courant if courant is None else float(courant)
    unit = np.concatenate([_directions(directions)[0], _EXTREMES])
    # In units of h = 1 and c = 1: fmax is 1 / ppw.
    frequency = np.array([fraction / ppw])
    k, reached = _solve(scheme, frequency, unit, 1.0, 1.0, lam)
    ideal = 2.0 * np.pi * frequency[0]
    error = np.where(reached[0], ideal / k[0] - 1.0, -1.0)
    worst = int(np.argmax(np.abs(error)))
    return {
        "worst": float(error[worst]),
        "axis": float(error[-3]),
        "face_diagonal": float(error[-2]),
        "body_diagonal": float(error[-1]),
        "mean": float(np.mean(error[:-3])),
    }


def points_per_wavelength(scheme: Scheme, budget: float, *, courant: float | None = None) -> float:
    """The fewest points per wavelength that hold the worst velocity error within ``budget``."""
    low, high = 2.0, 60.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        if abs(velocity_error(scheme, mid, courant=courant, directions=8)["worst"]) > budget:
            low = mid
        else:
            high = mid
    return float(high)


def wavenumber(
    scheme: Scheme,
    frequency_hz: np.ndarray,
    grid_step_m: float,
    sound_speed_m_s: float,
    *,
    courant: float | None = None,
    directions: int = 24,
) -> np.ndarray:
    """The wavenumber the scheme propagates, averaged over direction: the encoder's ``k``.

    :func:`reverberate.spatial.encode.numerical_wavenumber` for either
    scheme, at the Courant number the engine runs it at. Above the scheme's
    cutoff the ideal value is returned.
    """
    lam = scheme.courant if courant is None else float(courant)
    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    unit, weight = _directions(directions)
    k, reached = _solve(scheme, frequency, unit, grid_step_m, sound_speed_m_s, lam)
    ideal = 2.0 * np.pi * frequency / sound_speed_m_s
    return np.asarray(np.where(reached.all(axis=1), k @ weight, ideal), dtype=float)


def numbers(
    scheme: Scheme,
    *,
    fmax_hz: float,
    duration_s: float,
    ppw: float | None = None,
    volume_m3: float | None = None,
    reference_nodes: float | None = None,
    sound_speed_m_s: float = 343.2,
) -> dict[str, Any]:
    """Step, rate, steps and nodes of a solve to ``fmax_hz`` over ``duration_s``.

    The nodes come from ``volume_m3`` (the grid's box) or, with
    ``reference_nodes``, from the nodes of the Cartesian grid at 10.5 points
    per wavelength over the same box.
    """
    points = scheme.ppw if ppw is None else float(ppw)
    h = sound_speed_m_s / (fmax_hz * points)
    ts = scheme.courant * h / sound_speed_m_s
    steps = math.ceil(duration_s / ts)
    record: dict[str, Any] = {
        "scheme": scheme.name,
        "ppw": points,
        "grid_step_m": h,
        "node_spacing_m": h * (math.sqrt(2.0) if scheme.fcc else 1.0),
        "sample_rate_hz": 1.0 / ts,
        "steps": steps,
        "velocity_error_at_fmax": velocity_error(scheme, points)["worst"],
    }
    if reference_nodes is not None:
        h_reference = sound_speed_m_s / (fmax_hz * CARTESIAN.ppw)
        record["nodes"] = float(reference_nodes) * scheme.nodes_per_cell * (h_reference / h) ** 3
    elif volume_m3 is not None:
        record["nodes"] = volume_m3 / h**3 * scheme.nodes_per_cell
    if "nodes" in record:
        record["node_updates"] = record["nodes"] * steps
    return record
