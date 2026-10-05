"""A source between the positions it was solved at: the weights of each of them.

The band under the crossover is solved from a few positions of a rail, and a
source on the move is somewhere between them. Its response there is a
weighted sum of the solved ones. Two neighbours weighted linearly is what
``docs/formats/scene-pack.md`` began with, and it needs a position every
8 cm to hold 1 kHz. The sampling theorem allows far more: at a frequency
``f`` the pressure over space holds no wavenumber above ``k = 2 pi f / c``,
so positions half a wavelength apart determine it, 17 cm at 1 kHz.

:func:`band_limited_weights` are the weights that use this. For a field
whose spectrum over space is flat inside the ball ``|k| <= k_f`` the
correlation of the pressure at two points ``r`` apart is
``sin(k_f r) / (k_f r)``, and the weights that leave the least expected
error at the target solve

    ``(G + noise I) w = g``,  ``G_ij = sinc(k_f |x_i - x_j|)``,
    ``g_i = sinc(k_f |x - x_i|)``.

They depend on the frequency: at 200 Hz the same positions are five times
closer in wavelengths than at 1 kHz and the weights spread over more of
them. They depend on distances alone, so the positions need not be on one
line: a corner of a rail, or the rungs of a seat's vertical rail, are read
the same way. ``noise`` is the share of a solved response that is not the
field (it keeps the weights from growing where the positions say nothing
new); ``margin`` widens the band a little past ``k_f`` for what the wave
grid's own dispersion adds; and ``floor_share`` sets the least band the
weights are made for, whatever the frequency, as a share of the band two
positions as far apart as the two round the source can hold: next to a
source the field varies over the distance to it, not over a wavelength,
and a 100 Hz weight made for a 3.4 m wave would average positions a metre
apart.

:func:`residual_db` is the error those weights leave on such a field, the
figure a plan can read before anything is solved. :func:`lagrange_weights`
is the polynomial through the same positions, kept for the comparison of
``experiments.w44_interpolation.rail_interpolation``; the measurement that
chose between them is ``docs/open-questions/rail-interpolation.md``.

Everything here is ``numpy`` on small arrays and reads no file.
"""

from __future__ import annotations

import numpy as np

from reverberate.spatial.translate import SOUND_SPEED_M_S

__all__ = [
    "FLOOR_SHARE",
    "KNOT_HZ",
    "MARGIN",
    "NOISE",
    "band_limited_weights",
    "knots_hz",
    "lagrange_weights",
    "nearest_samples",
    "residual_db",
]

#: The band the weights are made for, over ``2 pi f / c``.
MARGIN = 1.05
#: The share of a solved response's energy taken as not the field.
NOISE = 1e-4
#: The least band the weights are made for, over the band two positions as
#: far apart as the two round the source can hold, ``pi`` over their gap.
FLOOR_SHARE = 0.4
#: The frequencies a pack holds the weights at are this far apart, in Hz.
KNOT_HZ = 50.0


def knots_hz(limit_hz: float, step_hz: float = KNOT_HZ) -> np.ndarray:
    """The frequencies the weights are held at: every ``step_hz`` from zero past ``limit_hz``."""
    return np.arange(int(np.ceil(limit_hz / step_hz - 1e-9)) + 1, dtype=float) * step_hz


def _band(freqs_hz: np.ndarray, gap_m: np.ndarray, margin: float, floor_share: float) -> np.ndarray:
    """The band the weights are made for: ``[..., frequency]`` in radians a metre."""
    k = margin * 2.0 * np.pi * np.asarray(freqs_hz, dtype=float) / SOUND_SPEED_M_S
    floor = floor_share * np.pi / np.maximum(gap_m, 1e-6)
    return np.asarray(np.maximum(k, floor[..., None]))


def _system(
    nodes: np.ndarray,
    target: np.ndarray,
    freqs_hz: np.ndarray,
    valid: np.ndarray | None,
    margin: float,
    noise: float,
    floor_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=float)
    target = np.asarray(target, dtype=float)
    if nodes.shape[-1] != 3 or target.shape != nodes.shape[:-2] + (3,):
        raise ValueError("nodes are [..., slot, 3] and the target [..., 3]")
    between = np.linalg.norm(nodes[..., :, None, :] - nodes[..., None, :, :], axis=-1)
    to_target = np.linalg.norm(nodes - target[..., None, :], axis=-1)
    held = np.ones(nodes.shape[:-1], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    # The two positions round the source come first: their gap is the rail's pitch here.
    gap = between[..., 0, 1] if nodes.shape[-2] > 1 else np.full(nodes.shape[:-2], np.inf)
    gap = np.where(held[..., min(1, nodes.shape[-2] - 1)], gap, np.inf)
    k = _band(freqs_hz, gap, margin, floor_share)  # [..., f]
    # numpy's sinc is sin(pi x) / (pi x).
    gram = np.sinc(k[..., None, None] * between[..., None, :, :] / np.pi)  # [..., f, slot, slot]
    side = np.sinc(k[..., None] * to_target[..., None, :] / np.pi)  # [..., f, slot]
    both = held[..., None, :, None] & held[..., None, None, :]
    eye = np.eye(nodes.shape[-2])
    gram = np.where(both, gram, 0.0) + eye * np.where(held[..., None, :, None], noise, 1.0)
    side = np.where(held[..., None, :], side, 0.0)
    weights = np.linalg.solve(gram, side[..., None])[..., 0]
    return weights, gram - eye * noise * held[..., None, :, None], side


def band_limited_weights(
    nodes: np.ndarray,
    target: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    margin: float = MARGIN,
    noise: float = NOISE,
    floor_share: float = FLOOR_SHARE,
) -> np.ndarray:
    """The weight of each solved position at each frequency: ``[..., slot, frequency]``.

    ``nodes`` are the solved positions, ``[..., slot, 3]`` in metres, and
    ``target`` where the source is, ``[..., 3]``. A slot that ``valid`` marks
    false holds no position and gets no weight. A target on a position gets
    that position alone, to ``noise``.
    """
    weights, _, _ = _system(nodes, target, freqs_hz, valid, margin, noise, floor_share)
    return np.asarray(np.swapaxes(weights, -1, -2))


def residual_db(
    nodes: np.ndarray,
    target: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    margin: float = MARGIN,
    noise: float = NOISE,
    floor_share: float = FLOOR_SHARE,
) -> np.ndarray:
    """What the weights leave of a field flat in the band they are made for: ``[..., frequency]``.

    The expected energy of the error over the field's, in decibels: zero is
    no better than silence. It is the figure of the method on the field it
    assumes, not a bound on a real one.
    """
    weights, gram, side = _system(nodes, target, freqs_hz, valid, margin, noise, floor_share)
    left = (
        1.0
        - 2.0 * np.einsum("...s,...s->...", weights, side)
        + np.einsum("...s,...st,...t->...", weights, gram, weights)
    )
    return np.asarray(10.0 * np.log10(np.maximum(left, 1e-12)))


def lagrange_weights(offsets_m: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """The polynomial through the positions, read at the target: ``[..., slot]``.

    ``offsets_m`` are the positions along one line, the target at zero.
    """
    offsets = np.asarray(offsets_m, dtype=float)
    held = np.ones(offsets.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    slots = offsets.shape[-1]
    weights = np.where(held, 1.0, 0.0)
    for other in range(slots):
        ratio = np.ones(offsets.shape)
        gap = offsets - offsets[..., other : other + 1]
        use = held & held[..., other : other + 1] & (np.arange(slots) != other)
        np.divide(-offsets[..., other : other + 1], gap, out=ratio, where=use)
        weights = weights * ratio
    return np.asarray(weights)


def nearest_samples(
    arcs_m: np.ndarray, at_m: np.ndarray, count: int, *, lower: np.ndarray | None = None
) -> np.ndarray:
    """The ``count`` samples of a rail nearest each arc length: ``[..., count]``, ``-1`` to fill.

    The sample at or before the target comes first and the one after it
    second, as the two a linear reading takes (``lower`` names the first
    when the caller already holds it); the rest follow by distance along the
    rail, the earlier of two as far away first. Near an end the rail has
    none left on one side and the other side gives them all.
    """
    arcs = np.asarray(arcs_m, dtype=float)
    at = np.asarray(at_m, dtype=float)
    if arcs.ndim != 1 or arcs.size < 1 or count < 1:
        raise ValueError("a rail has samples, and a source reads at least one of them")
    if lower is None:
        lower = np.clip(np.searchsorted(arcs, at, side="right") - 1, 0, max(arcs.size - 2, 0))
    lower = np.asarray(lower, dtype=np.int64)
    found = np.full(at.shape + (count,), -1, dtype=np.int64)
    below, above = lower.copy(), lower + 1
    for slot in range(min(count, arcs.size)):
        if slot == 0:
            take_below = np.ones(at.shape, dtype=bool)
        elif slot == 1:
            take_below = np.zeros(at.shape, dtype=bool)
        else:
            low_gap = np.where(below >= 0, at - arcs[np.maximum(below, 0)], np.inf)
            high_gap = np.where(
                above < arcs.size, arcs[np.minimum(above, arcs.size - 1)] - at, np.inf
            )
            take_below = low_gap <= high_gap + 1e-9
        found[..., slot] = np.where(take_below, below, above)
        below = np.where(take_below, below - 1, below)
        above = np.where(take_below, above, above + 1)
    return found
