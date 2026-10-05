"""An ambisonic expansion read away from its centre, several fused into one, and which to read.

Under the crossover a listening cell holds the order 7 expansion of the wave
field about its centre, and a head that is not on a cell hears that expansion
moved to where the head is, or two cells' expansions fused
(``docs/formats/scene-pack.md``, ``low``). This module is that mathematics and
the rule that chooses the cells. It reads no file.

**One sign convention.** An offset is always ``d = target - centre``, in scene
coordinates and metres: from the expansion's centre to the point wanted. A
cell at ``p`` serving a head at ``l`` has the offset ``l - p``, which is the
``d_j`` of the pack format. Nothing here takes the opposite vector.

**The field model.** In the convention of ``numpy.fft.rfft``, a plane wave of
unit amplitude arriving from the unit direction ``s`` is ``exp(+i k s . x)``
at ``x`` and has the ambisonic coefficients ``Y_c(s)`` about the origin
(:mod:`reverberate.spatial.field`), so about a centre displaced by ``-d`` its
coefficients are ``Y_c(s) exp(-i k s . d)``. A field free of sources is a sum
of such waves, and the operators below are integrals over ``s`` taken on the
quadrature of :func:`reverberate.spatial.sh.quadrature`.

**Translation.** ``T(d) = (1 / 4 pi) integral Y(s) Y(s)' exp(+i k s . d) ds``
re-expands an order ``N`` expansion about the point ``d`` away, at order
``N``: the exact translation matrix of a regular expansion, cut to ``N`` on
both sides. Its first row is the pressure, ``i^n j_n(k d) Y_c(d / |d|)``
(:func:`translation_weights`). The cut is the whole of its error: output
degree ``n`` at distance ``d`` needs the input's degrees up to about
``n + k d``, and the input stops at ``N``. Measured on the dense line of
hssd_0076 (``docs/open-questions/low-band-translation.md``), with the source
1.2 m away or more: one cell read 0.20 m away, ``k d = 3.6`` at 1 kHz, is
within -21 dB on all 64 channels in the worst octave of the worst case, and
at 0.23 m it is not (-16 dB). The quadrature's own error is 1e-7 of the
operator at ``k d = 3.6`` and 6e-4 at 7.3, for :data:`QUADRATURE_DEGREE`.

**Fusion.** Several cells are read together by the minimum norm estimator
under a uniform prior over directions,
``G = Y' W A^H (A W A^H + lambda tr(A W A^H) / n I)^-1``, whose Gram matrix
depends on the vectors between the cells and not on the target:
:func:`fusion_inverse` factors it once per set of cells and
:func:`fusion_operator` then costs one product per target. With one cell it is
the translation divided by ``1 + lambda``. Two cells either side of the head
on one line hold -23 dB on all channels out to 0.33 m each, ``k d = 6.0``;
two on one side are worse than the nearer alone.

**Which cells.** Near the source an expansion's upper degrees grow as the
near field does and the cut costs far more: the same error needs the
distance read to stay under 0.15 of the cell's distance to the source
(0.25 for the pressure alone). :func:`serving_radius_m` is that rule and
:func:`choose_cells` applies it.

**Applied without being formed.** An operator is 64 by 64, or 64 by 128,
complex numbers a frequency, and a product with it is the cost of a render
that moves. Both operators are a sum over the quadrature's plane waves, so
:func:`apply_translation` and :func:`apply_fusion` spread the cells' fields
on the plane waves, turn each by its own phase and gather: the same
``T(f) b`` and ``G(f) [b_0; b_1]`` to rounding, in a twentieth of the time.
The signal engine reads a head through those two
(``reverberate.render.translate``).

**Arrays.** Every operator is computed in the array namespace of its
arguments, ``numpy`` or ``cupy``, or in ``xp`` when given; the harmonics of
the quadrature are evaluated once on the host.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import spherical_jn

from reverberate.spatial.sh import degrees_of, quadrature, real_sh, scene_to_ambisonic

__all__ = [
    "EXACT_UNDER_M",
    "FUSE_WITHIN_M",
    "MODE_EXACT",
    "MODE_FUSED",
    "MODE_INAUDIBLE",
    "MODE_TRANSLATED",
    "QUADRATURE_DEGREE",
    "REGULARISATION",
    "SOUND_SPEED_M_S",
    "TRANSLATE_WITHIN_M",
    "SOURCE_SHARE",
    "SURFACE_SHARE",
    "apply_fusion",
    "apply_translation",
    "cell_stride",
    "choose_cells",
    "clearance_m",
    "fusion_inverse",
    "fusion_operator",
    "fusion_weights",
    "kernels",
    "namespace_of",
    "pair_inverse",
    "serving_radius_m",
    "translation_matrices",
    "translation_operator",
    "translation_weights",
]

#: The speed of sound the wave fields are solved at, in m/s (20 C).
SOUND_SPEED_M_S = 343.2

#: Tikhonov term of the fusion, relative to the mean diagonal of its Gram matrix.
REGULARISATION = 1e-3

#: Degree the quadrature over directions integrates exactly: twice the order
#: for the two harmonics, and twelve for the exponential.
QUADRATURE_DEGREE = 26

#: A head nearer a cell's centre than this is on the cell, in metres.
EXACT_UNDER_M = 1e-3

#: How far one cell alone is read from, and how far each of two fused cells
#: is, in metres: what the dense line of hssd_0076 holds to -20 dB in the
#: worst octave under 1 kHz of its worst case, on all 64 channels, with the
#: source 1.2 m away or more (-21.1 dB at 0.196 m; -24.1 dB at 0.294 m).
TRANSLATE_WITHIN_M = 0.20
FUSE_WITHIN_M = 0.30

#: The share of a cell's distance to the source it may be read from: -22.6 dB
#: fused and -17.3 dB translated between 0.10 and 0.15, -13.0 dB over it.
SOURCE_SHARE = 0.15
#: The share of a cell's clearance to the nearest surface it may be read
#: from. Every cell of the line is 0.51 m or more from a surface and holds
#: at 0.294 m, a share of 0.58; nothing nearer a surface was measured.
SURFACE_SHARE = 0.5

#: The listener's modes of the pack format, ``low/mode``.
MODE_INAUDIBLE, MODE_EXACT, MODE_TRANSLATED, MODE_FUSED = 0, 1, 2, 3

#: Bytes of the largest intermediate one chunk of frequencies may take.
_CHUNK_BYTES = 256 << 20
#: Frequencies solved together in :func:`fusion_weights`.
_WEIGHT_CHUNK = 128


def namespace_of(*arrays: Any, xp: Any = None) -> Any:
    """The array library of ``arrays``: ``xp`` when given, ``cupy`` if one is its array."""
    if xp is not None:
        return xp
    for array in arrays:
        if type(array).__module__.split(".")[0] == "cupy":
            import cupy

            return cupy
    return np


@lru_cache(maxsize=8)
def _plane_waves(order: int, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directions in the scene frame, weights summing to one, harmonics ``[direction, channel]``."""
    directions, weights = quadrature(degree)
    # ``scene_to_ambisonic`` is a rotation ``R``, and ``s . R d = (R' s) . d``:
    # the quadrature is turned into the scene's frame once, so an offset is
    # never rotated on the device.
    rotation = scene_to_ambisonic(np.eye(3)).T
    return directions @ rotation, weights / weights.sum(), real_sh(order, directions)


def _on(xp: Any, array: np.ndarray, dtype: Any) -> Any:
    return xp.asarray(array, dtype=dtype)


def _wavenumbers(freqs_hz: Any, sound_speed_m_s: float, xp: Any) -> Any:
    return 2.0 * np.pi * xp.asarray(freqs_hz, dtype=xp.float64) / sound_speed_m_s


def translation_weights(
    offset_scene: np.ndarray,
    freqs_hz: np.ndarray,
    order: int,
    *,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
) -> np.ndarray:
    """What multiplies an expansion's coefficients to give the pressure at an offset.

    ``p(d) = sum_c i^n j_n(k |d|) Y_c(d / |d|) B_c``: the first row of
    :func:`translation_operator` in closed form, on the host.
    ``offset_scene`` is ``d``, from the expansion's centre to the point
    wanted. Returns ``[frequency, channel]``, complex64.
    """
    amb = scene_to_ambisonic(np.asarray(offset_scene, dtype=float)[None, :])[0]
    r = float(np.linalg.norm(amb))
    n = degrees_of(order)
    # At the centre only the omnidirectional channel is read, whatever the direction.
    basis = real_sh(order, (amb / r if r > 0.0 else np.array([1.0, 0.0, 0.0]))[None, :])[0]
    radial = spherical_jn(n[None, :], (2 * np.pi * freqs_hz / sound_speed_m_s * r)[:, None])
    weights: np.ndarray = ((1j**n)[None, :] * radial * basis[None, :]).astype(np.complex64)
    return weights


def translation_operator(
    offsets_scene: Any,
    freqs_hz: Any,
    order: int,
    *,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    quadrature_degree: int = QUADRATURE_DEGREE,
    xp: Any = None,
) -> Any:
    """The matrices that re-expand an order ``order`` expansion about each offset.

    ``offsets_scene`` is ``[..., 3]``, each from the expansion's centre to the
    new one. Returns ``[..., frequency, channel out, channel in]``, complex64:
    ``b_target(f) = T(f) b_centre(f)``, both ACN and N3D at ``order``.
    """
    xp = namespace_of(offsets_scene, freqs_hz, xp=xp)
    directions, weights, basis = _plane_waves(order, quadrature_degree)
    offsets = xp.asarray(offsets_scene, dtype=xp.float64)
    k = _wavenumbers(freqs_hz, sound_speed_m_s, xp)
    projection = offsets @ _on(xp, directions, xp.float64).T  # [..., direction]
    weighted = _on(xp, (basis * weights[:, None]).T, xp.complex128)  # [channel, direction]
    harmonics = _on(xp, basis, xp.complex128)  # [direction, channel]
    channels, count = weighted.shape
    lead = projection.shape[:-1]
    out = xp.zeros((*lead, int(k.shape[0]), channels, channels), dtype=xp.complex64)
    batch = max(1, int(np.prod(lead, dtype=np.int64)))
    step = max(1, _CHUNK_BYTES // (16 * batch * channels * count))
    for start in range(0, int(k.shape[0]), step):
        phase = xp.exp(1j * k[start : start + step, None] * projection[..., None, :])
        out[..., start : start + step, :, :] = (phase[..., None, :] * weighted) @ harmonics
    return out


@lru_cache(maxsize=4)
def _triple_products(order: int) -> np.ndarray:
    """``[c, a b]``: the mean over the sphere of ``Y_a Y_b Y_c``, ``c`` to degree ``2 order``.

    A product of two harmonics of degree ``order`` at most has no part above
    degree ``2 order``, and the quadrature takes the three exactly.
    """
    directions, weights = quadrature(4 * order)
    low = real_sh(order, directions)
    high = real_sh(2 * order, directions)
    found = np.einsum("q,qa,qb,qc->cab", weights / weights.sum(), low, low, high)
    flat: np.ndarray = np.ascontiguousarray(found.reshape(found.shape[0], -1))
    flat.setflags(write=False)
    return flat


def translation_matrices(
    offset_scene: np.ndarray,
    freqs_hz: np.ndarray,
    order: int,
    *,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
) -> np.ndarray:
    """:func:`translation_operator` in closed form, on the host.

    The plane wave's phase is its own expansion,
    ``exp(i k s . d) = sum_l i^l j_l(k |d|) sum_m Y_lm(s) Y_lm(d / |d|)``,
    and the integral of ``Y_a Y_b Y_lm`` over ``s`` is a table that depends
    on nothing (:func:`_triple_products`). The operator is then fifteen
    real matrices for the direction, weighed at each frequency by
    ``i^l j_l``: 6e4 products a frequency against the quadrature's 1.6e6,
    and no quadrature error (the quadrature's is 1e-7 of the operator at
    ``k d = 3.6`` and 6e-4 at 7.3). ``offset_scene`` is one offset or
    ``[offset, 3]``. Returns ``[frequency, channel out, channel in]``,
    complex64, with the offsets' axis before them when several were given.
    What the fast engine moves a head with.
    """
    offsets = np.asarray(offset_scene, dtype=float)
    single = offsets.ndim == 1
    offsets = offsets.reshape(-1, 3)
    freqs = np.asarray(freqs_hz, dtype=float)
    channels = (order + 1) ** 2
    amb = scene_to_ambisonic(offsets)
    r = np.linalg.norm(amb, axis=1)
    top = 2 * order
    # At the centre the direction is not read: every degree but the first is nothing.
    unit = np.where(r[:, None] > 0.0, amb / np.maximum(r, 1e-300)[:, None], [1.0, 0.0, 0.0])
    towards = real_sh(top, unit)
    products = _triple_products(order)
    n = np.arange(top + 1)
    turn = 1j**n
    made = np.empty((offsets.shape[0], freqs.size, channels * channels), dtype=np.complex64)
    for index in range(offsets.shape[0]):
        # Degree ``n`` is the channels ``n^2`` to ``(n + 1)^2`` of the ACN order.
        per_degree = np.stack(
            [
                # A sum of rows, taken a row at a time: a vector by a matrix is the
                # library's level 2, whose rounding moves with where its arrays lie.
                (
                    towards[index, d * d : (d + 1) * (d + 1), None]
                    * products[d * d : (d + 1) * (d + 1)]
                ).sum(axis=0)
                for d in range(top + 1)
            ]
        ).astype(np.float32)
        radial = turn[None, :] * spherical_jn(
            n[None, :], (2.0 * np.pi * freqs / sound_speed_m_s * r[index])[:, None]
        )
        made[index].real = radial.real.astype(np.float32) @ per_degree
        made[index].imag = radial.imag.astype(np.float32) @ per_degree
    shaped = made.reshape(offsets.shape[0], freqs.size, channels, channels)
    return shaped[0] if single else shaped


def _cell_rows(positions: Any, k: Any, order: int, quadrature_degree: int, xp: Any) -> Any:
    """``A W^(1/2)``: each cell's coefficients of every plane wave, ``[..., f, cell x channel, d]``.

    ``positions`` is ``[..., cell, 3]``, each cell's centre seen from the
    point the waves' amplitudes are taken at.
    """
    directions, weights, basis = _plane_waves(order, quadrature_degree)
    projection = positions @ _on(xp, directions, xp.float64).T  # [..., cell, direction]
    phase = xp.exp(1j * k[:, None, None] * projection[..., None, :, :])  # [..., f, cell, direction]
    scaled = _on(xp, (basis * np.sqrt(weights)[:, None]).T, xp.complex128)  # [channel, direction]
    rows = phase[..., :, None, :] * scaled  # [..., f, cell, channel, direction]
    return rows.reshape(*rows.shape[:-3], rows.shape[-3] * rows.shape[-2], rows.shape[-1])


def fusion_inverse(
    cells_scene: Any,
    freqs_hz: Any,
    order: int,
    *,
    regularisation: float = REGULARISATION,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    quadrature_degree: int = QUADRATURE_DEGREE,
    xp: Any = None,
) -> Any:
    """``(A W A^H + lambda tr / n I)^-1`` for a set of cells: the part that no target moves.

    ``cells_scene`` is ``[cell, 3]``, the cells' centres about any common
    origin: only the vectors between them enter. Returns
    ``[frequency, cell x channel, cell x channel]``, complex128, to be passed
    to :func:`fusion_operator` for every target read from these cells.
    """
    xp = namespace_of(cells_scene, freqs_hz, xp=xp)
    positions = xp.asarray(cells_scene, dtype=xp.float64)
    k = _wavenumbers(freqs_hz, sound_speed_m_s, xp)
    _, weights, basis = _plane_waves(order, quadrature_degree)
    size = positions.shape[0] * basis.shape[1]
    step = max(1, _CHUNK_BYTES // (16 * size * weights.size))
    eye = xp.eye(size, dtype=xp.complex128)
    out = xp.zeros((int(k.shape[0]), size, size), dtype=xp.complex128)
    for start in range(0, int(k.shape[0]), step):
        rows = _cell_rows(positions, k[start : start + step], order, quadrature_degree, xp)
        gram = rows @ xp.conj(xp.swapaxes(rows, -1, -2))
        scale = xp.real(xp.trace(gram, axis1=-2, axis2=-1)) / size
        gram = gram + regularisation * scale[:, None, None] * eye
        out[start : start + step] = xp.linalg.inv(gram)
    return out


def pair_inverse(
    between_scene: np.ndarray,
    freqs_hz: np.ndarray,
    order: int,
    *,
    regularisation: float = REGULARISATION,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    quadrature_degree: int = QUADRATURE_DEGREE,
) -> np.ndarray:
    """:func:`fusion_inverse` of two cells, the second ``between_scene`` from the first, by blocks.

    A cell's own block of the Gram matrix is the identity, the quadrature
    taking a product of two harmonics exactly, so the matrix is
    ``[[a I, C], [C^H, a I]]`` with ``a = 1 + lambda`` and ``C`` the two
    cells' rows against each other. Its inverse is then that of the 64 by
    64 Schur complement ``S = a I - C^H C / a`` and four products, in a
    fifth of the time of the 128 by 128 inverse: what a walk pays at every
    pair of cells it crosses. On the host, in double precision; returns
    ``[frequency, 2 channel, 2 channel]``, complex128, :func:`fusion_inverse`
    of ``[0, between_scene]`` to rounding.
    """
    directions, weights, basis = _plane_waves(order, quadrature_degree)
    channels = basis.shape[1]
    k = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float) / sound_speed_m_s
    projection = np.asarray(between_scene, dtype=float) @ directions.T  # [direction]
    turned = np.exp(-1j * k[:, None] * projection[None, :])  # the second cell's rows, conjugate
    weighted = (basis * weights[:, None]).T.astype(np.complex128)  # [channel, direction]
    cross = (weighted[None, :, :] * turned[:, None, :]) @ basis.astype(np.complex128)
    a = 1.0 + regularisation
    eye = np.eye(channels)
    back = np.conj(np.swapaxes(cross, -1, -2))
    schur = np.linalg.inv(a * eye - back @ cross / a)
    out = np.empty((k.size, 2 * channels, 2 * channels), dtype=np.complex128)
    upper = -(cross @ schur) / a
    out[:, :channels, :channels] = eye / a - (upper @ back) / a
    out[:, :channels, channels:] = upper
    out[:, channels:, :channels] = np.conj(np.swapaxes(upper, -1, -2))
    out[:, channels:, channels:] = schur
    return out


def fusion_operator(
    offsets_scene: Any,
    freqs_hz: Any,
    order: int,
    *,
    inverse: Any = None,
    channels_out: int | None = None,
    regularisation: float = REGULARISATION,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    quadrature_degree: int = QUADRATURE_DEGREE,
    xp: Any = None,
) -> Any:
    """The matrices that give the expansion at a target from the stacked cells'.

    ``offsets_scene`` is ``[..., cell, 3]``, ``d_j = target - cell j``.
    Returns ``[..., frequency, channel out, cell x channel]``, complex64:
    ``b_target(f) = G(f) [b_0(f); b_1(f); ...]``, the ``G`` of the pack
    format. ``channels_out`` keeps the first channels of the target only;
    one is its pressure.

    ``inverse`` is :func:`fusion_inverse` of the same cells on the same
    frequencies. Without it the Gram matrix is taken from ``offsets_scene``,
    which must then be one set of cells, ``[cell, 3]``.
    """
    xp = namespace_of(offsets_scene, freqs_hz, inverse, xp=xp)
    offsets = xp.asarray(offsets_scene, dtype=xp.float64)
    if inverse is None:
        if offsets.ndim != 2:
            raise ValueError("a batch of targets needs the cells' inverse, from fusion_inverse")
        inverse = fusion_inverse(
            -offsets,
            freqs_hz,
            order,
            regularisation=regularisation,
            sound_speed_m_s=sound_speed_m_s,
            quadrature_degree=quadrature_degree,
            xp=xp,
        )
    k = _wavenumbers(freqs_hz, sound_speed_m_s, xp)
    _, weights, basis = _plane_waves(order, quadrature_degree)
    keep = basis.shape[1] if channels_out is None else int(channels_out)
    target = _on(xp, (basis[:, :keep] * np.sqrt(weights)[:, None]).T, xp.complex128)
    size = offsets.shape[-2] * basis.shape[1]
    lead = offsets.shape[:-2]
    batch = max(1, int(np.prod(lead, dtype=np.int64)))
    out = xp.zeros((*lead, int(k.shape[0]), keep, size), dtype=xp.complex64)
    step = max(1, _CHUNK_BYTES // (16 * batch * size * weights.size))
    for start in range(0, int(k.shape[0]), step):
        # A cell sits at ``-d_j`` from the target.
        rows = _cell_rows(-offsets, k[start : start + step], order, quadrature_degree, xp)
        rhs = target @ xp.conj(xp.swapaxes(rows, -1, -2))  # Y' W A^H
        out[..., start : start + step, :, :] = rhs @ inverse[start : start + step]
    return out


def fusion_weights(
    offsets_scene: np.ndarray,
    freqs_hz: np.ndarray,
    order: int,
    *,
    regularisation: float = REGULARISATION,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
) -> np.ndarray:
    """What multiplies each cell's coefficients to give the pressure at the target.

    The first row of :func:`fusion_operator`, as ``[frequency, cell, channel]``,
    complex64. ``offsets_scene`` is ``[cell, 3]``, ``target - cell``.
    """
    offsets = np.asarray(offsets_scene, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float)
    channels = _plane_waves(order, QUADRATURE_DEGREE)[2].shape[1]
    weights = np.zeros((freqs.size, offsets.shape[0], channels), dtype=np.complex64)
    # A few frequencies at a time: the inverse of four cells is 256 by 256 a frequency.
    for start in range(0, freqs.size, _WEIGHT_CHUNK):
        operator = fusion_operator(
            offsets,
            freqs[start : start + _WEIGHT_CHUNK],
            order,
            channels_out=1,
            regularisation=regularisation,
            sound_speed_m_s=sound_speed_m_s,
            xp=np,
        )
        weights[start : start + _WEIGHT_CHUNK] = operator[:, 0, :].reshape(
            operator.shape[0], offsets.shape[0], channels
        )
    return weights


def _gathered(
    spread: Any, offsets: Any, k: Any, order: int, quadrature_degree: int, xp: Any
) -> Any:
    """``Y' W sum_j diag(exp(+i k s . d_j)) Y z_j``: ``[cell, channel, f]`` to ``[channel, f]``."""
    directions, weights, basis = _plane_waves(order, quadrature_degree)
    projection = offsets @ _on(xp, directions, xp.float64).T  # [cell, direction]
    phase = xp.exp(1j * projection[:, :, None] * k[None, None, :])  # [cell, direction, f]
    waves = xp.matmul(_on(xp, basis, xp.complex128)[None], spread) * phase
    gather = _on(xp, np.ascontiguousarray((basis * weights[:, None]).T), xp.complex128)
    return xp.matmul(gather, waves.sum(axis=0))


def apply_translation(
    field: Any,
    offset_scene: Any,
    freqs_hz: Any,
    order: int,
    *,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    quadrature_degree: int = QUADRATURE_DEGREE,
    xp: Any = None,
) -> Any:
    """``T(f) b(f)`` of :func:`translation_operator`, without forming ``T``.

    ``field`` is ``[channel, frequency]``, one cell's expansion in the
    ``rfft`` convention; ``offset_scene`` is ``d = target - centre``. Returns
    the expansion about the target, the same shape, complex128.
    """
    xp = namespace_of(field, xp=xp)
    offsets = xp.asarray(offset_scene, dtype=xp.float64).reshape(1, 3)
    k = _wavenumbers(freqs_hz, sound_speed_m_s, xp)
    spread = xp.asarray(field, dtype=xp.complex128)[None]
    return _gathered(spread, offsets, k, order, quadrature_degree, xp)


def apply_fusion(
    fields: Any,
    offsets_scene: Any,
    freqs_hz: Any,
    order: int,
    *,
    inverse: Any = None,
    regularisation: float = REGULARISATION,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    quadrature_degree: int = QUADRATURE_DEGREE,
    xp: Any = None,
) -> Any:
    """``G(f) [b_0(f); b_1(f); ...]`` of :func:`fusion_operator`, without forming ``G``.

    ``fields`` is ``[cell, channel, frequency]`` and ``offsets_scene``
    ``[cell, 3]``, ``d_j = target - cell j``. The stacked fields are solved
    against the cells' Gram matrix (``inverse``, :func:`fusion_inverse` of
    the same cells on the same frequencies; made here when not given), and
    what comes out is spread on the plane waves, each turned by its phase
    for the target, and gathered. Returns ``[channel, frequency]``,
    complex128.
    """
    xp = namespace_of(fields, inverse, xp=xp)
    offsets = xp.asarray(offsets_scene, dtype=xp.float64)
    stacked = xp.asarray(fields, dtype=xp.complex128)
    cells, channels, count = stacked.shape
    if inverse is None:
        inverse = fusion_inverse(
            -offsets,
            freqs_hz,
            order,
            regularisation=regularisation,
            sound_speed_m_s=sound_speed_m_s,
            quadrature_degree=quadrature_degree,
            xp=xp,
        )
    k = _wavenumbers(freqs_hz, sound_speed_m_s, xp)
    columns = xp.swapaxes(stacked.reshape(cells * channels, count), 0, 1)[:, :, None]
    solved = xp.swapaxes(xp.matmul(xp.asarray(inverse), columns)[:, :, 0], 0, 1)
    spread = solved.reshape(cells, channels, count)
    return _gathered(spread, offsets, k, order, quadrature_degree, xp)


def kernels(operator: Any, taps: int, *, xp: Any = None) -> Any:
    """An operator sampled on ``rfftfreq(taps, 1 / rate)`` as real kernels of ``taps`` samples.

    ``operator`` is ``[..., taps // 2 + 1, out, in]`` from
    :func:`translation_operator` or :func:`fusion_operator`. Returns
    ``[..., out, in, taps]``, float32, each kernel centred on tap
    ``taps // 2``: a translation by ``d`` moves an arrival by at most
    ``d / c``, 0.9 ms or under four samples at 4 kHz for 0.30 m. Measured
    under 1414 Hz at 4 kHz, 32 taps are the operator to -49 dB at 0.30 m and
    -56 dB at 0.10 m, in 0.52 MB an offset; 16 taps to -36 dB. The centre is
    the kernel's time zero; the caller takes ``taps // 2`` samples of delay
    out.
    """
    xp = namespace_of(operator, xp=xp)
    if taps % 2 or operator.shape[-3] != taps // 2 + 1:
        raise ValueError(f"{operator.shape[-3]} frequencies are not the transform of {taps} taps")
    spectrum = xp.moveaxis(xp.asarray(operator, dtype=xp.complex128), -3, -1)
    shift = xp.exp(-1j * np.pi * xp.arange(taps // 2 + 1))  # half the length, in the spectrum
    out: Any = xp.fft.irfft(spectrum * shift, n=taps, axis=-1).astype(xp.float32)
    return out


def _to_triangles(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Distance from one point to each of ``triangles``, by the regions of the closest point.

    Ericson, *Real-Time Collision Detection*, 5.1.5: the face first, then
    each edge and each vertex region that overrides it, in the book's order.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = point - a, point - b, point - c
    d1, d2 = np.einsum("tj,tj->t", ab, ap), np.einsum("tj,tj->t", ac, ap)
    d3, d4 = np.einsum("tj,tj->t", ab, bp), np.einsum("tj,tj->t", ac, bp)
    d5, d6 = np.einsum("tj,tj->t", ab, cp), np.einsum("tj,tj->t", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        total = va + vb + vc
        s, t = vb / total, vc / total
        edge_ab, edge_ac = d1 / (d1 - d3), d2 / (d2 - d6)
        edge_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    on_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    s, t = np.where(on_bc, 1.0 - edge_bc, s), np.where(on_bc, edge_bc, t)
    on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    s, t = np.where(on_ac, 0.0, s), np.where(on_ac, edge_ac, t)
    on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    s, t = np.where(on_ab, edge_ab, s), np.where(on_ab, 0.0, t)
    at_c = (d6 >= 0) & (d5 <= d6)
    s, t = np.where(at_c, 0.0, s), np.where(at_c, 1.0, t)
    at_b = (d3 >= 0) & (d4 <= d3)
    s, t = np.where(at_b, 1.0, s), np.where(at_b, 0.0, t)
    at_a = (d1 <= 0) & (d2 <= 0)
    s, t = np.where(at_a, 0.0, s), np.where(at_a, 0.0, t)
    # A triangle without area has no face; its vertices and edges were tried above.
    s, t = np.nan_to_num(s), np.nan_to_num(t)
    closest = a + s[:, None] * ab + t[:, None] * ac
    distance: np.ndarray = np.linalg.norm(point - closest, axis=1)
    return distance


def clearance_m(points: np.ndarray, triangles: np.ndarray, *, large_m: float = 0.25) -> np.ndarray:
    """Distance from each point to the nearest of ``triangles``, in metres.

    ``points`` is ``[point, 3]`` and ``triangles`` ``[triangle, 3, 3]``, in one
    frame: the occluders of ``mirror/scene.npz`` for a cell's ``clearance_m``.
    The nearest vertex bounds the answer from above, so only the triangles
    whose centroid lies within that bound plus their own radius are measured;
    those wider than ``large_m`` are always measured.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not triangles.shape[0]:
        raise ValueError(f"triangles must be [triangle, 3, 3] and not empty, got {triangles.shape}")
    centroid = triangles.mean(axis=1)
    radius = np.linalg.norm(triangles - centroid[:, None, :], axis=2).max(axis=1)
    large = np.flatnonzero(radius > large_m)
    small = np.flatnonzero(radius <= large_m)
    bound = cKDTree(triangles.reshape(-1, 3)).query(points)[0]
    tree = cKDTree(centroid[small]) if small.size else None
    out = np.empty(points.shape[0])
    for i, point in enumerate(points):
        near = (
            small[tree.query_ball_point(point, bound[i] + large_m)] if tree is not None else small
        )
        chosen = np.concatenate([large, near])
        out[i] = min(float(bound[i]), float(_to_triangles(point, triangles[chosen]).min()))
    return out


def serving_radius_m(
    cells: np.ndarray,
    cell_clearance_m: np.ndarray,
    source: np.ndarray | None = None,
    *,
    surface_share: float = SURFACE_SHARE,
    source_share: float = SOURCE_SHARE,
) -> np.ndarray:
    """How far each cell's expansion may be read from, in metres.

    An interior expansion holds in the ball about its centre that is free of
    surfaces and of sources, and its order 7 cut holds in a part of that
    ball only: ``surface_share`` of the clearance to the nearest surface
    (:func:`clearance_m`) and ``source_share`` of the distance to the
    source, whichever is smaller. ``source`` is one position or
    ``[source, 3]``; the nearest counts.
    """
    radius = surface_share * np.asarray(cell_clearance_m, dtype=float)
    if source is not None:
        sources = np.atleast_2d(np.asarray(source, dtype=float))
        cells = np.atleast_2d(np.asarray(cells, dtype=float))
        to_source = np.linalg.norm(cells[:, None, :] - sources[None, :, :], axis=-1).min(axis=1)
        radius = np.minimum(radius, source_share * to_source)
    return np.asarray(radius, dtype=float)


def cell_stride(
    to_source_m: float,
    pitch_m: float,
    *,
    source_share: float = SOURCE_SHARE,
    fuse_within_m: float = FUSE_WITHIN_M,
) -> int:
    """How many cells of a line of cells ``pitch_m`` apart a source may skip.

    Two cells ``n pitch_m`` apart serve every head between them when half
    that is within what a cell may be read from: ``source_share`` of the
    source's distance, and ``fuse_within_m``. A source 2 m away reads every
    fourth cell of a line at 0.15 m and one 0.5 m away reads them all, which
    is what keeps the pairs of a scene few. At least one.
    """
    reach = min(fuse_within_m, source_share * float(to_source_m))
    return max(1, int(np.floor(2.0 * reach / pitch_m + 1e-9)))


def choose_cells(
    head: np.ndarray,
    cells: np.ndarray,
    serving_radius: np.ndarray,
    *,
    exact_under_m: float = EXACT_UNDER_M,
    translate_within_m: float = TRANSLATE_WITHIN_M,
    fuse_within_m: float = FUSE_WITHIN_M,
) -> tuple[int, tuple[int, int]]:
    """The listener's mode at ``head`` and the two cells it is read from.

    A cell may serve a head at distance ``d`` only when ``d`` is under the
    cell's serving radius (:func:`serving_radius_m`) and under
    ``fuse_within_m``. Slot 0 is the nearest cell that may serve. Slot 1 is
    the nearest that may serve on the other side of the head, which is what
    makes two cells better than one. The mode is exact on a cell, fused with
    two, and translated with one when that one is within
    ``translate_within_m``.

    Returns ``(mode, (cell 0, cell 1))`` with ``-1`` for a slot not used.
    Raises ``LookupError`` when no cell may serve: a cell is needed there
    (``/cells/kind`` 2), and it is the caller that names the step.
    """
    head = np.asarray(head, dtype=float)
    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    radius = np.asarray(serving_radius, dtype=float)
    away = cells - head[None, :]
    distance = np.linalg.norm(away, axis=1)
    nearest = int(np.argmin(distance))
    if distance[nearest] < exact_under_m:
        return MODE_EXACT, (nearest, -1)
    serving = np.flatnonzero((distance <= radius) & (distance <= fuse_within_m))
    if serving.size:
        serving = serving[np.argsort(distance[serving], kind="stable")]
        first = int(serving[0])
        opposite = [int(j) for j in serving[1:] if float(away[j] @ away[first]) < 0.0]
        if opposite:
            return MODE_FUSED, (first, opposite[0])
        if distance[first] <= translate_within_m:
            return MODE_TRANSLATED, (first, -1)
    raise LookupError(
        f"no cell may serve the head at {np.round(head, 3).tolist()}: the nearest is"
        f" {distance[nearest]:.3f} m away and may be read {float(radius[nearest]):.3f} m away"
    )
