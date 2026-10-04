"""From the field at one or two cells to the field at the head: the one interface.

The mathematics of moving an order 7 expansion belongs to
``reverberate.spatial.translate`` (lot L4). The engine asks one thing of it,
:class:`Translation`: the fields at the cells and where the head is seen
from each, in; the field at the head, out. Whatever L4 settles on is
swapped in through :class:`OperatorTranslation`, which takes a function
returning the operator ``G(f)``, or by any object with the same method.

:class:`PlaneWaveFusion` is the estimator ``scene-pack.md`` writes down,
for all 64 channels, and what the engine uses until then. It never forms
``G``: with ``Y`` the harmonics on the quadrature and ``W`` its weights,

    G [b_0; b_1] = Y^T W sum_j diag(e^{+i k s . r_j}) Y z_j,
    [z_0; z_1] = (A W A^H + mu I)^-1 [b_0; b_1],

so the cells' fields are solved against a matrix that depends only on the
vector between the two cells (factored once per vector, as the format
says), spread on the plane waves, each plane wave turned by its own phase
for the head's offset, and gathered. One cell needs no solve:
``A W A^H`` is the identity.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from reverberate.spatial.sh import quadrature, real_sh, scene_to_ambisonic

__all__ = ["OperatorTranslation", "PlaneWaveFusion", "Translation"]


class Translation(Protocol):
    """What the engine asks of the translation's mathematics."""

    def to_head(self, fields: Any, offsets_m: np.ndarray, freqs_hz: np.ndarray, xp: Any) -> Any:
        """The field at the head from the fields at one or two cells.

        ``fields`` is ``[cell, channel, frequency]``, complex, on ``xp``:
        each cell's ambisonic signal in the ``rfft`` convention.
        ``offsets_m`` is ``[cell, 3]``, the head seen from each cell in the
        scene frame, ``l - position[cell]``. Returns ``[channel, frequency]``.
        """
        ...


class OperatorTranslation:
    """A :class:`Translation` from a function that returns the operator itself.

    ``operator(offsets_m, freqs_hz)`` gives ``G`` as
    ``[frequency, channel, cell * channel]``: the form the format writes and
    the natural result of a function of ``spatial.translate``.
    """

    def __init__(self, operator: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self.operator = operator

    def to_head(self, fields: Any, offsets_m: np.ndarray, freqs_hz: np.ndarray, xp: Any) -> Any:
        g = xp.asarray(self.operator(np.asarray(offsets_m, dtype=float), freqs_hz))
        stacked = fields.reshape(-1, fields.shape[-1])
        return xp.einsum("fab,bf->af", g, stacked)


class PlaneWaveFusion:
    """The minimum norm plane wave estimator of ``scene-pack.md``, on every channel."""

    def __init__(
        self,
        order: int,
        sound_speed_m_s: float,
        *,
        quadrature_degree: int = 26,
        regularisation: float = 1e-3,
        held: int = 8,
    ) -> None:
        self.order = order
        self.sound_speed_m_s = sound_speed_m_s
        self.regularisation = regularisation
        directions, weights = quadrature(quadrature_degree)
        self.directions = directions
        self.weights = weights / weights.sum()
        self.basis = real_sh(order, directions)  # [direction, channel]
        self.gather = np.ascontiguousarray((self.basis * self.weights[:, None]).T)
        self._held = held
        self._solved: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._moved: dict[int, tuple[Any, Any]] = {}

    @classmethod
    def from_fusion(
        cls, fusion: Mapping[str, float], order: int, sound_speed_m_s: float
    ) -> PlaneWaveFusion:
        """With the pack's ``fusion_json``."""
        return cls(
            order,
            sound_speed_m_s,
            quadrature_degree=int(fusion.get("quadrature_degree", 26)),
            regularisation=float(fusion.get("lambda", 1e-3)),
        )

    def _on(self, xp: Any) -> tuple[Any, Any]:
        if id(xp) not in self._moved:
            self._moved[id(xp)] = (xp.asarray(self.basis), xp.asarray(self.gather))
        return self._moved[id(xp)]

    def _inverse(self, between: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """``(A W A^H + mu I)^-1`` per frequency, for two cells ``between`` apart."""
        key = (tuple(np.round(between, 6)), freqs_hz.size, float(freqs_hz[0]), float(freqs_hz[-1]))
        if key in self._solved:
            self._solved.move_to_end(key)
            return self._solved[key]
        k = 2.0 * np.pi * freqs_hz / self.sound_speed_m_s
        phase = np.exp(-1j * k[:, None] * (self.directions @ between)[None, :])  # [f, direction]
        cross = np.matmul(self.gather[None, :, :] * phase[:, None, :], self.basis)
        own = self.gather @ self.basis
        channels = own.shape[0]
        gram = np.empty((freqs_hz.size, 2 * channels, 2 * channels), dtype=complex)
        gram[:, :channels, :channels] = own
        gram[:, channels:, channels:] = own
        gram[:, :channels, channels:] = cross
        gram[:, channels:, :channels] = cross.conj().transpose(0, 2, 1)
        scale = np.trace(gram, axis1=1, axis2=2).real / (2 * channels)
        gram += (self.regularisation * scale)[:, None, None] * np.eye(2 * channels)[None]
        inverse = np.linalg.inv(gram)
        self._solved[key] = inverse
        while len(self._solved) > self._held:
            self._solved.popitem(last=False)
        return inverse

    def to_head(self, fields: Any, offsets_m: np.ndarray, freqs_hz: np.ndarray, xp: Any) -> Any:
        cells, channels, _ = fields.shape
        if cells not in (1, 2):
            raise ValueError("the head is read from one cell or from two")
        basis, gather = self._on(xp)
        offsets = scene_to_ambisonic(np.asarray(offsets_m, dtype=float))  # [cell, 3]
        k = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float) / self.sound_speed_m_s
        phase = np.exp(1j * (self.directions @ offsets.T).T[:, :, None] * k[None, None, :])
        if cells == 1:
            solved = fields / (1.0 + self.regularisation)
        else:
            inverse = xp.asarray(self._inverse(offsets[0] - offsets[1], np.asarray(freqs_hz)))
            stacked = fields.reshape(2 * channels, -1)
            solved = xp.einsum("fab,bf->af", inverse, stacked).reshape(2, channels, -1)
        waves = xp.matmul(basis[None], solved) * xp.asarray(phase)  # [cell, direction, f]
        return xp.matmul(gather, waves.sum(axis=0))
