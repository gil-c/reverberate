"""From the field at one or two cells to the field at the head: the one interface.

The mathematics of moving an order 7 expansion is
:mod:`reverberate.spatial.translate`: its plane wave model, its quadrature,
its regularisation and its one sign convention, ``d = target - centre``.
The engine asks one thing of it, :class:`Translation`: the fields at the
cells and where the head is seen from each, in; the field at the head, out.

:class:`SpatialTranslation` is the engine's default and is that library
behind the interface. One cell is the plain translation
(:func:`~reverberate.spatial.translate.apply_translation`), so a head on its
cell is the cell; two cells are the minimum norm fusion
(:func:`~reverberate.spatial.translate.apply_fusion`), whose Gram matrix
depends only on the vector between the two cells and is inverted once per
vector (:func:`~reverberate.spatial.translate.fusion_inverse`), as
``scene-pack.md`` says. Neither forms the operator.

:class:`OperatorTranslation` takes a function that returns the operator
``G(f)`` itself, which is how another estimator is tried and how the
applied forms are tested against the library's operators.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from reverberate.spatial.translate import (
    QUADRATURE_DEGREE,
    REGULARISATION,
    apply_fusion,
    apply_translation,
    fusion_inverse,
)

__all__ = ["OperatorTranslation", "SpatialTranslation", "Translation"]


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
    what :func:`~reverberate.spatial.translate.translation_operator` and
    :func:`~reverberate.spatial.translate.fusion_operator` return.
    """

    def __init__(self, operator: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self.operator = operator

    def to_head(self, fields: Any, offsets_m: np.ndarray, freqs_hz: np.ndarray, xp: Any) -> Any:
        g = xp.asarray(self.operator(np.asarray(offsets_m, dtype=float), freqs_hz))
        stacked = fields.reshape(-1, fields.shape[-1])
        return xp.einsum("fab,bf->af", g, stacked)


class SpatialTranslation:
    """``reverberate.spatial.translate`` behind the interface: one cell moved, two fused."""

    def __init__(
        self,
        order: int,
        sound_speed_m_s: float,
        *,
        quadrature_degree: int = QUADRATURE_DEGREE,
        regularisation: float = REGULARISATION,
        held: int = 8,
    ) -> None:
        self.order = order
        self.sound_speed_m_s = sound_speed_m_s
        self.quadrature_degree = quadrature_degree
        self.regularisation = regularisation
        self._held = held
        self._inverses: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    @classmethod
    def from_fusion(
        cls, fusion: Mapping[str, float], order: int, sound_speed_m_s: float
    ) -> SpatialTranslation:
        """With the pack's ``fusion_json``."""
        return cls(
            order,
            sound_speed_m_s,
            quadrature_degree=int(fusion.get("quadrature_degree", QUADRATURE_DEGREE)),
            regularisation=float(fusion.get("lambda", REGULARISATION)),
        )

    def _inverse(self, between: np.ndarray, freqs_hz: np.ndarray, xp: Any) -> Any:
        """The Gram matrix's inverse for two cells ``between`` apart, kept per vector.

        Inverted on the host in double precision whatever the device, then
        moved: a card's own inverse would be another rounding of the render.
        """
        key = (
            tuple(np.round(between, 6).tolist()),
            int(freqs_hz.size),
            float(freqs_hz[0]),
            float(freqs_hz[-1]),
            id(xp),
        )
        if key in self._inverses:
            self._inverses.move_to_end(key)
            return self._inverses[key]
        made = xp.asarray(
            fusion_inverse(
                np.stack([np.zeros(3), between]),
                freqs_hz,
                self.order,
                regularisation=self.regularisation,
                sound_speed_m_s=self.sound_speed_m_s,
                quadrature_degree=self.quadrature_degree,
                xp=np,
            )
        )
        self._inverses[key] = made
        while len(self._inverses) > self._held:
            self._inverses.popitem(last=False)
        return made

    def to_head(self, fields: Any, offsets_m: np.ndarray, freqs_hz: np.ndarray, xp: Any) -> Any:
        cells = int(fields.shape[0])
        offsets = np.asarray(offsets_m, dtype=float)
        freqs = np.asarray(freqs_hz, dtype=float)
        common = {
            "sound_speed_m_s": self.sound_speed_m_s,
            "quadrature_degree": self.quadrature_degree,
            "xp": xp,
        }
        if cells == 1:
            return apply_translation(fields[0], xp.asarray(offsets[0]), freqs, self.order, **common)
        if cells != 2:
            raise ValueError("the head is read from one cell or from two")
        # Cell 1 seen from cell 0: ``d_0 - d_1``, whatever the head does.
        inverse = self._inverse(offsets[0] - offsets[1], freqs, xp)
        return apply_fusion(
            fields, xp.asarray(offsets), freqs, self.order, inverse=inverse, **common
        )
