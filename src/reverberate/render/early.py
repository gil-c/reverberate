"""The arrivals above the crossover: every path a moving delay line into the listener's basis.

Between two steps a path's apparent source and the listener both move in a
straight line (``scene-pack.md``), so its delay is continuous and is heard
as the Doppler shift it is. What the path does to the signal besides is a
set of gains that are linear between the steps: the seven band gains, the
voice's directivity, the air's loss, the crossover's mask and the levelling
scalar. The engine therefore filters the dry signal **once**, into every
combination of bank band, mask and air distance (the *variants*), and a
path is two things only: a mix of the variants with its own coefficients,
and a read of that mix at its moving delay. The read being linear, the mix
at the step's coefficients and the mix at the next step's are read at the
same delays and cross-faded, which is the format's linear gain exactly.

The band filters are the octave bank's own (``metrics.octave_bank``), laid
as ``mirror.render.early_signals`` lays them, so a path at rest whose delay
is a whole number of samples is that module's pulse convolved with the dry
signal, to the rounding of a transform.

The direction is the apparent source seen from the listener, on the sphere
by construction; its harmonics are evaluated with ``spatial.sh.real_sh`` at
:attr:`~reverberate.render.engine.RenderSettings.direction_nodes` instants
a step and are linear between two of them. A path turning by ``a`` radians
in a step errs by ``(7 a / nodes)^2 / 8`` of its top order at most: 0.2 per
cent for a source half a metre away crossed at 1.5 m/s.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, TypeVar

import numpy as np
from scipy.fft import next_fast_len

from reverberate.compute import usable_cores
from reverberate.metrics import octave_bank
from reverberate.mirror.directivity import directivity_gain
from reverberate.render import delay
from reverberate.render.dry import DryTrack
from reverberate.render.pack import JUMP_M, ScenePack, Source, band_map
from reverberate.spatial.sh import real_sh, scene_to_ambisonic

__all__ = ["AIR_NODE_M", "EarlyPart", "air_weights", "onset_weight", "run_all"]

_T = TypeVar("_T")
_R = TypeVar("_R")

#: The air's loss is filtered at distances this far apart and linear between two:
#: 3 per cent of the 20 kHz component at worst, 0.1 per cent at 8 kHz.
AIR_NODE_M = 8.0
#: No more air distances than this; a longer path spaces them further apart.
AIR_NODES_MOST = 12
#: Samples a variant is filtered beyond what is read, each side: the bank's half
#: length, and the air's and the taper's.
MARGIN = 256 + 256
#: The early part's band ends in a raised cosine from this share of the Nyquist
#: frequency to the whole of it: 22 to 24 kHz.
TAPER_FROM = 11.0 / 12.0


def fft_module(xp: Any) -> Any:
    """The transforms of ``xp``: scipy's, which take ``workers``, on the host."""
    if xp is np:
        import scipy.fft

        return scipy.fft
    return xp.fft


def workers_of(xp: Any, workers: int) -> dict[str, int]:
    """The transforms' threads on the host: ``workers``, or every core this process may use.

    Not scipy's own "every core", which counts the host's and not the
    container's share of them (:func:`reverberate.compute.usable_cores`).
    """
    if xp is not np:
        return {}
    return {"workers": usable_cores() if workers < 0 else workers}


def run_all(work: Callable[[_T], _R], items: Iterable[_T], xp: Any, workers: int) -> list[_R]:
    """``work`` on every item, the results in the items' order.

    On the host each item has a thread, up to ``workers`` (every core when
    negative): numpy releases the lock in the products, the transforms and
    the copies that are the work. On a card the items run one after the
    other. The results do not depend on which.
    """
    items = list(items)
    count = usable_cores() if workers < 0 else workers
    if xp is not np or count < 2 or len(items) < 2:
        return [work(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(count, len(items))) as pool:
        return list(pool.map(work, items))


def onset_weight(tau_s: np.ndarray, onset_s: np.ndarray, coherent_s: float, fade_s: float) -> Any:
    """``Crossover.onset_window`` at the time ``tau_s``: one, then a raised cosine to zero."""
    if coherent_s <= 0.0:
        return np.zeros_like(tau_s)
    x = np.clip((tau_s - onset_s - coherent_s) / max(fade_s, 1e-12), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def air_weights(distance_m: np.ndarray, nodes_m: np.ndarray) -> np.ndarray:
    """``[row, node]``: each distance's weights on the two nodes either side of it."""
    weights = np.zeros((distance_m.size, nodes_m.size))
    if nodes_m.size == 1:
        weights[:, 0] = 1.0
        return weights
    spacing = nodes_m[1] - nodes_m[0]
    at = np.clip(distance_m / spacing, 0.0, nodes_m.size - 1.0)
    lower = np.minimum(at.astype(int), nodes_m.size - 2)
    rows = np.arange(distance_m.size)
    weights[rows, lower] = 1.0 - (at - lower)
    weights[rows, lower + 1] = at - lower
    return weights


class EarlyPart:
    """One source's arrivals, rendered a run of steps at a time."""

    def __init__(
        self,
        pack: ScenePack,
        source: Source,
        tracks: list[DryTrack],
        xp: Any,
        *,
        directivity: bool,
        direction_nodes: int,
        workers: int,
        band_gain_db: np.ndarray | None = None,
    ) -> None:
        h = pack.header
        self.pack, self.source, self.tracks, self.xp = pack, source, tracks, xp
        self.rate = h.sample_rate_hz
        self.step = h.step_samples
        self.nodes = direction_nodes
        if self.step % direction_nodes:
            raise ValueError("the direction nodes do not divide a step")
        self.workers = workers
        early = source.early
        self.picks = band_map(h.bands_hz, h.bank)
        self.kernels = np.ascontiguousarray(
            np.asarray(octave_bank(int(round(self.rate))).filters, dtype=float).T
        )
        self.centre = (self.kernels.shape[1] - 1) // 2
        rows = int(early.path_id.shape[0])
        step_of = np.repeat(np.arange(h.steps), np.diff(early.offsets))
        delay_s = np.asarray(early.delay_s, dtype=float)
        # Air: the distances the dry signal is filtered at, from the longest path.
        longest = float(delay_s.max()) * h.sound_speed_m_s if rows else 0.0
        if pack.air.enabled and rows:
            spacing = max(AIR_NODE_M, longest / (AIR_NODES_MOST - 1))
            self.air_m = spacing * np.arange(int(np.ceil(longest / spacing)) + 1)
        else:
            self.air_m = np.zeros(1)
        self.masks = len(tracks)
        bands = len(h.bank)
        # A row's coefficients on the variants, [row, mask, band, node], at its own step.
        gain = np.asarray(early.gain, dtype=float)[:, self.picks]
        if directivity and rows:
            # The pattern of the source's model at each row's own departure and facing,
            # read in the pack's single precision: a pack in memory renders as its file.
            pattern = pack.directivity[source.directivity_model]
            pattern = replace(pattern, gain_db=np.asarray(pattern.gain_db, dtype=np.float32))
            gain = (
                gain
                * directivity_gain(
                    pattern,
                    np.asarray(early.departure, dtype=float),
                    np.asarray(source.yaw_deg, dtype=float)[step_of],
                )[:, self.picks]
            )
        if band_gain_db is None:
            gain = (
                gain
                * (10.0 ** (np.asarray(source.level.high_gain_db, float)[step_of] / 20.0))[:, None]
            )
        else:
            # THE SEAM (reverberate.render.seam): a level a step and a band of the bank,
            # ``[step, bank]`` in dB, in place of the pack's scalar.
            gain = gain * 10.0 ** (np.asarray(band_gain_db, dtype=float)[step_of] / 20.0)
        if self.masks == 2:
            together = onset_weight(
                delay_s + pack.mirror.lead_s,
                np.asarray(source.level.onset_s, dtype=float)[step_of],
                pack.crossover.coherent_s,
                pack.crossover.coherent_fade_s,
            )
            mask = np.stack([together, 1.0 - together], axis=1)
        else:
            mask = np.ones((rows, 1))
        air = air_weights(delay_s * h.sound_speed_m_s, self.air_m)
        self.variants = self.masks * bands * self.air_m.size
        self.coefficients = (
            mask[:, :, None, None] * gain[:, None, :, None] * air[:, None, None, :]
        ).reshape(rows, self.variants)
        listener = np.asarray(pack.listener.position, dtype=float)
        # The apparent source of every row: the image itself for a specular path. The
        # pack's float32 direction is unit to 1e-7 only; made unit here, so the delay
        # at a step is the stored delay and not the stored delay times that.
        arrival = np.asarray(early.arrival, dtype=float)
        arrival = arrival / np.maximum(np.linalg.norm(arrival, axis=1, keepdims=True), 1e-300)
        self.apparent = listener[step_of] + (h.sound_speed_m_s * delay_s)[:, None] * arrival
        self.listener = listener
        self.lead = pack.mirror.lead_s
        self.sound_speed = h.sound_speed_m_s
        atmosphere = pack.air.atmosphere
        self._attenuation = atmosphere.attenuation_np_per_m

    # ----------------------------------------------------------------------

    def _variants(self, first: int, last: int) -> Any:
        """Every variant over the samples ``[first, last)``, at twice the rate: ``[v, 2 n]``."""
        xp = self.xp
        fft = fft_module(xp)
        kw = workers_of(xp, self.workers)
        length = last - first + 2 * MARGIN
        n = next_fast_len(length, real=True)
        n += n % 2
        freqs = np.fft.rfftfreq(n, 1.0 / self.rate)
        # The bank laid as fftconvolve's "same" lays it: its centre tap on the sample.
        advance = np.exp(2j * np.pi * freqs * self.centre / self.rate)
        bank = np.fft.rfft(self.kernels, n, axis=1) * advance[None, :]  # [band, f]
        loss = np.exp(-self._attenuation(freqs)[None, :] * self.air_m[:, None])  # [node, f]
        # A signal held to its very Nyquist has no samples between its samples that
        # do not ring across the whole run: the top of the band is faded out.
        fade = np.clip(
            (freqs - TAPER_FROM * 0.5 * self.rate) / ((1 - TAPER_FROM) * 0.5 * self.rate), 0, 1
        )
        loss = loss * np.cos(0.5 * np.pi * fade)[None, :] ** 2
        shape = xp.asarray((bank[:, None, :] * loss[None, :, :]).reshape(-1, freqs.size))
        padded = xp.zeros((self.masks, shape.shape[0], n + 1), dtype=xp.complex128)
        for m, track in enumerate(self.tracks):
            dry = xp.asarray(track.read(first - MARGIN, first - MARGIN + n))
            spectrum = fft.rfft(dry)
            padded[m, :, : freqs.size] = spectrum[None, :] * shape
        # Twice the rate: the same spectrum in a transform twice as long, its
        # own Nyquist bin shared between the two it becomes.
        padded[:, :, freqs.size - 1] *= 0.5
        doubled = fft.irfft(padded.reshape(self.variants, n + 1), 2 * n, axis=-1, **kw)
        doubled *= 2.0
        return doubled[:, 2 * MARGIN : 2 * (MARGIN + last - first)]

    def render(self, k0: int, k1: int) -> Any:
        """The intervals ``k0`` to ``k1 - 1``: ``[channel, (k1 - k0) step]`` on ``xp``."""
        xp = self.xp
        step, rate = self.step, self.rate
        channels = self.pack.header.channels
        out = xp.zeros((channels, (k1 - k0) * step))
        offsets = np.asarray(self.source.early.offsets)
        lo_row, hi_row = int(offsets[k0]), int(offsets[k1 + 1])
        if hi_row == lo_row:
            return out
        over = delay.OVERSAMPLE
        guard = delay.TAPS
        # A delay between two steps never exceeds the longer of its two ends by
        # more than the listener's and the source's travel; a step of each covers it.
        reach = float(self.source.early.delay_s[lo_row:hi_row].max()) + self.lead
        first = k0 * step - int(np.ceil(reach * rate)) - step - guard
        last = k1 * step + guard
        if all(track.silent(first - MARGIN, last + MARGIN) for track in self.tracks):
            return out
        variants = self._variants(first, last)
        ids = np.asarray(self.source.early.path_id)
        u = np.arange(step) / step
        u_x = xp.asarray(u)
        node_u = np.arange(self.nodes + 1) / self.nodes
        within = step // self.nodes
        v = xp.asarray(np.arange(within) / within)

        def one(k: int) -> Any:
            """The interval from step ``k`` to the next: ``[channel, step]``, or ``None``."""
            a = slice(int(offsets[k]), int(offsets[k + 1]))
            b = slice(int(offsets[k + 1]), int(offsets[k + 2]))
            if a.stop == a.start and b.stop == b.start:
                return None
            q0, q1, c0, c1 = self._interval(ids, a, b)
            l0, l1 = self.listener[k], self.listener[k + 1]
            # [path, sample, 3]: the path seen from the head, sample by sample.
            seen = (
                q0[:, None, :] * (1.0 - u)[None, :, None]
                + q1[:, None, :] * u[None, :, None]
                - (l0[None, :] * (1.0 - u)[:, None] + l1[None, :] * u[:, None])[None]
            )
            tau = np.linalg.norm(seen, axis=2) / self.sound_speed + self.lead
            position = over * ((k * step + np.arange(step))[None, :] - tau * rate - first)
            low = max(int(np.floor(position.min())) - guard, 0)
            high = min(int(np.ceil(position.max())) + guard, variants.shape[1])
            window = variants[:, low:high]
            mixes = xp.stack(
                [xp.asarray(c0) @ window, xp.asarray(c1) @ window], axis=-1
            )  # [path, sample, 2]
            got = delay.read(mixes, position - low, xp)
            signal = got[:, :, 0] * (1.0 - u_x)[None, :] + got[:, :, 1] * u_x[None, :]
            # The harmonics at the nodes of the interval, linear between two.
            at = (
                q0[:, None, :] * (1.0 - node_u)[None, :, None]
                + q1[:, None, :] * node_u[None, :, None]
                - (l0[None, :] * (1.0 - node_u)[:, None] + l1[None, :] * node_u[:, None])[None]
            )
            paths = at.shape[0]
            harmonics = xp.asarray(
                real_sh(self.pack.header.order, scene_to_ambisonic(at.reshape(-1, 3))).reshape(
                    paths, self.nodes + 1, channels
                )
            )
            pieces = signal.reshape(paths, self.nodes, within)
            # Per node, the paths' harmonics at its two ends by the paths' signals
            # under the two ramps: [node, channel, 2 path] by [node, 2 path, sample].
            ends = xp.concatenate([harmonics[:, :-1], harmonics[:, 1:]], axis=0)
            ramps = xp.concatenate(
                [pieces * (1.0 - v)[None, None, :], pieces * v[None, None, :]], axis=0
            )
            encoded = xp.matmul(
                xp.ascontiguousarray(ends.transpose(1, 2, 0)), ramps.transpose(1, 0, 2)
            ).transpose(1, 0, 2)
            return encoded.reshape(channels, step)

        # The intervals are independent: a thread each on the host.
        for index, made in enumerate(run_all(one, range(k0, k1), xp, self.workers)):
            if made is not None:
                out[:, index * step : (index + 1) * step] = made
        return out

    def _interval(
        self, ids: np.ndarray, a: slice, b: slice
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The paths of one interval: apparent source and coefficients at its two ends.

        A path at both steps moves from one to the other. A path at one step
        only, or one whose apparent source jumped, keeps the apparent source
        of the step where it is and has no gain at the other.
        """
        ids_a, ids_b = ids[a], ids[b]
        _, in_a, in_b = np.intersect1d(ids_a, ids_b, assume_unique=True, return_indices=True)
        row_a, row_b = a.start + in_a, b.start + in_b
        moved = np.linalg.norm(self.apparent[row_b] - self.apparent[row_a], axis=1)
        same = moved <= JUMP_M
        kept_a, kept_b = row_a[same], row_b[same]
        only_a = np.setdiff1d(np.arange(a.start, a.stop), kept_a, assume_unique=True)
        only_b = np.setdiff1d(np.arange(b.start, b.stop), kept_b, assume_unique=True)
        nothing = np.zeros((1, self.variants))
        q0 = np.concatenate([self.apparent[kept_a], self.apparent[only_a], self.apparent[only_b]])
        q1 = np.concatenate([self.apparent[kept_b], self.apparent[only_a], self.apparent[only_b]])
        c0 = np.concatenate(
            [
                self.coefficients[kept_a],
                self.coefficients[only_a],
                np.repeat(nothing, only_b.size, 0),
            ]
        )
        c1 = np.concatenate(
            [
                self.coefficients[kept_b],
                np.repeat(nothing, only_a.size, 0),
                self.coefficients[only_b],
            ]
        )
        return q0, q1, c0, c1
