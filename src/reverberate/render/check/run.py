"""The checks themselves: a pack rendered through the engine, measured, judged.

Three families, in the order a listener would meet their faults:

- **impulse probes** (:func:`probe_source`): the dry signal is one sample,
  so what is rendered is the scene's response at that instant, and its
  arrival, level, direction, spectrum and decay are read off it;
- **continuity** (:func:`continuity_of`): the dry signal is two steady tones,
  then pink noise, over the stretch of the scene where the most changes, and
  what is not the tone is a fault;
- **the mix** (:func:`mix_of`): the recipe's clips, every source's level at
  the listener in dB SPL, and the two ears the page would play.

Every result carries its number, its limits and why the limits are those
(:data:`LIMITS`). A limit is what a listener would reject, not what the
present engine reaches.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.metrics import band_centres
from reverberate.mirror.directivity import directivity_gain
from reverberate.render.check import measure
from reverberate.render.check.binaural import BLOCK, PageDecoder, head_matrix
from reverberate.render.check.clips import ClipSource, Feed, feed_of
from reverberate.render.dry import DryTrack
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.pack import KIND_DIRECT, ScenePack, Source
from reverberate.spatial.binaural import BinauralDecoder, ild_db, itd_s
from reverberate.spatial.sh import real_sh, scene_to_ambisonic

__all__ = [
    "LIMITS",
    "CheckSettings",
    "Limit",
    "Result",
    "continuity_of",
    "duplicate_arrivals",
    "mix_of",
    "probe_source",
    "worst",
]

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"
_RANK = {SKIP: 0, INFO: 0, PASS: 1, WARN: 2, FAIL: 3}

#: The two tones of the continuity probe: one each side of the crossover's ramp.
TONES_HZ = (400.0, 2500.0)


@dataclass(frozen=True)
class Limit:
    """What a number is held to: it passes to ``ok``, warns to ``warn``, fails beyond."""

    ok: float
    warn: float
    unit: str
    reason: str
    #: ``"abs"``: the magnitude must stay under; ``"max"``: the value must stay under.
    kind: str = "abs"

    def judge(self, value: float) -> str:
        if not np.isfinite(value):
            return PASS if value < 0 and self.kind == "max" else FAIL
        size = abs(value) if self.kind == "abs" else value
        return PASS if size <= self.ok else WARN if size <= self.warn else FAIL

    def text(self) -> str:
        bar = "|x|" if self.kind == "abs" else "x"
        return f"{bar} <= {self.ok:g} {self.unit}; warn to {self.warn:g}"


LIMITS: dict[str, Limit] = {
    "arrival_time": Limit(
        0.1,
        0.5,
        "ms",
        "0.1 ms is 3.4 cm: a cell's node lies 2 cm from the point asked for and the solver's "
        "pulse trails by two samples. Half a millisecond is 17 cm, a source visibly misplaced.",
    ),
    "band_alignment": Limit(
        0.5,
        1.0,
        "ms",
        "The two sides of the crossover add over the onset. Half a period of 1 kHz is 0.5 ms: "
        "beyond it they cancel at the join, and a few milliseconds are heard as bass before "
        "the consonant.",
    ),
    "direct_level": Limit(
        3.0,
        6.0,
        "dB",
        "The convention (clip-library.md): the direct sound is the clip times the source's "
        "gain and directivity over the distance. The source's signature colours it by up to "
        "3 dB; 6 dB is a voice twice as near or as far as drawn.",
    ),
    "direction": Limit(
        3.0,
        10.0,
        "degrees",
        "The minimum audible angle is 1 to 2 degrees ahead and 5 to 10 to the side; a cell's "
        "node 2 cm off a head 0.8 m from a mouth is 1.4 degrees.",
    ),
    "pre_arrival_energy": Limit(
        -40.0,
        -30.0,
        "dB",
        "What comes more than 1 ms before the first arrival, over the whole response. The "
        "crossover's masks are zero phase and ring 40 dB down by design; backward masking "
        "covers a few milliseconds only, and -30 dB before a consonant is heard as a smear.",
        "max",
    ),
    "late_echo": Limit(
        -60.0,
        -40.0,
        "dB",
        "A response decays. Where its level per 100 ms rises again by more than 6 dB, the "
        "level it rises to, re its loudest 100 ms. Under -60 dB it is below what a "
        "reverberation time is read on; over -40 dB a second later it is an echo in the "
        "silence after a word.",
        "max",
    ),
    "seam_third_octaves": Limit(
        3.0,
        6.0,
        "dB",
        "The third octaves from 630 Hz to 1.6 kHz against the line through 315-500 Hz and "
        "2-3.15 kHz. One point of a room varies by 3 dB a third octave; 6 dB at 1 kHz is a "
        "colouration of speech anyone hears.",
    ),
    "reverberation_reference": Limit(
        20.0,
        35.0,
        "per cent",
        "T20 per octave from 250 Hz to 4 kHz against the validated field's at the nearest "
        "lattice point, worst band. The just noticeable difference is 5 to 10 per cent; the "
        "reference has another source position, which moves it by 10 to 15 in a dwelling.",
    ),
    "late_spectrum_reference": Limit(
        4.0,
        8.0,
        "dB",
        "The reverberant part's third octaves from 250 Hz to 4 kHz, each less the mean, "
        "against the validated field's at the nearest lattice point: the room's colour, "
        "which does not depend on the source's place. Worst band.",
    ),
    "tone_click": Limit(
        -50.0,
        -40.0,
        "dB",
        "What is left of a steady tone once the tone is removed, re its amplitude, worst "
        "sample. A 50 ms crossfade from nothing to full leaves -58 dB; a step of 1 per cent "
        "leaves -40 dB, which under a tone is a tick.",
        "max",
    ),
    "noise_click": Limit(
        6.5,
        8.0,
        "sigma",
        "The largest sample to sample difference of the rendered pink noise over the local "
        "rms of the differences. Gaussian noise passes 6.5 once in 1.2e10 samples (2e-5 a "
        "probe) and 8 once in 8e14.",
        "max",
    ),
    "level_step": Limit(
        1.0,
        3.0,
        "dB",
        "The level of a tone between one 10 ms frame and the next but one, where it is "
        "within 15 dB of its median. 1 dB is the just noticeable step of level; a step of "
        "3 dB in 20 ms is heard as a switch.",
        "max",
    ),
    "zipper": Limit(
        -40.0,
        -26.0,
        "dB",
        "Lines 20 Hz apart (the step rate) or 2 Hz apart (the run rate) round a tone, re "
        "the tone. -40 dB is a modulation of 2 per cent, the smallest heard at 20 Hz; "
        "-26 dB is 10 per cent, a flutter.",
        "max",
    ),
    "doppler": Limit(
        20.0,
        50.0,
        "per cent",
        "The measured shift of a 2.5 kHz tone against the radial speed's, worst 100 ms "
        "frame, as a share of the largest shift.",
    ),
    "level_distance": Limit(
        6.0,
        10.0,
        "dB per decade",
        "The slope of the level against the logarithm of the distance, less the -20 of a "
        "free field, read where the direct sound leads; a room flattens it, so only a "
        "steeper slope or a rise counts.",
    ),
    "level_at_listener": Limit(
        6.0,
        12.0,
        "dB",
        "The source's level at the listener less the clip's own level at 1 m, the gains "
        "and the distance: the convention's free field. Facing the listener and in a "
        "room a source gains up to 6 dB; beyond 12 dB either way the scale is wrong.",
    ),
    "binaural_peak": Limit(
        -3.0,
        0.0,
        "dB re full scale",
        "The largest sample of the two ears at the page's default level "
        f"({measure.PAGE_DEFAULT_LEVEL_DB:g} dB, at which the output's full scale stands for "
        f"{measure.FULL_SCALE_SPL_DB - measure.PAGE_DEFAULT_LEVEL_DB:g} dB SPL). Past full "
        "scale the page's output clips; 3 dB are kept for the sources the window does not "
        "hold.",
        "max",
    ),
    "dc": Limit(
        -40.0,
        -20.0,
        "dB",
        "The mean over the rms. A constant is not heard, but it takes headroom and "
        "clicks when the stream starts and stops.",
        "max",
    ),
    "silence": Limit(
        -90.0,
        -60.0,
        "dB",
        "The largest sample where the recipe gives the source nothing to say (from 1.4 s "
        "after an interval to 0.1 s before the next), re the stem's peak.",
        "max",
    ),
    "interval_edge": Limit(
        -40.0,
        -20.0,
        "dB",
        "The dry signal's first and last sample of an interval re the interval's rms: a "
        "source that starts or stops away from zero is a click on 64 channels.",
        "max",
    ),
}


@dataclass
class Result:
    """One test on one source: its number, what it is held to, and the verdict."""

    test: str
    source: str
    status: str
    value: float | None
    unit: str
    threshold: str
    reason: str
    note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        plain: dict[str, Any] = _plain(asdict(self))
        return plain


def _plain(value: Any) -> Any:
    """JSON's own types: arrays to lists, NaN and infinities to ``None``."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def judged(
    test: str, source: str, value: float, *, note: str = "", cap: str = FAIL, **detail: Any
) -> Result:
    """``value`` against :data:`LIMITS` of ``test``; ``cap`` is the worst it may be called."""
    limit = LIMITS[test.split(":")[0]]
    status = limit.judge(value)
    if _RANK[status] > _RANK[cap]:
        status = cap
    return Result(
        test, source, status, float(value), limit.unit, limit.text(), limit.reason, note, detail
    )


def said(test: str, source: str, status: str, note: str, **detail: Any) -> Result:
    """A result with no number of its own: something skipped, or told."""
    value = detail.pop("value", None)
    unit = str(detail.pop("unit", ""))
    threshold = str(detail.pop("threshold", ""))
    reason = str(detail.pop("reason", ""))
    return Result(test, source, status, value, unit, threshold, reason, note, detail)


def worst(results: Iterable[Result]) -> str:
    """The verdict of several: the worst of them."""
    status = PASS
    for result in results:
        if _RANK[result.status] > _RANK[status]:
            status = result.status
    return status


@dataclass
class CheckSettings:
    """What a check is run with."""

    #: Seconds of steady signal analysed per continuity probe.
    probe_seconds: float = 5.0
    #: Impulse probes per source that moves; one is enough at rest.
    probes_moving: int = 3
    #: The validated field's W channel, its positions and scene, or ``None``.
    reference: Mapping[str, Any] | None = None
    #: The page's level control at its default, as a gain
    #: (:data:`reverberate.render.check.measure.PAGE_DEFAULT_LEVEL_DB`).
    page_gain: float = measure.PAGE_DEFAULT_GAIN
    #: Threads of the engine's transforms.
    workers: int = -1
    #: Told what is being rendered.
    say: Callable[[str], None] = field(default=lambda text: None)

    def render(self) -> RenderSettings:
        return RenderSettings(workers=self.workers)


# --- the pack alone -----------------------------------------------------------------------


def duplicate_arrivals(pack: ScenePack, source: Source, steps: Iterable[int]) -> Result:
    """Arrivals written twice: same delay, same direction, same gain, another name.

    One reflection counted twice is 6 dB louder than it is. Read on the
    pack, because in the sound it is only a reflection too loud.
    """
    early = source.early
    found: dict[tuple[int, int], dict[str, Any]] = {}
    seen = 0
    over_direct = 0.0
    for k in steps:
        rows = early.rows(int(k))
        if rows.stop - rows.start < 2:
            continue
        seen += 1
        delay = np.asarray(early.delay_s[rows], dtype=float)
        order = np.argsort(delay, kind="stable")
        arrival = np.asarray(early.arrival[rows], dtype=float)[order]
        gain = np.asarray(early.gain[rows], dtype=float)[order]
        kinds = np.asarray(early.kind[rows])[order]
        ids = np.asarray(early.path_id[rows])[order]
        delay = delay[order]
        direct = gain[kinds == KIND_DIRECT].max(initial=0.0)
        for i in np.flatnonzero(np.diff(delay) < 1e-6):
            same = np.linalg.norm(arrival[i] - arrival[i + 1]) < 1e-3 and np.allclose(
                gain[i], gain[i + 1], rtol=1e-4
            )
            if same:
                if direct > 0.0:
                    over_direct = max(over_direct, float(2.0 * gain[i].max() / direct))
                key = (int(ids[i]), int(ids[i + 1]))
                found.setdefault(
                    key,
                    {
                        "path_ids": [f"{int(ids[i]):016x}", f"{int(ids[i + 1]):016x}"],
                        "delay_ms": float(delay[i] * 1e3),
                        "order": int(np.asarray(early.order[rows])[order][i]),
                        "arrival": arrival[i],
                        "first_step": int(k),
                    },
                )
    reason = (
        "A reflection written twice is 6 dB too loud; where the pair together passes the "
        "direct sound the first wave front is no longer the loudest and the voice is "
        "pulled towards the wall."
    )
    if not found:
        return said(
            "duplicate_arrivals",
            source.id,
            PASS,
            f"none in {seen} steps",
            value=0.0,
            unit="pairs",
            threshold="0 pairs",
            reason=reason,
        )
    status = FAIL if over_direct > 1.0 else WARN
    return said(
        "duplicate_arrivals",
        source.id,
        status,
        f"{len(found)} pairs of rows are one arrival; the loudest pair is "
        f"{measure.db(over_direct):+.1f} dB re the direct sound",
        value=float(len(found)),
        unit="pairs",
        threshold="0 pairs; fail when a pair passes the direct sound",
        reason=reason,
        pairs=list(found.values())[:8],
        pair_over_direct_db=float(measure.db(over_direct)),
    )


# --- impulse probes -----------------------------------------------------------------------


def _steps_audible(source: Source, first: int, last: int, need: int) -> list[int]:
    """Steps in ``[first, last)`` from which ``need`` steps on end are audible."""
    heard = np.asarray(source.audible, dtype=bool)
    run = np.zeros(heard.size + 1, dtype=int)
    for k in range(heard.size - 1, -1, -1):
        run[k] = run[k + 1] + 1 if heard[k] else 0
    return [k for k in range(first, min(last, heard.size - 1)) if run[k] >= need]


def _moves(pack: ScenePack, source: Source, steps: list[int]) -> bool:
    if not steps:
        return False
    at = np.asarray(steps)
    span = np.ptp(np.asarray(source.position)[at] - np.asarray(pack.listener.position)[at], axis=0)
    return bool(np.linalg.norm(span) > 0.01)


def _at(track: np.ndarray, sample: int, step: int) -> np.ndarray:
    """A per step track at ``sample``, linear between steps."""
    k = min(sample // step, track.shape[0] - 2)
    u = sample / step - k
    return np.asarray((1.0 - u) * track[k] + u * track[k + 1])


def _pose_at(pack: ScenePack) -> Callable[[int], tuple[float, float, float]]:
    orientation = np.asarray(pack.listener.orientation, dtype=float)
    step = pack.header.step_samples

    def pose(sample: int) -> tuple[float, float, float]:
        yaw, pitch, roll = _at(orientation, max(sample, 0), step)
        return float(yaw), float(pitch), float(roll)

    return pose


def _impulse(
    pack: ScenePack, source: Source, sample: int, lo: int, hi: int, settings: CheckSettings
) -> dict[str, np.ndarray]:
    """The scene's response to one sample of ``source`` at ``sample``, part by part."""
    rate = pack.header.sample_rate_hz
    dry = DryTrack.from_array(np.array([1.0]), start_s=sample / rate, rate=rate)
    engine = Engine(pack, {source.id: dry}, settings=settings.render())
    return {
        name: engine.stem(source.id, lo, hi, parts=[name])
        for name in ("early", "low", "tail")
        if name in engine.source(source.id).parts
    }


def _window(block: np.ndarray, a: int, b: int, edge: int = 8) -> np.ndarray:
    """Samples ``[a, b)`` of ``block`` under a taper of ``edge`` samples at each end."""
    a, b = max(a, 0), min(b, block.shape[-1])
    out = np.array(block[..., a:b], dtype=float)
    ramp = np.sin(0.5 * np.pi * (np.arange(edge) + 0.5) / edge) ** 2
    if out.shape[-1] >= 2 * edge:
        out[..., :edge] *= ramp
        out[..., -edge:] *= ramp[::-1]
    return out


def _band_mean(spectrum: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    inside = (freqs >= low) & (freqs <= high)
    return float(np.sqrt(np.mean(np.abs(spectrum[inside]) ** 2)))


def _reference_at(
    reference: Mapping[str, Any] | None, pack: ScenePack, head: np.ndarray
) -> tuple[np.ndarray, float, str] | None:
    """The validated field's W response at the lattice point nearest ``head``."""
    if reference is None or reference.get("scene_id") != pack.header.scene_id:
        return None
    positions = np.asarray(reference["positions"], dtype=float)
    index = int(np.argmin(np.linalg.norm(positions - head[None, :], axis=1)))
    distance = float(np.linalg.norm(positions[index] - head))
    response = np.asarray(reference["response"](index), dtype=float)
    return response, distance, str(reference.get("name", "reference"))


def probe_source(
    pack: ScenePack,
    source: Source,
    step: int,
    settings: CheckSettings,
    decoder: BinauralDecoder | None,
) -> tuple[list[Result], dict[str, Any]]:
    """One impulse through ``source`` at ``step``: the results, and what the plots draw."""
    h = pack.header
    rate, n = h.sample_rate_hz, h.step_samples
    sample = step * n + n // 2
    pre = int(0.1 * rate)
    lo = max(sample - pre, 0)
    reach = h.low_samples / h.low_sample_rate_hz if h.has_low else 0.3
    if source.tail is not None:
        reach = max(reach, 1.2)
    hi = min(sample + int((reach + pack.mirror.lead_s + 0.15) * rate), h.samples)
    parts = _impulse(pack, source, sample, lo, hi, settings)
    full = sum(parts.values())
    assert isinstance(full, np.ndarray)
    w = full[0]
    name = f"{source.id}@{sample / rate:.2f}s"
    results: list[Result] = []
    if not np.any(w):
        return [said("impulse", name, FAIL, "the response is silent")], {}

    # The geometry now, and what the pack holds of it.
    head = _at(np.asarray(pack.listener.position, dtype=float), sample, n)
    mouth = _at(np.asarray(source.position, dtype=float), sample, n)
    distance = float(np.linalg.norm(mouth - head))
    rows = source.early.rows(step)
    delays = np.sort(np.asarray(source.early.delay_s[rows], dtype=float))
    kinds = np.asarray(source.early.kind[rows])
    direct = bool(np.any(kinds == KIND_DIRECT))
    first_s = distance / h.sound_speed_m_s if direct else float(delays[0])
    expected = (sample - lo) + (pack.mirror.lead_s + first_s) * rate
    geometric = scene_to_ambisonic((mouth - head)[None, :])[0] / max(distance, 1e-9)
    if not direct:
        order = np.argsort(np.asarray(source.early.delay_s[rows], dtype=float))
        geometric = scene_to_ambisonic(np.asarray(source.early.arrival[rows], float)[order[:1]])[0]
    seen = "" if direct else "no direct path at this step: read against the pack's first arrival"
    cap = FAIL if direct else WARN

    high = parts["early"][0] + (parts["tail"][0] if "tail" in parts else 0.0)
    arrival = measure.envelope_arrival(parts["early"][0])
    results.append(
        judged(
            "arrival_time",
            name,
            (arrival - expected) / rate * 1e3,
            note=seen,
            cap=cap,
            distance_m=distance,
            expected_ms=(expected - (sample - lo)) / rate * 1e3,
            lead_ms=pack.mirror.lead_s * 1e3,
        )
    )
    if "low" in parts:
        low_arrival = measure.envelope_arrival(parts["low"][0])
        results.append(
            judged(
                "band_alignment",
                name,
                (low_arrival - arrival) / rate * 1e3,
                note="the band under the crossover against the band over it, first arrival",
                low_ms=(low_arrival - (sample - lo)) / rate * 1e3,
                high_ms=(arrival - (sample - lo)) / rate * 1e3,
            )
        )

    # The direct sound alone: from half a millisecond before it to the next arrival.
    gap = int((delays[1] - delays[0]) * rate) if delays.size > 1 else 96
    a, b = arrival - 24, arrival + int(np.clip(gap - 10, 24, 72))
    freqs = np.fft.rfftfreq(4800, 1.0 / rate)
    got = _band_mean(np.fft.rfft(_window(w, a, b), 4800), freqs, 2000.0, 8000.0)
    gain = 10.0 ** (source.gain_db / 20.0)
    pattern = 1.0
    directive = source.directivity_enabled and source.directivity_model in pack.directivity
    if directive and direct:
        table = directivity_gain(
            pack.directivity[source.directivity_model],
            (head - mouth)[None, :],
            float(_at(np.asarray(source.yaw_deg, dtype=float), sample, n)),
        )[0]
        bands = [i for i, f in enumerate(h.bands_hz) if 2000 <= f <= 8000]
        pattern = float(np.sqrt(np.mean(table[bands] ** 2)))
    wanted = gain * pattern / max(distance, 1e-9)
    signature = _band_mean(np.fft.rfft(pack.mirror.signature, 4800), freqs, 2000.0, 8000.0)
    scale_db = float(source.level.high_gain_db[step]) + float(measure.db(signature))
    level = float(measure.db(got / wanted))
    if direct:
        results.append(
            judged(
                "direct_level",
                name,
                level,
                note=(
                    f"the pack's own scale (high_gain_db and the signature) is {scale_db:+.1f} dB: "
                    f"without it the direct sound is {level - scale_db:+.1f} dB from the convention"
                ),
                measured_db=float(measure.db(got)),
                convention_db=float(measure.db(wanted)),
                source_gain_db=source.gain_db,
                directivity_db=float(measure.db(pattern)),
                pack_scale_db=scale_db,
                after_pack_scale_db=level - scale_db,
                window_ms=(b - a) / rate * 1e3,
            )
        )
    if direct and "low" in parts:
        # The direct sound under the crossover: the envelope's peak against the peak of
        # the convention's impulse through the crossover's low side, with no directivity.
        from scipy.signal import hilbert

        low_mask, _ = pack.crossover.masks(57600, rate, power=False)
        unit = float(np.max(np.abs(np.fft.irfft(low_mask, n=57600))))
        envelope = np.abs(hilbert(parts["low"][0]))
        near = slice(max(low_arrival - 48, 0), low_arrival + 48)
        low_level = float(measure.db(envelope[near].max() / (gain * unit / max(distance, 1e-9))))
        balance = low_level - level
        results.append(
            said(
                "direct_band_balance",
                name,
                PASS if abs(balance) <= 3.0 else WARN if abs(balance) <= 6.0 else FAIL,
                "the direct sound under the crossover against the direct sound over it, each re "
                "the convention: omnidirectional under (the low band has no directivity, by "
                "design), with the source's pattern over",
                value=balance,
                unit="dB",
                threshold="|x| <= 3 dB; warn to 6",
                reason="A voice whose band under 1 kHz is 3 dB from its band over it is coloured; "
                "the signature's own ripple is within that. 6 dB is a telephone or a blanket.",
                low_direct_db=low_level,
                high_direct_db=level,
                directivity_over_the_crossover_db=float(measure.db(pattern)),
            )
        )
    heard = measure.direction_of(_window(full[:4], a, b))
    angle = float(np.degrees(np.arccos(np.clip(np.dot(heard, geometric), -1.0, 1.0))))
    results.append(
        judged(
            "direction",
            name,
            angle,
            note=seen,
            cap=cap,
            heard=heard,
            geometric=geometric,
        )
    )
    if decoder is not None:
        results.extend(_ears(name, _window(full, a, b), geometric, decoder, pack, sample))

    before = max(arrival - int(0.001 * rate), 0)
    total = float(np.sum(w * w))
    shares: dict[str, Any] = {
        f"{part}_db": float(measure.db(np.sum(x[0][:before] ** 2) / total, power=True))
        for part, x in parts.items()
    }
    results.append(
        judged(
            "pre_arrival_energy",
            name,
            float(measure.db(np.sum(w[:before] ** 2) / total, power=True)),
            **shares,
        )
    )

    frames = measure.frame_levels_db(w[max(arrival - 48, 0) :], int(0.1 * rate))
    rise = np.diff(frames)
    back = int(np.argmax(rise)) + 1 if rise.size and float(rise.max()) > 6.0 else 0
    risen: dict[str, Any] = {
        f"{part}_at_the_rise_db": float(
            measure.frame_levels_db(x[0][max(arrival - 48, 0) :], int(0.1 * rate))[back]
            - frames.max()
        )
        for part, x in parts.items()
        if back
    }
    results.append(
        judged(
            "late_echo",
            name,
            float(frames[back] - frames.max()) if back else float("-inf"),
            note=f"the level rises by {float(rise.max()):.0f} dB {0.1 * back:.1f} s after the "
            "first arrival"
            if back
            else "the level per 100 ms never rises by 6 dB",
            level_per_100_ms_db=frames - frames.max(),
            **risen,
        )
    )

    # The spectrum: the whole response, and its first 50 ms.
    split = arrival + int(0.05 * rate)
    whole = measure.third_octave_db(w, rate)
    first = measure.third_octave_db(w[: min(split, w.size)], rate)
    late = measure.third_octave_db(w[min(split, w.size) :], rate)
    worst_whole, seam_whole = measure.seam_deviation_db(whole)
    worst_early, seam_early = measure.seam_deviation_db(first)
    sides: dict[str, Any] = {
        "low_side_db": float(measure.db(np.sum(parts["low"][0] ** 2), power=True))
        if "low" in parts
        else None,
        "high_side_db": float(measure.db(np.sum(high**2), power=True)),
    }
    centres = np.asarray(measure.THIRD_OCTAVES_HZ)
    octave: dict[str, Any] = {
        f"octave_{f}_db": float(
            measure.db(
                np.sum(10.0 ** (whole[(centres > f * 0.7) & (centres < f * 1.42)] / 10.0)),
                power=True,
            )
        )
        for f in (500, 1000, 2000)
    }
    results.append(
        judged("seam_third_octaves:whole", name, worst_whole, seam_db=seam_whole, **sides, **octave)
    )
    results.append(
        judged(
            "seam_third_octaves:early",
            name,
            worst_early,
            cap=WARN,
            seam_db=seam_early,
            note="the first 50 ms: one comb of early reflections moves a third octave, so "
            "this warns and does not fail",
        )
    )

    # The decay, against the validated field and against a dwelling.
    # Read from 50 ms after the first arrival, on both: before it the curve is the direct
    # sound's cliff, which is the source's distance and not the room.
    t20, t30 = measure.decay_times_s(w, rate, split)
    c50 = measure.c50_db(w, rate, arrival)
    bank = np.asarray(band_centres(int(rate)))
    middle = (bank >= 250) & (bank <= 4000)
    plots: dict[str, Any] = {
        "name": name,
        "whole_db": whole,
        "early_db": first,
        "late_db": late,
        "response": w,
        "arrival": arrival,
        "rate": rate,
    }
    found = _reference_at(settings.reference, pack, head)
    if found is not None:
        response, off, label = found
        ref_arrival = measure.envelope_arrival(response)
        ref20, ref30 = measure.decay_times_s(response, rate, ref_arrival + int(0.05 * rate))
        both = middle & np.isfinite(t20) & np.isfinite(ref20)
        if np.any(both):
            departure = 100.0 * (t20[both] / ref20[both] - 1.0)
            results.append(
                judged(
                    "reverberation_reference",
                    name,
                    float(departure[int(np.argmax(np.abs(departure)))]),
                    note=f"{label}, lattice point {off:.2f} m from the head",
                    bands_hz=bank[both],
                    t20_s=t20[both],
                    reference_t20_s=ref20[both],
                    t30_s=t30[both],
                    reference_t30_s=ref30[both],
                )
            )
        ref_late = measure.third_octave_db(response[ref_arrival + int(0.05 * rate) :], rate)
        band = (centres >= 240.0) & (centres <= 4100.0)
        shape = (late[band] - late[band].mean()) - (ref_late[band] - ref_late[band].mean())
        under, over = (centres[band] < 700.0), (centres[band] > 1500.0)
        balance = float(shape[under].mean() - shape[over].mean())
        results.append(
            judged(
                "late_spectrum_reference",
                name,
                float(shape[int(np.argmax(np.abs(shape)))]),
                note=f"{label}, lattice point {off:.2f} m from the head; under the crossover "
                f"(250 to 630 Hz) against over it (1.6 to 4 kHz) the late part is "
                f"{balance:+.1f} dB from the reference's",
                third_octaves_hz=centres[band],
                difference_db=shape,
                low_over_high_db=balance,
            )
        )
        plots["reference_late_db"] = ref_late
        plots["reference_response"] = response
        plots["reference_arrival"] = ref_arrival
    inside = t20[middle & np.isfinite(t20)]
    plausible = inside.size > 0 and float(inside.min()) >= 0.15 and float(inside.max()) <= 1.0
    results.append(
        said(
            "reverberation_plausible",
            name,
            PASS if plausible else WARN if pack.header.has_tail else SKIP,
            "T20 from 250 Hz to 4 kHz, per octave"
            if pack.header.has_tail
            else "no tail in the pack",
            value=float(np.nanmax(t20[middle])) if inside.size else None,
            unit="s",
            threshold="0.15 s <= T20 <= 1.0 s in every octave",
            reason="A furnished dwelling's rooms read 0.3 to 0.6 s; under 0.15 s is a booth, over "
            "1 s an empty hall.",
            bands_hz=bank,
            t20_s=t20,
            t30_s=t30,
        )
    )
    speech = c50[(bank >= 500) & (bank <= 4000)]
    close = distance <= 2.5 and direct
    results.append(
        said(
            "c50",
            name,
            INFO if not close else PASS if 0.0 <= float(speech.min()) <= 40.0 else WARN,
            "the early to late ratio per octave, 500 Hz to 4 kHz"
            + ("" if close else "; told only"),
            value=float(speech.min()),
            unit="dB",
            threshold="0 dB <= C50 <= 40 dB for a source it sees within 2.5 m",
            reason="Under 0 dB a voice at arm's length would be lost in its own room; over 40 dB "
            "there is no room.",
            bands_hz=bank,
            c50_db=c50,
        )
    )
    plots.update({"bank_hz": bank, "t20_s": t20})
    return results, plots


def _plane_wave(decoder: BinauralDecoder, direction: np.ndarray) -> np.ndarray:
    """The decoder's two ears for a plane wave from ``direction`` in the head's frame."""
    return np.asarray(np.einsum("ect,c->et", decoder.filters, real_sh(decoder.order, direction)[0]))


def _ild_spectrum(ears: np.ndarray, rate: float) -> np.ndarray:
    """Left over right per third octave from 2 to 12.5 kHz, dB."""
    centres = np.asarray(measure.THIRD_OCTAVES_HZ)
    band = (centres >= 1900.0) & (centres <= 13000.0)
    padded = np.zeros((2, 4800))
    padded[:, : min(ears.shape[1], 4800)] = ears[:, :4800]
    left, right = (measure.third_octave_db(ear, rate)[band] for ear in padded)
    return np.asarray(left - right)


def _ears(
    name: str,
    direct: np.ndarray,
    geometric: np.ndarray,
    decoder: BinauralDecoder,
    pack: ScenePack,
    sample: int,
) -> list[Result]:
    """Left and right, front and back, through the page's decoder, on the direct sound alone."""
    rate = pack.header.sample_rate_hz
    page = PageDecoder(decoder)
    azimuth = float(np.degrees(np.arctan2(geometric[1], geometric[0])))
    padded = np.concatenate([direct, np.zeros((direct.shape[0], 2 * BLOCK))], axis=1)

    def heard(head_yaw: float) -> np.ndarray:
        return page.decode(padded, lambda _: (head_yaw, 0.0, 0.0))

    sides = {}
    for label, offset in (("left", 60.0), ("right", -60.0)):
        ears = heard(azimuth - offset)
        sides[label] = {"itd_ms": itd_s(ears, rate) * 1e3, "ild_db": ild_db(ears)}
    yaw, pitch, roll = _pose_at(pack)(sample)
    own = geometric @ head_matrix(yaw, pitch, roll)
    scene = heard(yaw) if pitch == 0.0 and roll == 0.0 else page.decode(padded, _pose_at(pack))
    sides["scene"] = {
        "itd_ms": itd_s(scene, rate) * 1e3,
        "ild_db": ild_db(scene),
        "azimuth_deg": float(np.degrees(np.arctan2(own[1], own[0]))),
    }
    signs = (
        sides["left"]["itd_ms"] > 0.0
        and sides["left"]["ild_db"] > 0.0
        and sides["right"]["itd_ms"] < 0.0
        and sides["right"]["ild_db"] < 0.0
    )
    margin = min(sides["left"]["ild_db"], -sides["right"]["ild_db"])
    sized = all(0.25 <= abs(sides[s]["itd_ms"]) <= 0.9 for s in ("left", "right")) and margin >= 3.0
    results = [
        said(
            "binaural_left_right",
            name,
            PASS if signs and sized else WARN if signs else FAIL,
            "the head turned so that the source is 60 degrees to its left, then to its right: "
            "the near ear leads and is the louder",
            value=float(margin),
            unit="dB",
            threshold="the near ear leads by 0.25 to 0.9 ms and is louder by 3 dB at least",
            reason="A sphere's delay at 60 degrees is 0.49 ms (Woodworth). A sign the other way is "
            "left and right exchanged, in the engine's frame or in the decoder.",
            head=decoder.head,
            **sides,
        )
    ]
    # Front and back: the ears' level difference per third octave against the decoder's own
    # for a plane wave 30 degrees to the front left, and for its image behind.
    front = np.array([np.cos(np.radians(30.0)), np.sin(np.radians(30.0)), 0.0])
    back = front * np.array([-1.0, 1.0, 1.0])
    got = _ild_spectrum(heard(azimuth - 30.0), rate)
    to_front = float(
        np.sqrt(np.mean((got - _ild_spectrum(_plane_wave(decoder, front), rate)) ** 2))
    )
    to_back = float(np.sqrt(np.mean((got - _ild_spectrum(_plane_wave(decoder, back), rate)) ** 2)))
    apart = float(
        np.sqrt(
            np.mean(
                (
                    _ild_spectrum(_plane_wave(decoder, front), rate)
                    - _ild_spectrum(_plane_wave(decoder, back), rate)
                )
                ** 2
            )
        )
    )
    if apart < 1.5:
        status, note = SKIP, "this head's front and back differ by under 1.5 dB: it cannot tell"
    else:
        status = (
            PASS if to_front < to_back and to_front <= 2.0 else WARN if to_front < to_back else FAIL
        )
        note = "the source put 30 degrees to the front left: the ears read the front, not its image"
    results.append(
        said(
            "binaural_front_back",
            name,
            status,
            note,
            value=to_front,
            unit="dB",
            threshold="nearer the front's pattern than the back's, and within 2 dB rms of it",
            reason="Front and back differ at the ears by the pinna alone, in the level per third "
            "octave from 2 to 12.5 kHz; a field mirrored front to back reads the other pattern.",
            to_front_db=to_front,
            to_back_db=to_back,
            front_from_back_db=apart,
        )
    )
    return results


# --- continuity ---------------------------------------------------------------------------


def _events(pack: ScenePack, source: Source) -> tuple[np.ndarray, list[set[str]]]:
    """Per step, how much of the pack changes from the step before, and what."""
    steps = pack.header.steps
    what: list[set[str]] = [set() for _ in range(steps)]
    move = np.linalg.norm(np.diff(np.asarray(source.position, dtype=float), axis=0), axis=1)
    walk = np.linalg.norm(np.diff(np.asarray(pack.listener.position, dtype=float), axis=0), axis=1)
    for k in np.flatnonzero(move > 1e-4) + 1:
        what[k].add("the source moves")
    for k in np.flatnonzero(walk > 1e-4) + 1:
        what[k].add("the listener moves")
    if source.low is not None:
        low = source.low
        for label, track in (("mode", low.mode), ("pair", low.pair), ("cell", low.cell)):
            track = np.asarray(track).reshape(steps, -1)
            for k in np.flatnonzero(np.any(track[1:] != track[:-1], axis=1)) + 1:
                what[k].add(f"the low band changes {label}")
    if source.tail is not None:
        hist = np.asarray(source.tail.hist).reshape(steps, -1)
        for k in np.flatnonzero(np.any(hist[1:] != hist[:-1], axis=1)) + 1:
            what[k].add("the tail changes histogram")
    offsets = np.asarray(source.early.offsets)
    ids = np.asarray(source.early.path_id)
    for step in range(1, steps):
        before = ids[offsets[step - 1] : offsets[step]]
        now = ids[offsets[step] : offsets[step + 1]]
        if before.size != now.size or np.any(before != now):
            what[step].add("a path is born or dies")
    return np.array([len(w) for w in what]), what


def _stretch(
    pack: ScenePack, source: Source, first: int, last: int, lead: int, steps: int
) -> tuple[int, int] | None:
    """The audible run of ``lead + steps`` steps in which the most changes after ``lead``."""
    count, _ = _events(pack, source)
    heard = np.asarray(source.audible, dtype=bool)
    best: tuple[int, int, int] | None = None
    k = first
    while k < last:
        if not heard[k]:
            k += 1
            continue
        end = k
        while end < last and heard[end]:
            end += 1
        # Two steps are kept before the run's end: there the pack stops, not the scene.
        length = min(steps, end - k - lead - 2)
        if length >= 20:
            for start in range(k, end - lead - length + 1, 5):
                score = int(count[start + lead : start + lead + length].sum())
                if best is None or score > best[0]:
                    best = (score, start, length)
        k = end
    return None if best is None else (best[1], best[2])


def _where(pack: ScenePack, what: list[set[str]], sample: int) -> str:
    """Where a fault lies: its time, its place in the step and the run, what the pack changes."""
    n = pack.header.step_samples
    step = sample // n
    near = sorted(what[min(step, len(what) - 1)] | what[min(step + 1, len(what) - 1)])
    return (
        f"at {sample / pack.header.sample_rate_hz:.4f} s, sample {sample % n} of step {step}"
        f" ({sample % (10 * n)} of its run of ten)"
        + (": " + ", ".join(near) if near else ": nothing changes in the pack there")
    )


def continuity_of(
    pack: ScenePack,
    source: Source,
    first: int,
    last: int,
    settings: CheckSettings,
    decoder: BinauralDecoder | None,
) -> tuple[list[Result], dict[str, Any]]:
    """Steady tones, then pink noise, through ``source`` where the most changes."""
    h = pack.header
    rate, n = h.sample_rate_hz, h.step_samples
    lead = int(np.ceil(1.35 / h.step_s))
    chosen = _stretch(
        pack, source, first, last, lead, int(round(settings.probe_seconds / h.step_s))
    )
    if chosen is None:
        note = "no stretch of 2.4 s in which the source is audible throughout"
        return [said("continuity", source.id, SKIP, note)], {}
    start, length = chosen
    _, what = _events(pack, source)
    s0, s1 = start * n, (start + lead + length) * n
    a0 = (start + lead) * n  # the first sample analysed: the room has filled
    count = s1 - s0
    time = np.arange(count) / rate
    fade = np.minimum(1.0, np.arange(count) / (0.02 * rate))
    tones = 0.25 * fade * sum(np.sin(2.0 * np.pi * f * time) for f in TONES_HZ)
    noise = fade * measure.pink_noise(count, rate, seed=1)
    pose = _pose_at(pack)
    results: list[Result] = []
    name = source.id
    changes = sorted(set().union(*what[start + lead : start + lead + length]))
    span = f"{a0 / rate:.2f} to {s1 / rate:.2f} s"
    told = f"{span}; in it: " + (", ".join(changes) if changes else "nothing changes (at rest)")

    def rendered(dry: np.ndarray, ears: bool) -> np.ndarray:
        track = DryTrack.from_array(dry, start_s=s0 / rate, rate=rate)
        engine = Engine(pack, {source.id: track}, settings=settings.render())
        stem = engine.stem(source.id, a0 - BLOCK, s1)
        out = [stem[:4, BLOCK:]]
        if ears and decoder is not None:
            page = PageDecoder(decoder)
            out.append(page.decode(stem[:, BLOCK:], pose, start=a0, before=stem[:, :BLOCK]))
        return np.concatenate(out, axis=0)

    settings.say(f"{name}: tones over {span}")
    toned = rendered(np.asarray(tones), True)
    trim = 1200
    labels = ["W", "X", "Y", "Z", "left ear", "right ear"][: toned.shape[0]]
    worst_click: tuple[float, str, int, float] = (float("-inf"), "", 0, 0.0)
    steps_db: tuple[float, int, float] = (0.0, 0, 0.0)
    lines: dict[str, Any] = {}
    zip_worst = (float("-inf"), "")
    carriers = {}
    for index, tone in enumerate(TONES_HZ):
        halves = [measure.band_split(channel, rate, 1000.0)[index][trim:-trim] for channel in toned]
        scale = {
            "field": float(np.sqrt(2.0 * np.mean(halves[0] ** 2))),
            "ears": float(np.sqrt(2.0 * max(np.mean(x**2) for x in halves[4:])))
            if len(halves) > 4
            else 0.0,
        }
        carriers[tone] = halves[0]
        for label, half in zip(labels, halves, strict=True):
            amplitude = scale["ears" if "ear" in label else "field"]
            if amplitude <= 0.0:
                continue
            rest = np.abs(measure.annihilate(half, tone, rate))
            at = int(np.argmax(rest))
            size = float(measure.db(rest[at] / amplitude))
            if size > worst_click[0]:
                worst_click = (size, label, a0 + trim + at + 1, tone)
        step_db, at = measure.level_step_db(halves[0], rate)
        if step_db > steps_db[0]:
            steps_db = (step_db, a0 + trim + at, tone)
        for spacing in (1.0 / h.step_s, 1.0 / (10.0 * h.step_s)):
            bands = measure.sidebands_db(halves[0], rate, tone, spacing)
            lines[f"{tone:g} Hz, lines {spacing:g} Hz apart"] = asdict(bands)
            # A line counts when it stands 6 dB over what lies between the lines.
            if bands.prominence_db >= 6.0 and bands.level_db > zip_worst[0]:
                zip_worst = (bands.level_db, f"{spacing:g} Hz apart round {tone:g} Hz")
    results.append(
        judged(
            "tone_click",
            name,
            worst_click[0],
            note=f"{told}. Worst on {worst_click[1]}, the {worst_click[3]:g} Hz tone, "
            + _where(pack, what, worst_click[2]),
            channel=worst_click[1],
            sample=worst_click[2],
            tone_hz=worst_click[3],
            floor_on_the_dry_tone_db=measure.tone_residual_db(
                np.sin(2.0 * np.pi * TONES_HZ[1] * time[2400:]), TONES_HZ[1], rate
            )[0],
        )
    )
    results.append(
        judged(
            "level_step",
            name,
            steps_db[0],
            note=f"the {steps_db[2]:g} Hz tone, " + _where(pack, what, steps_db[1]),
        )
    )
    results.append(
        judged(
            "zipper",
            name,
            zip_worst[0],
            note=zip_worst[1] or "no line stands 6 dB over what lies between the lines",
            lines=lines,
        )
    )
    moving = _moving(pack, source, a0 + trim, s1 - trim)
    results.extend(_doppler(name, carriers[TONES_HZ[1]], moving, pack))

    settings.say(f"{name}: pink noise over {span}")
    noised = rendered(noise, False)[0]
    ratio, at = measure.difference_outlier(noised, rate)
    results.append(
        judged(
            "noise_click",
            name,
            ratio,
            note=f"{told}. Worst " + _where(pack, what, a0 + at),
            on_the_dry_noise=measure.difference_outlier(noise[lead * n :], rate)[0],
        )
    )
    results.extend(_distance_law(name, noised[trim:-trim], moving, rate))
    plots = {
        "name": name,
        "rate": rate,
        "start_s": (a0 + trim) / rate,
        "tone": carriers[TONES_HZ[1]],
        "residual": np.abs(measure.annihilate(carriers[TONES_HZ[1]], TONES_HZ[1], rate)),
        "noise": noised,
    }
    return results, plots


def _moving(pack: ScenePack, source: Source, a: int, b: int) -> dict[str, np.ndarray]:
    """The distance and the radial speed at every 100 ms frame of ``[a, b)``."""
    h = pack.header
    frame = int(0.1 * h.sample_rate_hz)
    centres = a + frame // 2 + frame * np.arange((b - a) // frame)
    gap = np.asarray(source.position, dtype=float) - np.asarray(pack.listener.position, dtype=float)
    distance = np.linalg.norm(gap, axis=1)
    at = centres / h.step_samples
    speed = np.gradient(distance, h.step_s)
    return {
        "distance_m": np.interp(at, np.arange(h.steps), distance),
        "speed_m_s": np.interp(at, np.arange(h.steps), speed),
    }


def _doppler(
    name: str, tone: np.ndarray, moving: dict[str, np.ndarray], pack: ScenePack
) -> list[Result]:
    speed = moving["speed_m_s"]
    if speed.size == 0 or float(np.max(np.abs(speed))) < 0.2:
        return [said("doppler", name, SKIP, "the distance changes by under 0.2 m/s")]
    h = pack.header
    heard = measure.instantaneous_hz(tone, h.sample_rate_hz)[: speed.size] - TONES_HZ[1]
    # Quasi-static: the delay is the distance now, so the tone is at f (1 - d'/c).
    wanted = -TONES_HZ[1] * speed[: heard.size] / h.sound_speed_m_s
    largest = float(np.max(np.abs(wanted)))
    error = 100.0 * (heard - wanted) / largest
    middle = float(np.median(np.abs(error)))
    return [
        judged(
            "doppler",
            name,
            middle,
            cap=WARN if h.has_tail or h.has_low else FAIL,
            note="the median frame; in a room the reflections carry other shifts, so this "
            "warns and does not fail"
            if h.has_tail or h.has_low
            else "the median frame",
            largest_shift_hz=largest,
            worst_frame_per_cent=float(np.max(np.abs(error))),
            heard_hz=heard,
            wanted_hz=wanted,
        )
    ]


def _distance_law(
    name: str, noise: np.ndarray, moving: dict[str, np.ndarray], rate: float
) -> list[Result]:
    distance = moving["distance_m"]
    if distance.size < 6 or float(distance.max() / max(distance.min(), 1e-6)) < 1.25:
        return [said("level_distance", name, SKIP, "the distance changes by under a quarter")]
    levels = measure.frame_levels_db(noise, int(0.1 * rate))[: distance.size]
    slope, _ = np.polyfit(np.log10(distance[: levels.size]), levels, 1)
    # A room may flatten the law (a slope between -20 and 0); it may not steepen or invert it.
    excess = float(slope) + 20.0
    value = excess if excess < 0.0 else max(float(slope), 0.0)
    return [
        judged(
            "level_distance",
            name,
            value,
            note=f"the level falls {-float(slope):.1f} dB per decade of distance over "
            f"{distance.min():.2f} to {distance.max():.2f} m",
            slope_db_per_decade=float(slope),
        )
    ]


# --- the mix ------------------------------------------------------------------------------


def _active(feed: Feed, delay_s: float, lo: int, hi: int, rate: float) -> np.ndarray:
    """Which samples of ``[lo, hi)`` lie in an interval of ``feed``, heard ``delay_s`` later."""
    mask = np.zeros(hi - lo, dtype=bool)
    for a, b in feed.intervals:
        first = int(round((a + delay_s) * rate)) - lo
        mask[max(first, 0) : max(int(round((b + delay_s) * rate)) - lo, 0)] = True
    return mask


def _speech_level(samples: np.ndarray, rate: float) -> float:
    from reverberate.scenes.clips import active_speech_level

    return float(active_speech_level(samples, rate)[0])


def mix_of(
    pack: ScenePack,
    sources: list[Source],
    recipe: Mapping[str, Any],
    clips: ClipSource,
    lo: int,
    hi: int,
    settings: CheckSettings,
    decoder: BinauralDecoder | None,
) -> tuple[list[Result], dict[str, Any]]:
    """The recipe's clips through the engine: levels, the two ears, silence, the edges.

    Returns the results and, for the files and the plots, each source's
    omnidirectional channel and two ears and those of the mix.
    """
    h = pack.header
    rate = h.sample_rate_hz
    feeds = {s.id: feed_of(pack, s.id, recipe, clips) for s in sources}
    engine = Engine(
        pack, {name: feed.track for name, feed in feeds.items()}, settings=settings.render()
    )
    count = hi - lo
    piece = 48 * BLOCK
    omni = {s.id: np.zeros(count, dtype=np.float32) for s in sources}
    ears = {s.id: np.zeros((2, count), dtype=np.float32) for s in sources}
    page = PageDecoder(decoder) if decoder is not None else None
    pose = _pose_at(pack)
    for source in sources:
        settings.say(f"{source.id}: the stem, {count / rate:.0f} s ({feeds[source.id].label})")
        before = np.zeros((h.channels, BLOCK))
        for a in range(lo, hi, piece):
            b = min(a + piece, hi)
            stem = engine.stem(source.id, a, b)
            if np.any(stem) or np.any(before):
                omni[source.id][a - lo : b - lo] = stem[0]
                if page is not None:
                    ears[source.id][:, a - lo : b - lo] = page.decode(
                        stem, pose, start=a, before=before
                    )
            before = stem[:, -BLOCK:] if stem.shape[1] >= BLOCK else np.zeros((h.channels, BLOCK))
    results: list[Result] = []
    mix = np.sum([omni[s.id] for s in sources], axis=0, dtype=np.float64)
    both = np.sum([ears[s.id] for s in sources], axis=0, dtype=np.float64)
    told: dict[str, Any] = {}
    for source in sources:
        feed, w = feeds[source.id], omni[source.id].astype(float)
        if not feed.intervals or not np.any(w):
            results.append(said("level_at_listener", source.id, SKIP, feed.label + ": silent here"))
            continue
        results.extend(_level(pack, source, feed, w, mix - w, lo, hi, told))
        results.extend(_quiet(pack, source, feed, w, lo, hi))
        results.extend(_edges(source, feed, rate))
    if page is not None:
        peak = float(np.max(np.abs(both))) * settings.page_gain
        results.append(
            judged(
                "binaural_peak",
                "mix",
                float(measure.db(peak)),
                note=f"{len(sources)} of the recipe's {len(recipe.get('sources', []))} sources",
                rms_db=float(measure.db(float(np.mean(both**2)), power=True)),
            )
        )
        loud = float(measure.db(peak))
        results.append(
            said(
                "playback_level",
                "mix",
                PASS if loud >= -30.0 else WARN if loud >= -40.0 else FAIL,
                "the peak of the two ears at the page's default level",
                value=loud,
                unit="dB re full scale",
                threshold="x >= -30; warn to -40",
                reason="The page's level control stops at +20 dB. Speech whose peaks lie under "
                "-40 dB re full scale cannot be brought to a listening level with it, and a 16 "
                "bit output keeps 9 bits of it.",
            )
        )
        mean = abs(float(np.mean(both))) / max(float(np.sqrt(np.mean(both**2))), 1e-300)
        results.append(judged("dc", "mix, two ears", float(measure.db(mean))))
    return results, {"omni": omni, "ears": ears, "mix_omni": mix, "mix_ears": both, "levels": told}


def _level(
    pack: ScenePack,
    source: Source,
    feed: Feed,
    w: np.ndarray,
    others: np.ndarray,
    lo: int,
    hi: int,
    told: dict[str, Any],
) -> list[Result]:
    h = pack.header
    rate = h.sample_rate_hz
    voice = "voice" in source.kind
    gap = np.asarray(source.position, dtype=float) - np.asarray(pack.listener.position, dtype=float)
    distance = np.linalg.norm(gap, axis=1)
    steps = np.zeros(h.steps, dtype=bool)
    for a, b in feed.intervals:
        steps[int(a / h.step_s) : int(np.ceil(b / h.step_s)) + 1] = True
    steps &= np.asarray(source.audible, dtype=bool)
    if not np.any(steps):
        return [said("level_at_listener", source.id, SKIP, "active only where the pack is silent")]
    # The distance of the mean intensity, and the delay the level is read after.
    effective = float(1.0 / np.sqrt(np.mean(1.0 / distance[steps] ** 2)))
    delay = pack.mirror.lead_s + effective / h.sound_speed_m_s
    active = _active(feed, delay, lo, hi, rate)
    dry = feed.track.read(lo, hi) * 10.0 ** (source.gain_db / 20.0)
    weighted = measure.a_weighted(w, rate)
    if voice:
        heard = _speech_level(w, rate) + measure.FULL_SCALE_SPL_DB
        heard_a = _speech_level(weighted, rate) + measure.FULL_SCALE_SPL_DB
        fed = _speech_level(dry, rate) + measure.FULL_SCALE_SPL_DB
    else:
        heard, heard_a = measure.spl_db(w[active]), measure.spl_db(weighted[active])
        fed = measure.spl_db(dry[_active(feed, 0.0, lo, hi, rate)])
    free = fed - 20.0 * np.log10(effective)
    fast = measure.frame_levels_db(weighted, int(0.125 * rate)) + measure.FULL_SCALE_SPL_DB
    rest = measure.a_weighted(others, rate)[active]
    rows = source.early.rows(int(np.flatnonzero(steps)[0]))
    seen = bool(np.any(np.asarray(source.early.kind[rows]) == KIND_DIRECT))
    told[source.id] = {
        "kind": source.kind,
        "distance_m": effective,
        "spl_db": heard,
        "spl_dba": heard_a,
        "fed_at_1m_db": fed,
        "free_field_db": free,
        "fast_max_dba": float(fast.max()),
        "rest_dba": measure.spl_db(rest) if np.any(others) else None,
        "fast_dba": fast,
        "fed": feed.label,
    }
    measured = "the active speech level (ITU-T P.56)" if voice else "the level while it sounds"
    results = [
        judged(
            "level_at_listener",
            source.id,
            heard - free,
            cap=FAIL if seen else INFO,
            note=f"{measured}: {heard:.1f} dB SPL ({heard_a:.1f} dBA) at {effective:.2f} m; fed "
            f"{fed:.1f} dB SPL at 1 m, so {free:.1f} dB in a free field"
            + ("" if seen else "; the listener does not see it, so this is told and not judged"),
            **{k: v for k, v in told[source.id].items() if k != "fast_dba"},
        )
    ]
    hearable = PASS if 20.0 <= heard_a <= 85.0 else WARN if 10.0 <= heard_a <= 85.0 else FAIL
    if voice and "near" in source.kind and not 45.0 <= heard_a <= 80.0:
        hearable = FAIL
    results.append(
        said(
            "audibility",
            source.id,
            hearable,
            f"{heard_a:.1f} dBA while it sounds, {float(fast.max()):.1f} dBA at most (125 ms)",
            value=heard_a,
            unit="dBA",
            threshold="20 <= x <= 85; warn from 10; a near voice 45 to 80",
            reason="Full scale is 86 dB SPL at 1 m. A quiet room is 20 to 30 dBA: under 10 nobody "
            "hears the source. Over 85 it is louder than anything a dwelling holds. A voice at "
            "arm's length is 55 to 65 dB at 1 m, so 45 to 80 at the head with the room and the "
            "distance.",
        )
    )
    if voice and np.any(others):
        ratio = heard_a - measure.spl_db(rest)
        alone = measure.spl_db(rest) < -20.0
        results.append(
            said(
                "speech_to_rest",
                source.id,
                INFO,
                "nothing else sounds while it speaks"
                if alone
                else "the voice's active level "
                "over everything else of the window while it speaks, A weighted",
                value=None if alone else ratio,
                unit="dB",
                threshold="told, not judged",
                reason="What the listener has to separate; a scene is meant to be hard.",
            )
        )
    mean = abs(float(np.mean(w[active]))) / max(float(np.sqrt(np.mean(w[active] ** 2))), 1e-300)
    results.append(judged("dc", source.id, float(measure.db(mean))))
    return results


def _quiet(
    pack: ScenePack, source: Source, feed: Feed, w: np.ndarray, lo: int, hi: int
) -> list[Result]:
    """Nothing where the recipe gives the source nothing to say."""
    rate = pack.header.sample_rate_hz
    ring = pack.header.low_samples / pack.header.low_sample_rate_hz + pack.mirror.lead_s + 0.2
    sounding = np.zeros(hi - lo, dtype=bool)
    for a, b in feed.intervals:
        first = int((a - 0.1) * rate) - lo
        sounding[max(first, 0) : max(int((b + ring) * rate) - lo, 0)] = True
    if np.all(sounding):
        return [said("silence", source.id, SKIP, "the source rings through the whole window")]
    peak = float(np.max(np.abs(w)))
    left = float(np.max(np.abs(w[~sounding])))
    return [
        judged(
            "silence",
            source.id,
            float(measure.db(left / peak)) if left > 0.0 else float("-inf"),
            note=f"{np.count_nonzero(~sounding) / rate:.1f} s of silence read",
        )
    ]


def _edges(source: Source, feed: Feed, rate: float) -> list[Result]:
    """How far from zero the dry signal starts and stops, interval by interval."""
    worst_db, where = float("-inf"), ""
    for a, b in feed.intervals:
        first, last = int(round(a * rate)), int(round(b * rate))
        dry = feed.track.read(first, last)
        rms = float(np.sqrt(np.mean(dry * dry)))
        if rms <= 0.0:
            continue
        for label, sample in (("start", dry[0]), ("end", dry[-1])):
            size = float(measure.db(abs(float(sample)) / rms)) if sample != 0.0 else float("-inf")
            if size > worst_db:
                worst_db, where = size, f"the {label} of the interval {a:.3f} to {b:.3f} s"
    return [
        judged(
            "interval_edge",
            source.id,
            worst_db,
            note=(where or "every interval starts and ends on zero") + f"; fed: {feed.label}",
            intervals=len(feed.intervals),
        )
    ]


def reference_field(path: Path) -> dict[str, Any] | None:
    """A hybrid field (``docs/formats/mirror-field.md``) as the probes read it, or ``None``."""
    if not Path(path).is_file():
        return None
    import h5py

    handle = h5py.File(path, "r")
    return {
        "name": f"{Path(path).parent.name}/{Path(path).name}",
        "scene_id": str(handle.attrs.get("scene_id", "")),
        "positions": np.asarray(handle["positions"]),
        "response": lambda index: np.asarray(handle["ir"][index, 0, :], dtype=float),
        "record": json.loads(str(handle.attrs.get("provenance_json", "{}"))).get("kind", ""),
    }
