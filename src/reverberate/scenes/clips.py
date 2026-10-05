"""A library of dry clips: its manifest, how it is built, fetched and checked.

``docs/formats/clip-library.md`` is the contract. A recipe names a clip by
library, name and the SHA-256 of its file; the audit and the signal engine
read ``<data root>/clips/<library>/<name>.wav``. This module makes those
files and the manifest the generator draws from.

**The library is derived, and the manifest says from what.** Every clip is
cut out of the sibling project's audio library on the bucket
(:mod:`reverberate.clarify_library`): the manifest holds, for each clip, the
members it is made of (shard, byte range, CRC-32), the processing (channel,
cut, fades, loop, gain) and the digest of the result. :func:`fetch` runs
that again and refuses a file whose digest is another, so the files are
never stored anywhere but on the machine that listens.

**Three steps, three commands** (``python -m reverberate.scenes clips``):

- ``curate``: a *selection* (which members, which kind, which level) to a
  manifest. It measures: the active speech level, the utterances, the gain.
  Run once, by whoever chooses the material.
- ``fetch``: a manifest to the files, digests checked.
- ``check``: the files measured again, one row a clip, against fixed limits.

**Levels** (the convention, stated once in the format document). A voice is
stored at an active speech level of :data:`VOICE_ACTIVE_DBFS`, measured after
ITU-T P.56, and that stands for a normal vocal effort, 60 dB SPL at 1 m. So
a clip's full scale stands for :data:`FULL_SCALE_SPL_1M_DB` at 1 m, and a
noise is stored at the long term level its source has at 1 m on that scale.

Nothing here opens the network by itself: the store is handed in.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import wave
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import lfilter, welch

if TYPE_CHECKING:
    from reverberate.store import ObjectStore

__all__ = [
    "FULL_SCALE_SPL_1M_DB",
    "LIMITS",
    "RATE_HZ",
    "SCHEMA",
    "SCHEMA_VERSION",
    "VOICE_ACTIVE_DBFS",
    "ClipReport",
    "PartLoader",
    "active_speech_level",
    "build_clip",
    "check",
    "clip_path",
    "curate",
    "dumps",
    "fetch",
    "find_utterances",
    "format_table",
    "heard_level_db",
    "measure",
    "store_loader",
    "wav_bytes",
]

SCHEMA = "reverberate.clip-library"
SCHEMA_VERSION = 1
#: Every clip is mono, 16 bit, at the engine's output rate.
RATE_HZ = 48000
#: The active speech level every voice is stored at, dB re full scale.
VOICE_ACTIVE_DBFS = -26.0
#: What a full scale sine's level stands for, dB SPL at 1 m in free field:
#: a voice at :data:`VOICE_ACTIVE_DBFS` is then at 60 dB, a normal effort.
FULL_SCALE_SPL_1M_DB = 86.0
#: How many times a member is read off the bucket before the fetch gives up.
READ_ATTEMPTS = 3
#: No clip's peak is stored above this, dB re full scale.
HEADROOM_DBFS = -1.0
#: Kept before the first utterance of a voice clip and after its last, seconds.
VOICE_MARGIN_S = 0.2
#: A sentence is cut to its speech: what is this far under the talker's level
#: at either end goes, but for these margins, seconds.
SENTENCE_FLOOR_DB = 30.0
SENTENCE_MARGIN_S = (0.05, 0.08)

#: A frame of the level measurements, seconds.
FRAME_S = 0.01
#: Below the active level by this much, a frame is not speech: a breath is
#: 30 to 45 dB under a talker's level, the closure of a stop about 20.
SPEECH_FLOOR_DB = 18.0
#: A pause at least this long ends an utterance, seconds. A monologue has
#: few silences: it is cut where the talker breathes.
MIN_PAUSE_S = 0.18
#: An utterance is cut within this much of its speech, at most, seconds.
PAD_S = 0.10
#: An utterance longer than this is cut at its longest pause, seconds.
MAX_UTTERANCE_S = 12.0
#: The shortest pause an utterance may be cut at, and the shortest utterance kept.
MIN_BREATH_S = 0.10
MIN_UTTERANCE_S = 0.30

#: Where a render starts, as a filter: the low cut of every solve's fit and of the band
#: above the crossover, a Butterworth high pass of this order at this frequency
#: (``reverberate.spatial.lowband.LOWCUT_HZ``). A pack is valid from 45 Hz
#: (``docs/formats/scene-pack.md``), and a noise is levelled on what lies above.
HEARD_FROM_HZ, HEARD_ORDER = 40.0, 8

#: What ``check`` holds every clip to. A limit never moves to pass a clip.
LIMITS: dict[str, float] = {
    # Samples at either end of the scale.
    "clipped": 0,
    "peak_dbfs": -0.5,
    # The mean against the clip's level.
    "dc_db": -40.0,
    # A voice: how far from the stated active level, dB.
    "voice_level_error_db": 0.1,
    # A voice: the level 100 to 200 ms after speech stops, against the speech;
    # the lower quartile of the pauses, for a room rings in every one of them
    # and a talker breathes in some. A pause is under the speech by
    # ``SPEECH_FLOOR_DB`` by its definition; the limit is 10 dB further.
    "voice_decay_db": -(SPEECH_FLOOR_DB + 10.0),
    # A voice: its quietest 50 ms, dB re full scale: 40 dB under the speech.
    "voice_floor_dbfs": -66.0,
    # A voice: the loudest sample where an utterance is cut, dB re full scale.
    "voice_edge_dbfs": -50.0,
    # A noise: how far from the stated long term level, dB.
    "noise_level_error_db": 0.1,
}

#: A part of a clip to its samples, ``[frame, channel]``, and their rate.
PartLoader = Callable[[Mapping[str, Any]], tuple[np.ndarray, float]]


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------


def _db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0.0 else -math.inf


def active_speech_level(samples: np.ndarray, rate: float = RATE_HZ) -> tuple[float, float]:
    """The active speech level, dB re full scale, and the activity factor.

    After ITU-T P.56, method B: the envelope is the rectified signal through
    two first order low passes of 30 ms; for a threshold, the signal is active
    while the envelope is above it and for 200 ms after; the active level is
    the power over the active time, at the threshold that sits 15.9 dB under
    it. The thresholds are 3 dB apart here, not 6, and interpolated.
    """
    x = np.asarray(samples, dtype=float)
    power = float(np.sum(x * x))
    if power <= 0.0 or x.size == 0:
        return -math.inf, 0.0
    g = math.exp(-1.0 / (rate * 0.03))
    envelope = lfilter([1.0 - g], [1.0, -g], lfilter([1.0 - g], [1.0, -g], np.abs(x)))
    # The envelope has no content above a few tens of hertz: a sixteenth of
    # the samples count the active time as well as all of them.
    step = 16
    envelope = np.asarray(envelope)[::step]
    hang = int(round(0.2 * rate / step))
    index = np.arange(envelope.size)
    margin = 15.9
    previous: tuple[float, float] | None = None
    for threshold_db in np.arange(-96.0, 0.5, 3.0):
        above = envelope >= 10.0 ** (threshold_db / 20.0)
        last = np.maximum.accumulate(np.where(above, index, -hang - 1))
        active = int(np.count_nonzero(index - last <= hang)) * step
        if active == 0:
            break
        level = 10.0 * math.log10(power / min(active, x.size))
        delta = level - float(threshold_db)
        if delta < margin and previous is not None:
            was_level, was_delta = previous
            share = (was_delta - margin) / (was_delta - delta)
            level = was_level + share * (level - was_level)
            return level, min(1.0, 10.0 ** ((10.0 * math.log10(power / x.size) - level) / 10.0))
        previous = (level, delta)
    level = previous[0] if previous is not None else 10.0 * math.log10(power / x.size)
    return level, min(1.0, 10.0 ** ((10.0 * math.log10(power / x.size) - level) / 10.0))


def _frame_levels(samples: np.ndarray, rate: float) -> np.ndarray:
    """The level of every whole frame of :data:`FRAME_S`, dB re full scale."""
    hop = int(round(FRAME_S * rate))
    frames = samples.size // hop
    if frames == 0:
        return np.zeros(0)
    power = np.mean(np.asarray(samples[: frames * hop], float).reshape(frames, hop) ** 2, axis=1)
    return np.asarray(10.0 * np.log10(np.maximum(power, 1e-20)))


def _quietest(levels: np.ndarray, frames: int = 5) -> float:
    """The level of the quietest ``frames`` frames in a row, dB."""
    if levels.size < frames:
        return -math.inf
    power = np.convolve(10.0 ** (levels / 10.0), np.full(frames, 1.0 / frames), mode="valid")
    return float(10.0 * np.log10(np.min(power)))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """The runs of true in ``mask`` as ``(first, stop)``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def find_utterances(
    samples: np.ndarray, rate: float = RATE_HZ, active_db: float | None = None
) -> list[tuple[float, float]]:
    """Where a voice clip may be cut: its utterances, ``(start_s, end_s)``.

    An utterance is speech between two pauses of :data:`MIN_PAUSE_S` or
    more. One longer than :data:`MAX_UTTERANCE_S` is cut at its longest
    pause, again and again. Each end is put on the quietest 10 ms of the
    pause, no further than :data:`PAD_S` from the speech and no further than
    the middle of the pause, so no two overlap: a talk spurt that plays
    whole utterances cuts no word.
    """
    if active_db is None:
        active_db, _ = active_speech_level(samples, rate)
    levels = _frame_levels(samples, rate)
    speech = levels > active_db - SPEECH_FLOOR_DB
    frames = [(a, b) for a, b in _runs(speech)]
    pause = int(round(MIN_PAUSE_S / FRAME_S))

    groups: list[list[tuple[int, int]]] = []
    for run in frames:
        if groups and run[0] - groups[-1][-1][1] < pause:
            groups[-1].append(run)
        else:
            groups.append([run])

    def split(group: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        if (group[-1][1] - group[0][0]) * FRAME_S <= MAX_UTTERANCE_S or len(group) < 2:
            return [group]
        gaps = [group[k + 1][0] - group[k][1] for k in range(len(group) - 1)]
        k = int(np.argmax(gaps))
        if gaps[k] * FRAME_S < MIN_BREATH_S:
            return [group]
        return [*split(group[: k + 1]), *split(group[k + 1 :])]

    pieces = [piece for group in groups for piece in split(group)]
    pieces = [p for p in pieces if (p[-1][1] - p[0][0]) * FRAME_S >= MIN_UTTERANCE_S]
    total = samples.size / rate
    pad = int(round(PAD_S / FRAME_S))

    def quietest(a: int, b: int, otherwise: float) -> float:
        """The middle of the quietest frame of ``[a, b)``."""
        if b <= a:
            return otherwise
        return (a + int(np.argmin(levels[a:b])) + 0.5) * FRAME_S

    out: list[tuple[float, float]] = []
    for number, piece in enumerate(pieces):
        first, stop = piece[0][0], piece[-1][1]
        before = pieces[number - 1][-1][1] if number else 0
        after = pieces[number + 1][0][0] if number + 1 < len(pieces) else levels.size
        start = quietest(max(first - pad, (before + first) // 2), first, 0.0)
        end = quietest(
            stop, min(stop + pad, max((stop + after) // 2, stop + 1), levels.size), total
        )
        out.append((round(start, 3), round(min(end, total), 3)))
    return out


def heard_level_db(samples: np.ndarray, rate: float = RATE_HZ) -> float:
    """The long term level of what a render holds of ``samples``, dB re full scale.

    The RMS of the clip through the chain's own low cut
    (:data:`HEARD_FROM_HZ`, :data:`HEARD_ORDER`). A recording made in a room
    may hold most of its energy under 20 Hz, where no one hears and nothing
    is rendered: the first library's washing machine holds 98 per cent of it
    between 10 and 20 Hz, and levelled on its whole band it was rendered
    25 dB under the level its entry states
    (``docs/open-questions/chain-audit.md``, D12).
    """
    from scipy.signal import butter, sosfilt

    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return _db(0.0)
    sos = butter(HEARD_ORDER, HEARD_FROM_HZ, btype="high", fs=rate, output="sos")
    kept = np.asarray(sosfilt(sos, x))
    return _db(float(np.sqrt(np.mean(kept * kept))))


def measure(
    pcm: np.ndarray,
    *,
    kind: str,
    utterances: Sequence[Sequence[float]] = (),
    rate: float = RATE_HZ,
) -> dict[str, float | int | None]:
    """What ``check`` reports of a clip's 16 bit samples."""
    pcm = np.asarray(pcm)
    x = pcm.astype(float) / 32768.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    levels = _frame_levels(x, rate)
    freqs, density = welch(x, fs=rate, nperseg=4096)
    whole = float(np.sum(density))
    high = float(np.sum(density[freqs >= 8000.0]))
    out: dict[str, float | int | None] = {
        "duration_s": round(x.size / rate, 3),
        "peak_dbfs": round(_db(float(np.max(np.abs(x)))), 2),
        "clipped": int(np.count_nonzero((pcm >= 32767) | (pcm <= -32768))),
        "rms_dbfs": round(_db(rms), 2),
        "dc_db": round(_db(abs(float(np.mean(x)))) - _db(rms), 1),
        "floor_dbfs": round(_quietest(levels), 1),
        "above_8k_db": round(10.0 * math.log10(max(high, 1e-30) / whole), 1),
        "active_dbfs": None,
        "activity": None,
        "decay_db": None,
        "edge_dbfs": None,
    }
    if kind != "voice":
        return out
    active, activity = active_speech_level(x, rate)
    out["active_dbfs"] = round(active, 2)
    out["activity"] = round(activity, 3)
    # After speech stops: the level between 100 and 200 ms later, where a
    # room would still ring and an anechoic chamber does not.
    speech = levels > active - SPEECH_FLOOR_DB
    tails = []
    for _, stop in _runs(speech):
        if stop + 20 <= speech.size and not speech[stop : stop + 20].any():
            tails.append(10.0 * math.log10(np.mean(10.0 ** (levels[stop + 10 : stop + 20] / 10.0))))
    if tails:
        out["decay_db"] = round(float(np.percentile(tails, 25)) - active, 1)
    edges = [int(round(t * rate)) for pair in utterances for t in pair]
    if edges:
        worst = max(float(np.max(np.abs(x[max(e - 48, 0) : e + 48]), initial=0.0)) for e in edges)
        out["edge_dbfs"] = round(max(_db(worst), -120.0), 1)
    return out


# --------------------------------------------------------------------------
# a clip from its parts
# --------------------------------------------------------------------------


def wav_bytes(pcm: np.ndarray, rate: int = RATE_HZ) -> bytes:
    """Mono 16 bit samples as a RIFF/WAVE file: a 44 byte header and the samples."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    return buffer.getvalue()


def _fade(count: int) -> np.ndarray:
    return np.sin(0.5 * np.pi * (np.arange(count) + 0.5) / count) ** 2


def _part(part: Mapping[str, Any], load: PartLoader) -> np.ndarray:
    samples, rate = load(part)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    channel = part.get("channel")
    mono = samples.mean(axis=1) if channel is None else samples[:, int(channel)]
    if rate != RATE_HZ:
        from reverberate.audio import resample_to

        mono = resample_to(mono, rate, RATE_HZ)
    first = int(round(float(part.get("start_s") or 0.0) * RATE_HZ))
    end = part.get("end_s")
    stop = mono.size if end is None else int(round(float(end) * RATE_HZ))
    if not 0 <= first < stop <= mono.size:
        raise ValueError(f"{part.get('member_name')!r} has no samples {first} to {stop}")
    return np.array(mono[first:stop])


def build_clip(
    origin: Mapping[str, Any], load: PartLoader, *, gain_db: float | None = None
) -> np.ndarray:
    """A clip's samples from what the manifest says it is made of, as floats.

    In this order: each part decoded, one channel taken (or their mean),
    brought to 48 kHz, cut, faded at its ends; the parts joined with a
    silence between; the mean removed; the end folded onto the start so the
    clip loops; the whole faded at its ends; cut to a whole number of
    milliseconds; the gain.
    """
    process = origin.get("process", {})
    part_fade = int(round(float(process.get("part_fade_s", 0.0)) * RATE_HZ))
    gap = np.zeros(int(round(float(process.get("gap_s", 0.0)) * RATE_HZ)))
    pieces: list[np.ndarray] = []
    for part in origin["parts"]:
        piece = _part(part, load)
        if part_fade:
            ramp = _fade(part_fade)
            piece[:part_fade] *= ramp
            piece[-part_fade:] *= ramp[::-1]
        if pieces and gap.size:
            pieces.append(gap)
        pieces.append(piece)
    x = np.concatenate(pieces)
    if process.get("remove_dc", False):
        x = x - float(np.mean(x))
    fold = int(round(float(process.get("loop_crossfade_s", 0.0)) * RATE_HZ))
    if fold:
        if 2 * fold >= x.size:
            raise ValueError("the loop's crossfade is longer than half the clip")
        # Two stretches of a noise are not correlated: equal power, not equal amplitude.
        ramp = np.sqrt(_fade(fold))
        head = x[:fold] * ramp + x[-fold:] * ramp[::-1]
        x = np.concatenate([head, x[fold:-fold]])
    fade = int(round(float(process.get("fade_s", 0.0)) * RATE_HZ))
    if fade:
        ramp = _fade(fade)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    x = x[: x.size - x.size % (RATE_HZ // 1000)]
    gain = float(process.get("gain_db", 0.0)) if gain_db is None else gain_db
    return np.asarray(x * 10.0 ** (gain / 20.0))


def _pcm(samples: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.asarray(samples, dtype=float) * 32768.0)
    if scaled.size and (scaled.max() > 32767 or scaled.min() < -32768):
        raise ValueError("the clip does not fit sixteen bits at its gain")
    return scaled.astype("<i2")


def store_loader(store: ObjectStore) -> PartLoader:
    """The loader that reads a part off the bucket: one ranged read, CRC checked."""
    from reverberate import clarify_library

    def load(part: Mapping[str, Any]) -> tuple[np.ndarray, float]:
        import soundfile

        # The catalogue calls the shard's object ``shard_key``; a manifest
        # says ``shard_path``, so that no reader takes its lines for secrets.
        record = {**part, "shard_key": part["shard_path"]}
        member = clarify_library.Clip.from_record(record)
        for attempt in range(READ_ATTEMPTS):
            try:
                payload = clarify_library.fetch_clip(store, member)
                break
            except Exception:
                # A connection that breaks in the middle of a read: a
                # library is a thousand reads, and one of them does.
                if attempt + 1 == READ_ATTEMPTS:
                    raise
        samples, rate = soundfile.read(io.BytesIO(payload), dtype="float64", always_2d=True)
        return samples, float(rate)

    return load


def clip_path(root: Path, library: str, name: str) -> Path:
    """Where the audit and the engine look for a clip: ``<root>/<library>/<name>.wav``."""
    return Path(root) / library / f"{name}.wav"


# --------------------------------------------------------------------------
# curate: a selection to a manifest
# --------------------------------------------------------------------------


def _resolve(
    part: Mapping[str, Any], store: ObjectStore, cache: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """A selection's part, ``{dataset, shard, member}``, to the catalogue's row."""
    from reverberate import clarify_library

    where = clarify_library.SHARD_CATALOGUE.format(dataset=part["dataset"], shard=part["shard"])
    if where not in cache:
        rows = store.get_bytes(where, shared=True).decode().splitlines()
        cache[where] = {r["member_name"]: r for r in map(json.loads, filter(str.strip, rows))}
    try:
        row = cache[where][part["member"]]
    except KeyError:
        raise ValueError(f"{where} holds no member {part['member']!r}") from None
    keep = ("clip_id", "member_name", "payload_offset", "payload_length", "crc32")
    out: dict[str, Any] = {"dataset": part["dataset"], **{k: row[k] for k in keep}}
    out["shard_path"] = row["shard_key"]
    out["extension"] = row.get("extension", "")
    out["channel"] = part.get("channel")
    out["start_s"] = part.get("start_s")
    out["end_s"] = part.get("end_s")
    return out


def curate(
    selection: Mapping[str, Any],
    store: ObjectStore,
    root: Path | None = None,
    *,
    load: PartLoader | None = None,
    jobs: int = 4,
) -> dict[str, Any]:
    """The manifest of a selection; the files too, under ``root``, when one is given.

    A selection names, for every clip, the members it is made of and its
    level: ``spl_1m_db``, what the source is at 1 m. A voice (and anything
    whose ``measure`` is ``"active"``) is levelled on its active speech
    level, a noise on the long term level of what a render holds of it
    (``"heard"``, :func:`heard_level_db`); ``"rms"`` is the long term level
    of the whole band, which the first library's noises were levelled on
    and its selection says. A voice is cut to its speech, and its
    utterances are found and written.
    """
    cache: dict[str, dict[str, Any]] = {}
    loader = load if load is not None else store_loader(store)
    datasets = dict(selection.get("datasets", {}))
    library = str(selection["library"])

    def one(item: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(item["kind"])
        parts = [_resolve(part, store, cache) for part in item["parts"]]
        process = {k: v for k, v in dict(item.get("process", {})).items() if k != "gain_db"}
        origin: dict[str, Any] = {"parts": parts, "process": process}
        held: dict[tuple[str, int], tuple[np.ndarray, float]] = {}

        def once(part: Mapping[str, Any]) -> tuple[np.ndarray, float]:
            key = (str(part["shard_path"]), int(part["payload_offset"]))
            if key not in held:
                held[key] = loader(part)
            return held[key]

        raw = build_clip(origin, once, gain_db=0.0)
        utterances: list[tuple[float, float]] = []
        members: np.ndarray | None = None
        if kind == "voice" and len(parts) > 1:
            # The members as they are, end to end: what the floor and the
            # decay are measured on, for the clip's own silences are digital.
            members = np.concatenate([_part(part, once) for part in parts])
            # A clip of sentences: each member is one, cut to its speech, and
            # an utterance is a sentence with half the silence on either side.
            active, _ = active_speech_level(raw)
            gap = float(process.get("gap_s", 0.0))
            if gap < 0.2:
                raise ValueError(f"{item['name']}: sentences are joined by a silence, gap_s")
            edges, at = [], 0
            for part in parts:
                first, last = _speech_span(_part(part, once), active)
                part["start_s"], part["end_s"] = first, last
                count = int(round(last * RATE_HZ)) - int(round(first * RATE_HZ))
                edges.append((at / RATE_HZ, (at + count) / RATE_HZ))
                at += count + int(round(gap * RATE_HZ))
            raw = build_clip(origin, once, gain_db=0.0)
            total = raw.size / RATE_HZ
            utterances = [
                (
                    round(max(a - 0.5 * gap, 0.0), 3) if k else 0.0,
                    round(b + 0.5 * gap, 3) if k + 1 < len(edges) else round(total, 3),
                )
                for k, (a, b) in enumerate(edges)
            ]
        elif kind == "voice":
            found = find_utterances(raw)
            if not found:
                raise ValueError(f"{item['name']}: no speech was found")
            # Seconds of the member, which may itself be a cut of it.
            base = float(parts[0]["start_s"] or 0.0)
            first = math.floor(max(found[0][0] - VOICE_MARGIN_S, 0.0) * 1000) / 1000
            last = math.ceil(min(found[-1][1] + VOICE_MARGIN_S, raw.size / RATE_HZ) * 1000) / 1000
            parts[0]["start_s"], parts[0]["end_s"] = round(base + first, 3), round(base + last, 3)
            raw = build_clip(origin, once, gain_db=0.0)
            utterances = [(round(a - first, 3), round(b - first, 3)) for a, b in found]
        spl = float(item.get("spl_1m_db", 60.0))
        how = str(item.get("measure", "active" if kind == "voice" else "heard"))
        if how not in ("active", "heard", "rms"):
            raise ValueError(f"{item['name']}: a clip is levelled active, heard or rms")
        if how == "active":
            measured, _ = active_speech_level(raw)
        elif how == "heard":
            measured = heard_level_db(raw)
        else:
            measured = _db(float(np.sqrt(np.mean(raw * raw))))
        # A clip whose peaks would not fit at the level asked for is stored
        # lower, by whole decibels, and its entry states the level it has.
        peak = _db(float(np.max(np.abs(raw))))
        while peak + spl - FULL_SCALE_SPL_1M_DB - measured > HEADROOM_DBFS:
            spl -= 1.0
        process["gain_db"] = round(spl - FULL_SCALE_SPL_1M_DB - measured, 2)
        pcm = _pcm(build_clip(origin, once))
        payload = wav_bytes(pcm)
        if root is not None:
            _write(clip_path(root, library, str(item["name"])), payload)
        dataset = datasets.get(parts[0]["dataset"], {})
        levels = _levels(pcm, kind, utterances)
        if how == "heard":
            levels["heard_dbfs"] = round(heard_level_db(pcm.astype(float) / 32768.0), 2)
        if members is not None:
            gain = 10.0 ** (process["gain_db"] / 20.0)
            source = measure(_pcm(np.clip(members * gain, -0.999, 0.999)), kind="voice")
            levels["source_floor_dbfs"] = source["floor_dbfs"]
            levels["source_decay_db"] = source["decay_db"]
        entry: dict[str, Any] = {
            "name": str(item["name"]),
            "kind": "voice" if kind == "voice" else "noise",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "duration_s": round(pcm.size / RATE_HZ, 3),
            "loop": bool(process.get("loop_crossfade_s", 0.0)),
            "level": {"measure": how, "spl_1m_db": spl, **levels},
            "licence": str(item.get("licence", dataset.get("licence", ""))),
            "credit": str(item.get("credit", dataset.get("attribution", ""))),
            "what": str(item.get("what", "")),
            "origin": origin,
        }
        if kind == "voice":
            entry["speaker"] = str(item["speaker"])
            entry["utterances"] = [list(pair) for pair in utterances]
        else:
            entry["subtype"] = kind
        return entry

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        clips = list(pool.map(one, selection["clips"]))
    names = [clip["name"] for clip in clips]
    if len(set(names)) != len(names):
        raise ValueError("two clips of the selection have one name")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "library": library,
        "sample_rate_hz": RATE_HZ,
        "level": {
            "voice_active_dbfs": VOICE_ACTIVE_DBFS,
            "full_scale_spl_1m_db": FULL_SCALE_SPL_1M_DB,
        },
        "datasets": datasets,
        "clips": sorted(clips, key=lambda clip: clip["name"]),
    }


def _speech_span(samples: np.ndarray, active_db: float) -> tuple[float, float]:
    """Where the speech of one sentence starts and stops, seconds, with its margins."""
    levels = _frame_levels(samples, RATE_HZ)
    speech = np.flatnonzero(levels > active_db - SENTENCE_FLOOR_DB)
    if speech.size == 0:
        raise ValueError("a sentence holds no speech")
    total = math.floor(samples.size / RATE_HZ * 1000) / 1000
    first = max(round(speech[0] * FRAME_S - SENTENCE_MARGIN_S[0], 3), 0.0)
    last = min(round((speech[-1] + 1) * FRAME_S + SENTENCE_MARGIN_S[1], 3), total)
    return first, last


def dumps(tree: Mapping[str, Any]) -> str:
    """A manifest or a selection as it is kept in the repository: one clip a line."""
    head = json.dumps({key: value for key, value in tree.items() if key != "clips"}, indent=1)
    rows = ",\n".join(
        "  " + json.dumps(clip, separators=(",", ":"), ensure_ascii=False) for clip in tree["clips"]
    )
    return f'{head[:-2]},\n "clips": [\n{rows}\n ]\n}}\n'


def _levels(pcm: np.ndarray, kind: str, utterances: Sequence[Sequence[float]]) -> dict[str, Any]:
    found = measure(pcm, kind=kind, utterances=utterances)
    keys = ("rms_dbfs", "peak_dbfs") + (("active_dbfs", "activity") if kind == "voice" else ())
    return {key: found[key] for key in keys}


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload)
    partial.replace(path)


# --------------------------------------------------------------------------
# fetch and check
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(
    manifest: Mapping[str, Any],
    root: Path,
    store: ObjectStore | None,
    *,
    load: PartLoader | None = None,
    jobs: int = 4,
) -> dict[str, str]:
    """Every clip of the manifest under ``root``; what was done to each, by name.

    A file already there with the manifest's digest is kept. Any other is
    built again from its parts; one whose digest is then not the manifest's is
    not written, and the fetch fails once every clip has been tried.
    """
    if load is None:
        if store is None:
            raise ValueError("neither a store nor a loader was given")
        load = store_loader(store)
    library = str(manifest["library"])

    def one(clip: Mapping[str, Any]) -> tuple[str, str]:
        path = clip_path(root, library, str(clip["name"]))
        if path.is_file() and _sha256(path) == clip["sha256"]:
            return str(clip["name"]), "kept"
        payload = wav_bytes(_pcm(build_clip(clip["origin"], load)))
        if hashlib.sha256(payload).hexdigest() != clip["sha256"]:
            return str(clip["name"]), "REFUSED: built to another digest"
        _write(path, payload)
        return str(clip["name"]), "fetched"

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        done = dict(pool.map(one, manifest["clips"]))
    refused = sorted(name for name, state in done.items() if state.startswith("REFUSED"))
    if refused:
        raise ValueError(
            f"{len(refused)} clip(s) did not build to the manifest's digest: {', '.join(refused)}"
        )
    return done


@dataclass(frozen=True)
class ClipReport:
    """One clip as ``check`` found it."""

    name: str
    kind: str
    measured: Mapping[str, float | int | None]
    failures: tuple[str, ...]


def check(manifest: Mapping[str, Any], root: Path) -> list[ClipReport]:
    """Every clip on the disk measured against :data:`LIMITS` and its own manifest entry."""
    import soundfile

    library = str(manifest["library"])
    reports = []
    for clip in manifest["clips"]:
        name, voice = str(clip["name"]), clip["kind"] == "voice"
        kind = "voice" if voice else str(clip["subtype"])
        path = clip_path(root, library, name)
        if not path.is_file():
            reports.append(ClipReport(name, kind, {}, ("missing",)))
            continue
        failures = []
        if _sha256(path) != clip["sha256"]:
            failures.append("digest")
        info = soundfile.info(str(path))
        if (info.samplerate, info.channels, info.subtype) != (RATE_HZ, 1, "PCM_16"):
            failures.append("format")
        pcm, _ = soundfile.read(str(path), dtype="int16")
        found = measure(
            pcm, kind="voice" if voice else "noise", utterances=clip.get("utterances", ())
        )
        if found["duration_s"] != clip["duration_s"]:
            failures.append("duration")
        if int(found["clipped"] or 0) > LIMITS["clipped"]:
            failures.append("clipped")
        if float(found["peak_dbfs"] or 0.0) > LIMITS["peak_dbfs"]:
            failures.append("peak")
        if float(found["dc_db"] or 0.0) > LIMITS["dc_db"]:
            failures.append("dc")
        stated = float(clip["level"]["spl_1m_db"]) - FULL_SCALE_SPL_1M_DB
        if clip["level"]["measure"] == "active":
            level, _ = active_speech_level(pcm.astype(float) / 32768.0)
            limit = LIMITS["voice_level_error_db"]
        elif clip["level"]["measure"] == "heard":
            level = heard_level_db(pcm.astype(float) / 32768.0)
            limit = LIMITS["noise_level_error_db"]
        else:
            level, limit = float(found["rms_dbfs"] or 0.0), LIMITS["noise_level_error_db"]
        if abs(level - stated) > limit:
            failures.append("level")
        if voice:
            # A clip of sentences is silent between them by construction: its
            # floor and decay are those its members had, which the manifest
            # recorded when it was made, and they are what is held to the limits.
            found["floor_dbfs"] = clip["level"].get("source_floor_dbfs", found["floor_dbfs"])
            found["decay_db"] = clip["level"].get("source_decay_db", found["decay_db"])
            decay, edge, floor = found["decay_db"], found["edge_dbfs"], found["floor_dbfs"]
            if decay is None or float(decay) > LIMITS["voice_decay_db"]:
                failures.append("decay")
            if floor is None or float(floor) > LIMITS["voice_floor_dbfs"]:
                failures.append("floor")
            if edge is None or float(edge) > LIMITS["voice_edge_dbfs"]:
                failures.append("edge")
        reports.append(ClipReport(name, kind, found, tuple(failures)))
    return reports


def format_table(reports: Sequence[ClipReport]) -> str:
    """The reports as the table ``check`` prints."""
    columns = (
        ("duration_s", "s", 8),
        ("peak_dbfs", "peak", 7),
        ("clipped", "clip", 5),
        ("rms_dbfs", "rms", 7),
        ("active_dbfs", "active", 7),
        ("activity", "act", 6),
        ("dc_db", "dc", 7),
        ("floor_dbfs", "floor", 7),
        ("decay_db", "decay", 7),
        ("edge_dbfs", "edge", 7),
        ("above_8k_db", ">8k", 6),
    )
    width = max((len(report.name) for report in reports), default=4)
    lines = [f"{'clip':<{width}}  {'kind':<11}" + "".join(f"{t:>{w}}" for _, t, w in columns)]
    for report in reports:
        cells = ""
        for key, _, w in columns:
            value = report.measured.get(key)
            cells += f"{'-' if value is None else value:>{w}}"
        state = "" if not report.failures else "  FAILS " + ", ".join(report.failures)
        lines.append(f"{report.name:<{width}}  {report.kind:<11}{cells}{state}")
    return "\n".join(lines)
