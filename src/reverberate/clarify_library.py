"""Dry speech, read out of the sibling project's library one clip at a time.

The roadmap's W20 needs an **anechoic** speech clip to convolve, and left its
licence as an open question. This module settles both. The owner's other
project has already assembled a speech library on the same B2 bucket this
project now uses, and among its datasets is **EARS**, Richter and Gerkmann,
Interspeech 2024: 100 hours of speech recorded **in an anechoic chamber** at 48
kHz, released under **CC BY-NC 4.0**, the same non commercial terms as HSSD,
which this project's geometry already carries. So the clip is anechoic by
provenance rather than by hope, and the licence is recorded here and in every
response's provenance rather than settled later.

**One clip is one ranged read.** The library stores audio inside multi hundred
megabyte zip shards and publishes a JSONL catalogue giving each clip's byte
offset, length and CRC-32 inside its shard. Fetching one 35 MB clip out of a
592 MB shard is therefore a single HTTP range request, not a shard download.
The CRC-32 is checked, so a wrong offset is an error and not a burst of noise
convolved into a report.

**This module only reads.** The library belongs to the other project; nothing
here writes to its prefix.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reverberate.store import ObjectStore

__all__ = [
    "EARS_ATTRIBUTION",
    "EARS_LICENCE",
    "Clip",
    "catalogue",
    "choose_clip",
    "fetch_clip",
    "is_speech",
    "read_wav",
]

#: Settled from the dataset's own publication, and recorded in provenance.
EARS_LICENCE = "CC BY-NC 4.0"
EARS_ATTRIBUTION = (
    "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement "
    "and Dereverberation. Richter, Wu, Zhao, Gerkmann et al., Interspeech 2024."
)

#: Where the sibling project keeps the two halves of its library. Absolute keys
#: in the bucket, outside this project's prefix, hence ``shared=True`` at every
#: call site.
SHARD_CATALOGUE = "library/catalog/shards/{dataset}/{shard}.jsonl"


@dataclass(frozen=True)
class Clip:
    """One audio file, addressed by where its bytes sit inside a shard."""

    clip_id: str
    member_name: str
    shard_key: str
    payload_offset: int
    payload_length: int
    crc32: int
    extension: str

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Clip:
        def whole(field: str) -> int:
            value = record[field]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"catalogue field {field!r} is {value!r}, expected an integer")
            return value

        return cls(
            clip_id=str(record["clip_id"]),
            member_name=str(record["member_name"]),
            shard_key=str(record["shard_key"]),
            payload_offset=whole("payload_offset"),
            payload_length=whole("payload_length"),
            crc32=whole("crc32"),
            extension=str(record.get("extension", "")),
        )


def catalogue(store: ObjectStore, dataset: str, shard: str) -> list[Clip]:
    """Every clip in one shard of ``dataset``, in the catalogue's own order."""
    key = SHARD_CATALOGUE.format(dataset=dataset, shard=shard)
    payload = store.get_bytes(key, shared=True).decode()
    return [Clip.from_record(json.loads(line)) for line in payload.splitlines() if line.strip()]


def fetch_clip(store: ObjectStore, clip: Clip) -> bytes:
    """The clip's bytes, by one ranged read, CRC checked.

    Members stored without compression come back as they are; deflated members
    are inflated with a raw window, which is what a zip member is. The CRC-32
    in the catalogue is the zip's own, over the *uncompressed* bytes, so it
    verifies the decompression too.
    """
    raw = store.get_range(clip.shard_key, clip.payload_offset, clip.payload_length, shared=True)
    if _looks_stored(raw):
        payload = raw
    else:
        try:
            payload = zlib.decompress(raw, -zlib.MAX_WBITS)
        except zlib.error as error:
            # A wrong offset lands in the middle of a member and looks like
            # neither a RIFF header nor a deflate stream. Reported as the same
            # addressing failure as a CRC mismatch, because that is what it is.
            raise ValueError(
                f"{clip.member_name} at offset {clip.payload_offset} is neither stored "
                f"nor deflatable: {error}"
            ) from error
    actual = zlib.crc32(payload) & 0xFFFFFFFF
    if actual != (clip.crc32 & 0xFFFFFFFF):
        raise ValueError(
            f"CRC mismatch on {clip.member_name}: got {actual}, catalogue says {clip.crc32}"
        )
    return payload


def read_wav(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode a RIFF/WAVE payload to float64 in [-1, 1] and its sample rate.

    Mono is returned as a 1-D array; anything multi channel is averaged down,
    because a convolution against a single omnidirectional receiver has no use
    for a stereo image that the room is about to replace.
    """
    import io

    import soundfile

    samples, rate = soundfile.read(io.BytesIO(payload), dtype="float64", always_2d=True)
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    return np.ascontiguousarray(mono), int(rate)


#: EARS is not only speech: alongside its read passages it holds emotional
#: bursts, non verbal sounds and vegetative sounds such as eating and coughing.
#: A blind draw picked ``vegetative_eating.wav``, which is a fine anechoic
#: recording and a poor thing to convolve when the question is whether a room
#: is audible on a voice. These are the dataset's read and free speech
#: categories, matched on the member's own file name.
SPEECH_CATEGORIES = ("rainbow_", "freeform_speech_", "sentences_")

#: EARS records each passage in seven speaking styles, named in the file:
#: regular, fast, slow, loud, whisper, highpitch, lowpitch. A draw that ignores
#: them returns ``rainbow_04_highpitch`` as readily as a normal voice, and a
#: whispered or shouted passage changes the very thing the listen is meant to
#: show, which is what the room does to an ordinary voice. Excluded by name,
#: leaving ``_regular`` and the unstyled free form recordings.
STYLED_SPEECH = ("_fast", "_slow", "_loud", "_whisper", "_highpitch", "_lowpitch")


def is_speech(clip: Clip, *, plain_style: bool = True) -> bool:
    """Whether the clip is an ordinary spoken passage rather than a noise.

    ``plain_style`` additionally rejects EARS' shouted, whispered and pitch
    shifted takes.
    """
    name = clip.member_name.rsplit("/", 1)[-1]
    if not name.startswith(SPEECH_CATEGORIES):
        return False
    return not plain_style or not name.removesuffix(".wav").endswith(STYLED_SPEECH)


def choose_clip(
    clips: list[Clip],
    rng: np.random.Generator,
    *,
    min_seconds: float = 10.0,
    sample_rate_hz: int = 48_000,
    bytes_per_sample: int = 2,
    speech_only: bool = True,
) -> Clip:
    """Pick one clip at least ``min_seconds`` long, deterministically from ``rng``.

    The length is estimated from the payload size rather than by decoding every
    candidate, which would mean fetching the whole shard to choose one clip. The
    estimate is deliberately conservative: it assumes the WAV is uncompressed
    PCM at the stated rate and depth, which is what this library holds, and a
    44 byte header is neglected against a ten second minimum.

    ``speech_only`` keeps the draw inside EARS' ordinary spoken passages. It
    defaults to true because the alternatives are a plausible looking run built
    on a recording of someone eating, or on a whisper.
    """
    floor = int(min_seconds * sample_rate_hz * bytes_per_sample)
    candidates = [clip for clip in clips if not speech_only or is_speech(clip)]
    eligible = [clip for clip in candidates if clip.payload_length >= floor]
    if not eligible:
        raise ValueError(
            f"no clip reaches {min_seconds} s among {len(candidates)} candidates "
            f"of {len(clips)} in the shard"
        )
    eligible.sort(key=lambda clip: clip.clip_id)
    return eligible[int(rng.integers(len(eligible)))]


def _looks_stored(raw: bytes) -> bool:
    return raw[:4] == b"RIFF"
