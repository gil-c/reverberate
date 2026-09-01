"""Tests for reading one dry clip out of the sibling project's library.

The library is not ours: its shards, its catalogue, its byte offsets. So the
tests build a real zip with :mod:`zipfile`, derive the offsets the way the
catalogue does, and put it in the fake store under a **shared** key, outside
this project's prefix. Nothing here reaches the network.

The interesting failure mode is not "the download broke". It is a byte offset
that is off, which yields plausible bytes that are not the clip, and would end
up convolved into a report. Hence the CRC check, and hence a test that
deliberately corrupts the payload.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
import zlib

import numpy as np
import pytest

from reverberate.clarify_library import (
    EARS_ATTRIBUTION,
    EARS_LICENCE,
    Clip,
    catalogue,
    choose_clip,
    fetch_clip,
    is_speech,
    read_wav,
)
from reverberate.store import MemoryStore

SHARD_KEY = "library/shards/ears/p001-0000.zip"


def _wav_bytes(seconds: float, rate: int = 48_000, freq: float = 220.0) -> bytes:
    """A real RIFF/WAVE file, 16 bit PCM mono, as the library holds."""
    count = int(seconds * rate)
    time = np.arange(count) / rate
    samples = (np.sin(2 * np.pi * freq * time) * 20_000).astype("<i2")
    body = samples.tobytes()
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(body))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(body))
    )
    return header + body


def _build_shard(members: dict[str, bytes], *, compress: bool = False) -> tuple[bytes, list[Clip]]:
    """A zip and the catalogue records that address its members' payloads."""
    buffer = io.BytesIO()
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", mode) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    blob = buffer.getvalue()

    clips = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for info in archive.infolist():
            # The payload begins after the local header, whose variable part is
            # only readable from the file itself. This is what the sibling
            # project's catalogue builder does.
            head = blob[info.header_offset : info.header_offset + 30]
            name_len, extra_len = struct.unpack("<HH", head[26:30])
            offset = info.header_offset + 30 + name_len + extra_len
            clips.append(
                Clip(
                    clip_id=info.filename.removesuffix(".wav"),
                    member_name=info.filename,
                    shard_key=SHARD_KEY,
                    payload_offset=offset,
                    payload_length=info.compress_size,
                    crc32=info.CRC,
                    extension=".wav",
                )
            )
    return blob, clips


def _store_with_shard(
    members: dict[str, bytes], *, compress: bool = False
) -> tuple[MemoryStore, list[Clip]]:
    blob, clips = _build_shard(members, compress=compress)
    store = MemoryStore()
    # Written straight into the fake's objects, and not through ``put_bytes``,
    # because these keys belong to the other project and this module never
    # writes them.
    store.objects[SHARD_KEY] = blob
    store.objects["library/catalog/shards/ears/p001-0000.jsonl"] = b"\n".join(
        json.dumps(
            {
                "clip_id": clip.clip_id,
                "member_name": clip.member_name,
                "shard_key": clip.shard_key,
                "payload_offset": clip.payload_offset,
                "payload_length": clip.payload_length,
                "crc32": clip.crc32,
                "extension": clip.extension,
            }
        ).encode()
        for clip in clips
    )
    return store, clips


def test_the_catalogue_reads_every_clip_of_a_shard() -> None:
    store, clips = _store_with_shard({"a.wav": b"one", "b.wav": b"two"})
    read = catalogue(store, "ears", "p001-0000")
    assert [clip.clip_id for clip in read] == ["a", "b"]
    assert read == clips


def test_the_catalogue_ignores_blank_lines() -> None:
    store, _ = _store_with_shard({"a.wav": b"one"})
    key = "library/catalog/shards/ears/p001-0000.jsonl"
    store.objects[key] = store.objects[key] + b"\n\n"
    assert len(catalogue(store, "ears", "p001-0000")) == 1


def test_a_stored_member_is_one_ranged_read() -> None:
    """No shard download: exactly the clip's own bytes are asked for."""
    payload = _wav_bytes(0.2)
    store, clips = _store_with_shard({"a.wav": payload, "b.wav": _wav_bytes(0.3)})
    clip = next(c for c in clips if c.clip_id == "a")
    assert fetch_clip(store, clip) == payload
    assert clip.payload_length == len(payload) < len(store.objects[SHARD_KEY])


def test_a_deflated_member_is_inflated() -> None:
    payload = _wav_bytes(0.5)
    store, clips = _store_with_shard({"a.wav": payload}, compress=True)
    clip = clips[0]
    assert clip.payload_length < len(payload), "the fixture did not actually compress"
    assert fetch_clip(store, clip) == payload


def test_a_wrong_offset_is_an_error_and_not_noise() -> None:
    """The failure this exists to catch: plausible bytes that are not the clip."""
    store, clips = _store_with_shard({"a.wav": _wav_bytes(0.2), "b.wav": _wav_bytes(0.2)})
    shifted = Clip(**{**clips[0].__dict__, "payload_offset": clips[0].payload_offset + 8})
    with pytest.raises(ValueError, match="CRC mismatch|neither stored"):
        fetch_clip(store, shifted)


def test_a_corrupt_payload_is_an_error() -> None:
    store, clips = _store_with_shard({"a.wav": _wav_bytes(0.2)})
    blob = bytearray(store.objects[SHARD_KEY])
    blob[clips[0].payload_offset + 100] ^= 0xFF
    store.objects[SHARD_KEY] = bytes(blob)
    with pytest.raises(ValueError, match="CRC mismatch"):
        fetch_clip(store, clips[0])


def test_reading_a_wav_gives_float_samples_and_the_rate() -> None:
    pytest.importorskip("soundfile")
    samples, rate = read_wav(_wav_bytes(0.25, rate=48_000))
    assert rate == 48_000
    assert samples.shape == (12_000,)
    assert samples.dtype == np.float64
    assert 0.5 < float(np.max(np.abs(samples))) <= 1.0


def test_a_clip_shorter_than_the_minimum_is_refused() -> None:
    """Ten seconds is the owner's floor: a short clip cannot show a tail."""
    clips = [
        Clip("short", "p001/rainbow_01_regular.wav", SHARD_KEY, 0, 48_000 * 2 * 3, 0, ".wav"),
    ]
    with pytest.raises(ValueError, match="no clip reaches"):
        choose_clip(clips, np.random.default_rng(0), min_seconds=10.0)


def test_non_verbal_recordings_are_kept_out_of_the_draw() -> None:
    """A real draw on the real catalogue picked ``vegetative_eating.wav``.

    EARS holds emotional, non verbal and vegetative recordings beside its read
    passages. They are anechoic and useless for hearing a room on a voice.
    """
    long = 48_000 * 2 * 20
    noise = Clip("n", "p001/vegetative_eating.wav", SHARD_KEY, 0, long, 0, ".wav")
    speech = Clip("s", "p001/rainbow_01_regular.wav", SHARD_KEY, 0, long, 0, ".wav")
    assert is_speech(speech) and not is_speech(noise)
    for seed in range(16):
        assert choose_clip([noise, speech], np.random.default_rng(seed)) is speech
    with pytest.raises(ValueError, match="no clip reaches"):
        choose_clip([noise], np.random.default_rng(0))
    assert choose_clip([noise], np.random.default_rng(0), speech_only=False) is noise


def test_styled_takes_are_kept_out_of_the_draw() -> None:
    """A whispered or shouted passage changes what the listen is measuring."""
    long = 48_000 * 2 * 20
    whisper = Clip("w", "p001/rainbow_01_whisper.wav", SHARD_KEY, 0, long, 0, ".wav")
    regular = Clip("r", "p001/rainbow_01_regular.wav", SHARD_KEY, 0, long, 0, ".wav")
    freeform = Clip("f", "p001/freeform_speech_01.wav", SHARD_KEY, 0, long, 0, ".wav")
    assert not is_speech(whisper)
    assert is_speech(regular) and is_speech(freeform)
    assert is_speech(whisper, plain_style=False)
    for seed in range(8):
        assert choose_clip([whisper, regular], np.random.default_rng(seed)) is regular


def test_the_choice_is_deterministic_from_the_seed() -> None:
    clips = [
        Clip(
            f"c{index}",
            f"p001/sentences_{index}_regular.wav",
            SHARD_KEY,
            0,
            48_000 * 2 * 20,
            0,
            ".wav",
        )
        for index in range(8)
    ]
    first = choose_clip(clips, np.random.default_rng(7))
    again = choose_clip(list(reversed(clips)), np.random.default_rng(7))
    assert first == again, "the choice depended on catalogue order, not on the seed"
    assert choose_clip(clips, np.random.default_rng(8)) != first or len(clips) == 1


def test_only_long_enough_clips_are_eligible() -> None:
    long_enough = Clip("long", "p001/rainbow_02_regular.wav", SHARD_KEY, 0, 48_000 * 2 * 12, 0, "")
    clips = [
        Clip("short", "p001/rainbow_01_regular.wav", SHARD_KEY, 0, 48_000 * 2 * 2, 0, ""),
        long_enough,
    ]
    assert choose_clip(clips, np.random.default_rng(3)) is long_enough


def test_the_licence_is_carried_in_code_not_in_a_comment() -> None:
    """The roadmap left the licence open; provenance has to be able to read it."""
    assert EARS_LICENCE == "CC BY-NC 4.0"
    assert "Interspeech 2024" in EARS_ATTRIBUTION


def test_the_crc_convention_matches_zipfile() -> None:
    """Guards the masking: a signed CRC would compare unequal on some platforms."""
    payload = _wav_bytes(0.1)
    _, clips = _build_shard({"a.wav": payload})
    assert clips[0].crc32 == zlib.crc32(payload) & 0xFFFFFFFF
