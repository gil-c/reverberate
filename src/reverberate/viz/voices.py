"""Anechoic voices for the app, fetched once from the shared speech library.

The app convolves a dry voice with the field, so the voice has to be
anechoic: EARS (Richter and Gerkmann, Interspeech 2024), recorded in an
anechoic chamber at 48 kHz, reached through :mod:`reverberate.clarify_library`
on the shared bucket. Each speaker's freeform passage runs about six minutes;
the app wants a minute or more per voice, so a fixed length is cut from the
start of one passage per speaker and written as 16-bit WAV under
``<data root>/voices/`` with a ``voices.json`` beside it.

**EARS is CC BY-NC 4.0** and every file written here carries that in the
index, with the attribution, because a voice that travels without its licence
does not travel.

The speakers' sex is not in the catalogue and is not guessed here: twelve
distinct speakers are taken in order, and the index says which.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from reverberate import audio
from reverberate.clarify_library import (
    EARS_ATTRIBUTION,
    EARS_LICENCE,
    Clip,
    catalogue,
    fetch_clip,
    is_speech,
    read_wav,
)
from reverberate.settings import data_root
from reverberate.store import ObjectStore, shared_store

__all__ = ["VOICE_SECONDS", "fetch_voices", "publish_voices", "voices_dir"]

#: Seconds of speech kept per voice. Longer than the owner's minute, short
#: enough that twelve voices are about a hundred megabytes on disk.
VOICE_SECONDS = 90.0
SAMPLE_RATE_HZ = 48_000
DEFAULT_SPEAKERS = tuple(f"p{n:03d}" for n in range(1, 13))


def voices_dir() -> Path:
    return data_root() / "voices"


def _longest_freeform(clips: list[Clip]) -> Clip:
    speech = [clip for clip in clips if is_speech(clip) and "freeform" in clip.member_name]
    if not speech:
        raise ValueError("no freeform speech in this shard")
    return max(speech, key=lambda clip: clip.payload_length)


def fetch_voices(
    target: Path,
    *,
    store: ObjectStore | None = None,
    speakers: tuple[str, ...] = DEFAULT_SPEAKERS,
    seconds: float = VOICE_SECONDS,
) -> list[dict[str, Any]]:
    """One passage per speaker, cut to ``seconds``, with its provenance."""
    store = store or shared_store()
    if store is None:
        raise RuntimeError("the shared store is not reachable; no voice can be fetched")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for speaker in speakers:
        shard = f"{speaker}-0000"
        clip = _longest_freeform(catalogue(store, "ears", shard))
        samples, rate = read_wav(fetch_clip(store, clip))
        if rate != SAMPLE_RATE_HZ:
            samples = audio.resample_to(samples[np.newaxis, :], float(rate), SAMPLE_RATE_HZ)[0]
        kept = samples[: int(seconds * SAMPLE_RATE_HZ)]
        name = f"{speaker}.wav"
        soundfile.write(target / name, np.asarray(kept, dtype=np.float32), SAMPLE_RATE_HZ, "PCM_16")
        entries.append(
            {
                "id": speaker,
                "url": name,
                "seconds": round(kept.size / SAMPLE_RATE_HZ, 3),
                "dataset": "ears",
                "shard": shard,
                "clip_id": clip.clip_id,
                "member_name": clip.member_name,
                "licence": EARS_LICENCE,
                "attribution": EARS_ATTRIBUTION,
                "anechoic": True,
                "synthetic": False,
            }
        )
        print(f"{speaker}: {clip.member_name}, {entries[-1]['seconds']} s")
    (target / "voices.json").write_text(json.dumps(entries, indent=1))
    return entries


def publish_voices(target: Path, store: ObjectStore | None = None) -> dict[str, str]:
    """Push the voices and their index under ``voices/`` on the store.

    Roadmap section 12.2: what goes with the dataset goes in the store. A
    minute per voice is cheap to fetch again, but the index is the record of
    which passage each speaker's file is, and that is what has to outlive a
    laptop. Returns the digest per key.
    """
    store = store or shared_store()
    if store is None:
        raise RuntimeError("the shared store is not reachable; nothing published")
    target = Path(target)
    entries = json.loads((target / "voices.json").read_text())
    digests = {}
    for name in [entry["url"] for entry in entries] + ["voices.json"]:
        digests[f"voices/{name}"] = store.put_file(f"voices/{name}", target / name)
    return digests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fetch the app's anechoic voices")
    parser.add_argument("--target", type=Path, default=None, help="default: <data root>/voices")
    parser.add_argument("--seconds", type=float, default=VOICE_SECONDS)
    parser.add_argument("--publish", action="store_true", help="push what is there to the store")
    arguments = parser.parse_args(argv)
    target = arguments.target or voices_dir()
    if arguments.publish:
        for key, digest in publish_voices(target).items():
            print(f"{key}: {digest[:12]}")
        return 0
    entries = fetch_voices(target, seconds=arguments.seconds)
    print(f"{len(entries)} voices in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
