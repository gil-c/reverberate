"""Publishing the voices: every file and the index reach the store."""

from __future__ import annotations

import json
from pathlib import Path

from reverberate.store import MemoryStore
from reverberate.viz.voices import publish_voices


def test_publishing_puts_every_voice_and_the_index_in_the_store(tmp_path: Path) -> None:
    for name in ("p001.wav", "p002.wav"):
        (tmp_path / name).write_bytes(b"RIFF")
    (tmp_path / "voices.json").write_text(json.dumps([{"url": "p001.wav"}, {"url": "p002.wav"}]))
    store = MemoryStore()

    digests = publish_voices(tmp_path, store)

    assert set(digests) == {"voices/p001.wav", "voices/p002.wav", "voices/voices.json"}
    assert store.exists("voices/voices.json")
