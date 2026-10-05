"""The rendered signal on disk: raw float32 frames beside a JSON header.

Twenty minutes of order 7 at 48 kHz are 57.6 million frames of 64 float32,
14.7 GB. The container is the plainest that holds them:

- ``<name>.f32``: the frames one after the other, each its 64 channels in
  ACN order, little endian float32. Frame ``n`` starts at byte ``256 n``.
- ``<name>.json``: what the bytes are (rate, order, channels, frames,
  convention, the pack's identity, the sources summed) and, once the render
  is complete, their SHA-256 and their peak.

Why not HDF5, which the pack is. The signal is one rectangular array
written once from start to end and read by time range: there is nothing
ragged to index. Frames in a plain file are appended as they are rendered,
with no chunk cache and no B-tree to rewrite; a time range is one seek and
one read, so a server streams it with ``numpy.memmap`` or a byte range and
needs no library; and the checksum the audit compares (the page's stream
against the engine's signal) is the digest of the file itself, computed
while it is written. A WAV cannot hold it: its size field stops at 4 GB.

The header is written last and atomically. A signal without its header, or
whose header says ``complete: false``, is a render that stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SCHEMA", "Signal", "open_signal", "write_header", "write_signal"]

SCHEMA = "reverberate.scene-signal"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Signal:
    """A signal on disk: its header and its frames, mapped and not loaded."""

    header: dict[str, Any]
    #: ``[frame, channel]`` float32, read only.
    frames: np.ndarray

    def read(self, start: int, stop: int) -> np.ndarray:
        """Frames ``[start, stop)`` as ``[channel, sample]``, a copy."""
        return np.ascontiguousarray(self.frames[start:stop].T)


def _paths(target: Path) -> tuple[Path, Path]:
    target = Path(target)
    return target.with_suffix(".f32"), target.with_suffix(".json")


def write_signal(
    target: Path,
    blocks: Iterable[np.ndarray],
    *,
    sample_rate_hz: float,
    order: int,
    recipe_sha256: str = "",
    sources: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``blocks`` of ``[channel, sample]`` to ``target``'s two files; the header, returned.

    The blocks are what :meth:`reverberate.render.engine.Engine.blocks`
    yields, written as they come: memory holds one block.
    """
    data_path, header_path = _paths(target)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.unlink(missing_ok=True)
    digest = hashlib.sha256()
    frames, channels, peak = 0, 0, 0.0
    with open(data_path, "wb") as handle:
        for block in blocks:
            block = np.asarray(block)
            if block.ndim != 2:
                raise ValueError("a block is [channel, sample]")
            if channels and block.shape[0] != channels:
                raise ValueError("the blocks do not agree on the channel count")
            channels = int(block.shape[0])
            payload = np.ascontiguousarray(block.T).astype("<f4", copy=False).tobytes()
            handle.write(payload)
            digest.update(payload)
            frames += int(block.shape[1])
            if block.size:
                peak = max(peak, float(np.max(np.abs(block))))
    return write_header(
        target,
        frames=frames,
        channels=channels,
        peak=peak,
        sha256=digest.hexdigest(),
        sample_rate_hz=sample_rate_hz,
        order=order,
        recipe_sha256=recipe_sha256,
        sources=sources,
        extra=extra,
    )


def write_header(
    target: Path,
    *,
    frames: int,
    channels: int,
    peak: float,
    sha256: str,
    sample_rate_hz: float,
    order: int,
    recipe_sha256: str = "",
    sources: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The header of frames already at ``target``, written last and atomically; returned."""
    data_path, header_path = _paths(target)
    header = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "data": data_path.name,
        "dtype": "<f4",
        "layout": "frames of channels",
        "sample_rate_hz": float(sample_rate_hz),
        "order": int(order),
        "channels": channels,
        "frames": frames,
        "ordering": "ACN",
        "normalisation": "N3D",
        "frame": "scene, fixed: the head's rotation is applied at decode",
        "recipe_sha256": recipe_sha256,
        "sources": list(sources),
        "peak": peak,
        "sha256": sha256,
        "complete": True,
        **(extra or {}),
    }
    scratch = header_path.with_suffix(".json.part")
    scratch.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(scratch, header_path)
    return header


def open_signal(target: Path) -> Signal:
    """The signal at ``target``, mapped: nothing is read until a range is asked for."""
    data_path, header_path = _paths(target)
    header: dict[str, Any] = json.loads(header_path.read_text(encoding="utf-8"))
    if header.get("schema") != SCHEMA or not header.get("complete"):
        raise ValueError(f"{header_path} is not a complete scene signal")
    shape = (int(header["frames"]), int(header["channels"]))
    if data_path.stat().st_size != shape[0] * shape[1] * 4:
        raise ValueError(f"{data_path} is not the size its header says")
    if shape[0] == 0:
        return Signal(header, np.zeros(shape, dtype="<f4"))
    return Signal(header, np.memmap(data_path, dtype="<f4", mode="r", shape=shape))
