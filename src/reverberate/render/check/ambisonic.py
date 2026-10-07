"""The listening kit's order 7 files, and its two signals that are no clip.

The kit's WAV files hold two ears for the scene's own head: whoever listens
cannot turn his. Beside them the kit can keep what the engine rendered,
order 7 in the scene's fixed frame, so that a player decodes it with the
head its listener turns (``reverberate.viz.parts``): one signal a source, in
the format of ``docs/formats/scene-signal.md``, and the scene's own head
over the window, which a player starts from.

- :class:`StemFiles` takes a source's stem a piece at a time, as
  :func:`reverberate.render.check.run.mix_of` renders it, and writes
  ``<folder>/<name>.f32`` and its header. **A piece of zeros is not
  written**: the file has its full length and a hole where the source is
  silent, as the audit's stems have, so a source takes the disk of the time
  it sounds, 12.3 MB a second. No mix is written: it is its stems summed,
  which a reader does in one line and a player does with a fader a source.
- :func:`probe_signal` and :func:`probe_ears` feed every source one signal
  that is no clip, over the whole window: **clicks** (one sample of full
  scale a second, so that each is heard to its end) and **pink noise** (the
  check's own, at an rms of 0.1). A click says where a reflection is and
  noise says what a band holds; speech says neither as plainly. They sound
  where the scene's source is heard and nowhere else: a pack holds no path
  for a step at which its source is silent.
- :func:`head_track` is the scene's head a step, in the recipe's angles.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.check import measure
from reverberate.render.check.binaural import BLOCK, PageDecoder
from reverberate.render.check.run import CheckSettings, _pose_at
from reverberate.render.dry import DryTrack
from reverberate.render.engine import Engine
from reverberate.render.output import write_header
from reverberate.render.pack import ScenePack, Source
from reverberate.spatial.binaural import BinauralDecoder

__all__ = ["SIGNALS", "StemFiles", "head_track", "probe_ears", "probe_signal"]

#: What a source may be fed, and how the kit says it.
SIGNALS = {
    "clips": "the recipe's clips",
    "clicks": "one sample of full scale every second, through every source while it is heard",
    "pink": "pink noise, 100 Hz to 12 kHz, at an rms of 0.1, through every source while it is"
    " heard",
}
#: Seconds between two clicks: longer than a dwelling rings.
CLICK_EVERY_S = 1.0
#: A second of order 7 at 48 kHz, bytes.
BYTES_A_SECOND = 64 * 4 * 48000


def probe_signal(signal: str, samples: int, rate: float) -> np.ndarray:
    """The dry signal of ``signal`` (``clicks`` or ``pink``) over ``samples``."""
    if signal == "clicks":
        dry = np.zeros(samples)
        # The first a quarter of a second in: the page fades a start in over a block.
        dry[int(0.25 * rate) :: int(round(CLICK_EVERY_S * rate))] = 1.0
        return dry
    if signal == "pink":
        fade = np.minimum(1.0, np.arange(samples) / (0.02 * rate))
        return np.asarray(fade * fade[::-1] * measure.pink_noise(samples, rate, seed=1))
    raise ValueError(f"no signal named {signal!r}: one of clicks, pink")


class StemFiles:
    """Order 7 stems written a piece at a time, a file a source, silence left as a hole."""

    def __init__(self, folder: Path, prefix: str, suffix: str = "") -> None:
        self.folder = Path(folder)
        self.prefix, self.suffix = prefix, suffix
        self._open: dict[str, Any] = {}
        self._peak: dict[str, float] = {}
        self._channels = 0

    def path(self, source: str) -> Path:
        return self.folder / f"{self.prefix}_{source}{self.suffix}.f32"

    def keep(self, source: str, first: int, block: np.ndarray) -> None:
        """``block`` (``[channel, sample]``) of ``source``, ``first`` samples into the window."""
        block = np.asarray(block)
        self._channels = int(block.shape[0])
        if source not in self._open:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._open[source] = open(self.path(source), "wb")  # noqa: SIM115
            self._peak[source] = 0.0
        if not np.any(block):
            return
        handle = self._open[source]
        handle.seek(first * self._channels * 4)
        handle.write(np.ascontiguousarray(block.T).astype("<f4", copy=False).tobytes())
        self._peak[source] = max(self._peak[source], float(np.max(np.abs(block))))

    def close(
        self, frames: int, *, sample_rate_hz: float, order: int, extra: dict[str, Any]
    ) -> dict[str, Path]:
        """Every file brought to ``frames`` and given its header; the headers, by source.

        A source that never sounded has no file: there is nothing to play.
        """
        written: dict[str, Path] = {}
        for source, handle in self._open.items():
            handle.truncate(frames * self._channels * 4)
            handle.close()
            path = self.path(source)
            if self._peak[source] == 0.0:
                path.unlink()
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as held:
                while piece := held.read(1 << 24):
                    digest.update(piece)
            write_header(
                path,
                frames=frames,
                channels=self._channels,
                peak=self._peak[source],
                sha256=digest.hexdigest(),
                sample_rate_hz=sample_rate_hz,
                order=order,
                recipe_sha256=str(extra.get("recipe_sha256", "")),
                sources=[source],
                extra={k: v for k, v in extra.items() if k != "recipe_sha256"},
            )
            written[source] = path.with_suffix(".json")
        self._open.clear()
        return written


def probe_ears(
    pack: ScenePack,
    sources: list[Source],
    signal: str,
    lo: int,
    hi: int,
    settings: CheckSettings,
    decoder: BinauralDecoder | None,
    keep: Callable[[str, int, np.ndarray], None] | None = None,
) -> dict[str, np.ndarray]:
    """``signal`` through each of ``sources`` over ``[lo, hi)``: the two ears of each.

    As :func:`reverberate.render.check.run.mix_of` renders the clips: the
    same pieces, the same decode, the scene's own head. ``keep`` is handed
    every piece of every stem, order 7.
    """
    h = pack.header
    rate, count = h.sample_rate_hz, hi - lo
    dry = probe_signal(signal, count, rate)
    page = PageDecoder(decoder) if decoder is not None else None
    pose = _pose_at(pack)
    piece = 48 * BLOCK
    ears = {s.id: np.zeros((2, count), dtype=np.float32) for s in sources}
    for source in sources:
        settings.say(f"{source.id}: the stem, {count / rate:.0f} s ({signal})")
        track = DryTrack.from_array(dry, start_s=lo / rate, rate=rate)
        engine = Engine(pack, {source.id: track}, settings=settings.render())
        before = np.zeros((h.channels, BLOCK))
        for a in range(lo, hi, piece):
            b = min(a + piece, hi)
            stem = engine.stem(source.id, a, b)
            if keep is not None:
                keep(source.id, a - lo, stem)
            if page is not None and (np.any(stem) or np.any(before)):
                ears[source.id][:, a - lo : b - lo] = page.decode(
                    stem, pose, start=a, before=before
                )
            before = stem[:, -BLOCK:] if stem.shape[1] >= BLOCK else np.zeros((h.channels, BLOCK))
    return ears


def head_track(pack: ScenePack, first: int, stop: int) -> dict[str, Any]:
    """The scene's own head over steps ``[first, stop]``, a value a step, in the recipe's angles."""
    orientation = np.asarray(pack.listener.orientation, dtype=float)[first : stop + 1]
    return {
        "step_s": float(pack.header.step_s),
        "yaw_deg": [round(float(v), 3) for v in orientation[:, 0]],
        "pitch_deg": [round(float(v), 3) for v in orientation[:, 1]],
        "roll_deg": [round(float(v), 3) for v in orientation[:, 2]],
    }
