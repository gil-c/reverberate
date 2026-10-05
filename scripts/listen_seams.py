"""The same passages of a scene under each way of writing its level above the crossover.

    PYTHONPATH=src python scripts/listen_seams.py PACK OUT \
        --passage noise_3 483 493 --passage far_6 92 102 [--seams pair smooth constant tapered]

For every passage (a source and a window in seconds) and every seam
(:data:`reverberate.render.relevel.SEAMS`), two files through the page's
decode with the scene's own head: the source alone and the mix of every
source heard in the window, all at one gain a passage, nothing normalised
between the seams. And a blind set of the source alone, its key sealed
(:func:`reverberate.render.check.many.blind_set`). The pack is only read:
each seam is a table given to the engine (``Engine(high_gain_db=...)``).
The band under the crossover does not depend on the seam and is rendered
once a source.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.check import measure
from reverberate.render.check.binaural import BLOCK, PageDecoder, page_decoder
from reverberate.render.check.clips import ClipSource, feed_of
from reverberate.render.check.many import blind_set
from reverberate.render.check.report import defaults, write_ears
from reverberate.render.check.run import _pose_at
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.pack import read_pack
from reverberate.render.relevel import SEAMS, pack_tables

#: The loudest sample of a passage's files, dB re full scale: one gain for all of them.
PEAK_DB = -3.0


def ears_of(
    engine: Engine, page: PageDecoder, pose: Any, name: str, lo: int, hi: int, parts: tuple
) -> np.ndarray:
    """Two ears of ``parts`` of one source over ``[lo, hi)``, decoded in the page's blocks."""
    channels = engine.channels
    out = np.zeros((2, hi - lo))
    before = np.zeros((channels, BLOCK))
    piece = 48 * BLOCK
    for a in range(lo, hi, piece):
        b = min(a + piece, hi)
        stem = engine.stem(name, a, b, parts=parts)
        if np.any(stem) or np.any(before):
            out[:, a - lo : b - lo] = page.decode(stem, pose, start=a, before=before)
        before = stem[:, -BLOCK:] if stem.shape[1] >= BLOCK else np.zeros((channels, BLOCK))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pack", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--passage", nargs=3, action="append", required=True, metavar=("SOURCE", "START", "STOP")
    )
    parser.add_argument("--seams", nargs="+", default=list(SEAMS), choices=SEAMS)
    parser.add_argument("--constant-db", type=float, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--alone", action="store_true", help="the source alone, no mix")
    args = parser.parse_args()
    found = defaults(None, None, None, None)
    tables: dict[str, dict[str, np.ndarray] | None] = {}
    reports: dict[str, Any] = {}
    for seam in args.seams:
        made, reports[seam] = pack_tables(args.pack, seam=seam, constant_db=args.constant_db)
        tables[seam] = None if seam == "pair" else made
    settings = RenderSettings(workers=args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    said: dict[str, Any] = {"pack": str(args.pack), "passages": []}
    with read_pack(args.pack) as pack:
        h = pack.header
        rate = h.sample_rate_hz
        recipe = pack.recipe_json()
        clips = ClipSource(root=found["clips_root"], manifest=found["manifest"])
        page = PageDecoder(page_decoder(found["measured_head"], h.order, rate))
        pose = _pose_at(pack)
        for source, start, stop in args.passage:
            t0 = time.time()
            lo, hi = int(round(float(start) * rate)), int(round(float(stop) * rate))
            k0, k1 = int(float(start) / h.step_s), int(np.ceil(float(stop) / h.step_s))
            heard = [source]
            if not args.alone:
                heard += [
                    name
                    for name, held in pack.sources.items()
                    if name != source and np.any(np.asarray(held.audible[k0 : k1 + 1]))
                ]
            feeds = {name: feed_of(pack, name, recipe, clips) for name in heard}
            heard = [
                name
                for name in heard
                if any(a < float(stop) and b > float(start) for a, b in feeds[name].intervals)
            ]
            tracks = {name: feeds[name].track for name in heard}
            print(f"{source} {start} to {stop} s: {', '.join(heard)}", flush=True)
            # A source at a time, and an engine a seam: what one source holds is let go
            # before the next is read (a source of a whole scene is gigabytes of rows).
            ears: dict[str, dict[str, np.ndarray]] = {seam: {} for seam in args.seams}
            for name in heard:
                one = {name: tracks[name]}
                engine = Engine(pack, one, settings=settings)
                low = ears_of(engine, page, pose, name, lo, hi, ("low",))
                del engine
                for seam in args.seams:
                    engine = Engine(pack, one, settings=settings, high_gain_db=tables[seam])
                    high = ears_of(engine, page, pose, name, lo, hi, ("early", "tail"))
                    ears[seam][name] = (low + high).astype(np.float32)
                    del engine
                gc.collect()
                print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
            mixes = {
                seam: np.sum([ears[seam][n] for n in heard], axis=0, dtype=np.float64)
                for seam in args.seams
            }
            peak = max(
                max(float(np.abs(mixes[seam]).max()), float(np.abs(ears[seam][source]).max()))
                for seam in args.seams
            )
            scale = 10.0 ** (PEAK_DB / 20.0) / max(peak, 1e-12)
            label = f"{source}_{float(start):.0f}s"
            folder = args.out / label
            folder.mkdir(parents=True, exist_ok=True)
            alone: dict[str, Path] = {}
            for seam in args.seams:
                alone[seam] = folder / f"{label}__alone__{seam}.wav"
                write_ears(alone[seam], ears[seam][source], rate, scale)
                if len(heard) > 1:
                    write_ears(folder / f"{label}__mix__{seam}.wav", mixes[seam], rate, scale)
            blind_set(alone, folder / "blind_alone")
            table = pack.sources[source].level.high_gain_db
            said["passages"].append(
                {
                    "source": source,
                    "window_s": [float(start), float(stop)],
                    "in_the_mix": heard,
                    "gain_db": round(float(measure.db(scale)), 2),
                    "traced_high_gain_db_range": [
                        round(float(np.min(table[k0 : k1 + 1])), 2),
                        round(float(np.max(table[k0 : k1 + 1])), 2),
                    ],
                    "folder": str(folder),
                }
            )
    said["seams"] = {
        seam: {k: v for k, v in report.items() if k != "sources"}
        for seam, report in reports.items()
    }
    (args.out / "listen.json").write_text(json.dumps(said, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
