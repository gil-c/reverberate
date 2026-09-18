"""Binaural decoders, exported for the browser.

The page decodes the field to two ears itself, because the decode depends on
where the listener stands and which way they face. The filters it decodes
with are designed here, once, by the same code the offline renders use
(:func:`reverberate.spatial.binaural.design_decoder`), so what is heard in the
app is what a published render would hold.

Each decoder is one raw ``float32`` file ``[ear, channel, tap]`` beside a JSON
header carrying its order, length, sample rate, head and licence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.spatial.binaural import BinauralDecoder, design_decoder
from reverberate.spatial.hrtf import measured_head

__all__ = ["FILTER_LENGTH", "export_decoders"]

FILTER_LENGTH = 512
SAMPLE_RATE_HZ = 48_000.0


def _write(
    target: Path, name: str, decoder: BinauralDecoder, extra: dict[str, Any]
) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.f32").write_bytes(np.asarray(decoder.filters, dtype="<f4").tobytes())
    record = {
        "name": name,
        "url": f"{name}.f32",
        "ears": 2,
        "channels": int(decoder.filters.shape[1]),
        "taps": decoder.length,
        **decoder.record(),
        **extra,
    }
    (target / f"{name}.json").write_text(json.dumps(record))
    return record


def export_decoders(
    target: Path,
    *,
    measured_path: Path,
    order: int = 7,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    filter_length: int = FILTER_LENGTH,
) -> list[dict[str, Any]]:
    """Write the decoder of the measured head the page listens with; its record, in a list."""
    if not Path(measured_path).is_file():
        raise FileNotFoundError(f"measured head {measured_path} is not there")
    head, metadata = measured_head(measured_path, sample_rate_hz, filter_length)
    records = [
        _write(
            Path(target),
            "measured",
            design_decoder(
                head, order=order, sample_rate_hz=sample_rate_hz, filter_length=filter_length
            ),
            {
                "licence": metadata.get("licence"),
                "attribution": {
                    key: metadata.get(key) for key in ("author", "organisation", "listener")
                },
                "file": Path(measured_path).name,
            },
        )
    ]
    (Path(target) / "decoders.json").write_text(json.dumps(records))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="export binaural decoders for the app")
    parser.add_argument("target", type=Path)
    parser.add_argument("--order", type=int, default=7)
    parser.add_argument("--measured", type=Path, required=True, help="the SOFA head")
    arguments = parser.parse_args(argv)
    for record in export_decoders(
        arguments.target, order=arguments.order, measured_path=arguments.measured
    ):
        print(f"{record['name']}: order {record['order']}, {record['taps']} taps, {record['head']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
