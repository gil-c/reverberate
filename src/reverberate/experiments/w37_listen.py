"""W37: hear the synthetic tail against the real one, on the same room.

``w37_window`` measured that a 60 ms computed window plus a mid-band-calibrated
synthetic tail reproduces the reference's per-band T30 to inside the receiver
spread. That is a number. This builds the listen, because a decay time is not
the thing the dataset is for and a tail can measure right and sound wrong.

**Three variants of the same room, the same source and the same six
receivers**, presented to the viewer as three sources so they can be switched
between in one page:

0. ``reference``: the whole 1.0 s that was actually solved, with air absorption
   applied. This is the ground truth and it cost 5.14 USD of A100 at 0.917
   USD/h billed.
1. ``truncated``: the same response with everything after 60 ms removed and
   nothing put back. This is what stopping the solver early gives on its own,
   and it is the control that says whether the tail is doing anything audible.
2. ``synthetic``: the same 60 ms, with the tail rebuilt from the *4 kHz* run of
   the same room. Nothing above 4 kHz in the tail comes from a 16 kHz solve.
   It costs 0.31 USD.

**One gain across all three variants and all eighteen files.** Level is a
measurement here, and normalising each file to its own peak would make the
truncated variant as loud as the reference and destroy the only comparison the
page exists to make. Roadmap W30 records this defect being found and fixed for
``w29_16k``, but the fix lived in a script that was never committed, so the
gain is computed here.

**:func:`shared_gain` is temporary.** ``audio.peak_gain`` now exists on the W10
branch with the same contract, and ``write_wav`` there takes a ``gain=`` so a
caller writing a comparable set can stop any file renormalising itself. When
W10 lands, delete :func:`shared_gain` and call ``audio.peak_gain``.

Usage::

    python -m reverberate.experiments.w37_listen \\
        --reference data/runs/w29_16k --mid data/runs/w27_sealed \\
        --out data/runs/w37_listen --window-ms 60
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import audio
from reverberate import cost as cost_model
from reverberate.air import Atmosphere
from reverberate.air import apply as apply_air
from reverberate.experiments.w20_render import measure_all
from reverberate.experiments.w37_window import mean_absorption_of
from reverberate.metrics import band_centres, rt60_per_band
from reverberate.response import Provenance, ResponseSet, read_raw, write_raw
from reverberate.tail import splice, transpose

__all__ = ["VARIANTS", "build", "main", "shared_gain"]

#: The three things a listener is asked to compare, in the order the page shows
#: them. The label reaches the viewer's picker, so it says what is being heard
#: rather than "source 1".
VARIANTS = (
    ("reference", "the 1.0 s that was solved, air applied"),
    ("truncated", "60 ms solved, nothing after it"),
    ("synthetic", "60 ms solved, tail rebuilt from the 4 kHz run"),
)

HEADROOM_DB = 1.0


def shared_gain(blocks: list[np.ndarray], *, headroom_db: float = HEADROOM_DB) -> float:
    """One write gain over every signal that must stay comparable.

    The peak is taken across every variant and every receiver at once. A gain
    per file would make a dead 60 ms response as loud as a full one, which is
    exactly the comparison this run exists to make possible.
    """
    peak = max((float(np.max(np.abs(block))) for block in blocks if block.size), default=0.0)
    return 1.0 if peak == 0.0 else 10.0 ** (-headroom_db / 20.0) / peak


def _variants(
    truth: np.ndarray,
    mid: np.ndarray,
    sample_rate: int,
    absorption: np.ndarray,
    *,
    window_s: float,
    mid_fmax_hz: float,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
    seed: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """The three impulse response sets, and what each of them is."""
    cut = int(round(window_s * sample_rate))
    truncated = np.zeros_like(truth)
    truncated[:, :cut] = truth[:, :cut]

    synthetic = np.zeros_like(truth)
    predictions = []
    for receiver in range(truth.shape[0]):
        prediction = transpose(
            rt60_per_band(mid[receiver], sample_rate),
            absorption,
            sample_rate,
            mid_fmax_hz=mid_fmax_hz,
            atmosphere=atmosphere,
            sound_speed_m_s=sound_speed_m_s,
        )
        predictions.append(prediction.record())
        synthetic[receiver] = splice(
            truth[receiver],
            sample_rate,
            window_s,
            np.array(prediction.t60_s),
            rng=np.random.default_rng(seed + receiver),
        )

    notes: list[dict[str, Any]] = [
        {"variant": name, "what": what, "transposition": None} for name, what in VARIANTS
    ]
    notes[2]["transposition"] = predictions
    return [truth, truncated, synthetic], notes


def build(
    reference_dir: Path,
    mid_dir: Path,
    out: Path,
    *,
    window_ms: float = 60.0,
    atmosphere: Atmosphere | None = None,
    seed: int = 20250101,
    usd_per_hour_per_card: float = 0.917,
) -> dict[str, Any]:
    """Write a run directory the viewer can serve, with the A/B in it."""
    atmosphere = atmosphere or Atmosphere()
    out.mkdir(parents=True, exist_ok=True)

    reference = read_raw(reference_dir / "responses" / "source0.h5")
    mid_response = read_raw(mid_dir / "responses" / "source0.h5")
    if reference.provenance.scene_sha256 != mid_response.provenance.scene_sha256:
        raise ValueError("the reference and the mid band run were shot on different geometries")

    sample_rate = int(round(reference.sample_rate_hz))
    sound_speed = reference.provenance.sound_speed_m_s
    truth = np.atleast_2d(
        apply_air(reference.ir, reference.sample_rate_hz, atmosphere, sound_speed_m_s=sound_speed)
    )
    mid = np.atleast_2d(
        apply_air(
            mid_response.ir, mid_response.sample_rate_hz, atmosphere, sound_speed_m_s=sound_speed
        )
    )

    source_report = json.loads((reference_dir / "report.json").read_text())
    absorption = mean_absorption_of(source_report, len(band_centres(sample_rate)))
    responses, notes = _variants(
        truth,
        mid,
        sample_rate,
        absorption,
        window_s=window_ms / 1000.0,
        mid_fmax_hz=mid_response.provenance.fmax_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=sound_speed,
        seed=seed,
    )

    # The dry voice is copied rather than re-chosen, so the only thing that
    # differs between this page and w29_16k's is the response.
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dry_path = reference_dir / "audio" / "dry_voice.wav"
    if not dry_path.is_file():
        raise ValueError(f"{dry_path} is missing; render the reference run's audio first")
    shutil.copy2(dry_path, audio_dir / "dry_voice.wav")

    import soundfile

    dry, dry_rate = soundfile.read(str(dry_path))
    dry = np.asarray(dry, dtype=float)
    if dry.ndim > 1:
        dry = dry[:, 0]

    wet = [
        [audio.convolve(dry, response[receiver]) for receiver in range(response.shape[0])]
        for response in responses
    ]
    gain = shared_gain([block for group in wet for block in group])

    sources: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    for index, ((name, what), response) in enumerate(zip(VARIANTS, responses, strict=True)):
        write_raw(
            ResponseSet(
                ir=response,
                sample_rate_hz=reference.sample_rate_hz,
                source_position=reference.source_position,
                receiver_positions=reference.receiver_positions,
                provenance=Provenance(
                    **{
                        **vars(reference.provenance),
                        "run_id": f"{out.name}_{name}",
                        "notes": f"{name}: {what}. {reference.provenance.notes}",
                    }
                ),
                room_volume_m3=reference.room_volume_m3,
            ),
            out / "responses" / f"source{index}.h5",
        )
        for receiver in range(response.shape[0]):
            path = audio_dir / f"source{index}_receiver{receiver}_wet.wav"
            block = np.atleast_2d(wet[index][receiver] * gain)
            soundfile.write(str(path), block.T, int(dry_rate), subtype="FLOAT", format="WAV")
            written.append(
                {
                    "path": path.name,
                    "variant": name,
                    "receiver": receiver,
                    "write_gain": gain,
                    "peak_after_gain": round(float(np.max(np.abs(block))), 4),
                }
            )
        sources.append(
            {
                "source_index": index,
                "archetype": name,
                "label": f"{name}: {what}",
                "variant": name,
                "what": what,
                "measures": measure_all(
                    response,
                    reference.sample_rate_hz,
                    fmax_hz=reference.provenance.fmax_hz,
                ),
            }
        )

    plan = json.loads((reference_dir / "plan.json").read_text())
    placement = dict(plan["placement"])
    # One entry per variant so the page's source list lines up with its samples.
    placement["sources"] = [{**placement["sources"][0], "archetype": name} for name, _ in VARIANTS]
    (out / "plan.json").write_text(
        json.dumps({**plan, "placement": placement, "run": out.name}, indent=2) + "\n"
    )

    grid_points = int(source_report["cost"]["grid_points"])
    rate = float(source_report["cost"]["steps"]) / reference.duration_s
    full_cost = cost_model.estimate(
        grid_points, reference.duration_s, rate, usd_per_hour_per_card=usd_per_hour_per_card
    )
    short_cost = cost_model.estimate(
        grid_points, window_ms / 1000.0, rate, usd_per_hour_per_card=usd_per_hour_per_card
    )

    report = {
        **source_report,
        "run": out.name,
        "reference_run": reference_dir.name,
        "mid_band_run": mid_dir.name,
        "window_ms": window_ms,
        "atmosphere": atmosphere.record(),
        "variants": notes,
        "audio": written,
        "write_gain": gain,
        "gain_note": (
            "one gain over every variant and every receiver. Level is a "
            "measurement, and a gain per file would make the truncated variant "
            "as loud as the reference and destroy the comparison."
        ),
        "cost": {
            **source_report["cost"],
            "reference_usd": full_cost.usd,
            "synthetic_usd": short_cost.usd,
            "saving": full_cost.usd / short_cost.usd,
            "usd_per_hour_per_card": usd_per_hour_per_card,
        },
        "placement": placement,
        "sources": sources,
        "band_note": (
            f"Three variants of one room. reference is the whole {reference.duration_s:.1f} s "
            f"that was solved; truncated keeps only the first {window_ms:.0f} ms; synthetic "
            f"keeps that and rebuilds the tail from the {mid_response.provenance.fmax_hz / 1000:g} "
            "kHz run of the same room. Air absorption is applied to all three. Pick between them "
            "above and listen at the bottom; the gain is shared, so a difference in loudness is "
            "a real difference."
        ),
        "omissions": [
            *source_report.get("omissions", []),
            "the synthetic tail is noise with a measured per-band decay: it "
            "carries no individual reflection, which is correct above 1 kHz "
            "where interaural coherence in a diffuse field is about 0.04, and "
            "is why the low band is not synthesised at all",
            "one room whose absorption is nearly flat with frequency, so this "
            "listen cannot separate the transposition rule from copying the "
            "mid band; W30 chose the kitchen to make that residual large",
        ],
    }
    report.pop("omissions_note", None)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic tail listening test.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-ms", type=float, default=60.0)
    parser.add_argument("--relative-humidity-pct", type=float, default=50.0)
    args = parser.parse_args(argv)

    report = build(
        args.reference,
        args.mid,
        args.out,
        window_ms=args.window_ms,
        atmosphere=Atmosphere(relative_humidity_pct=args.relative_humidity_pct),
    )

    print(f"{report['run']}: {len(report['audio'])} files, one gain {report['write_gain']:.3f}")
    print(
        f"reference {report['cost']['reference_usd']:.2f} USD against "
        f"{report['cost']['synthetic_usd']:.2f} USD, {report['cost']['saving']:.1f}x, "
        f"at {report['cost']['usd_per_hour_per_card']:.3f} USD/h per card\n"
    )
    top = {8000, 16000}
    print(f"{'variant':>10}  T30 at 8 kHz and 16 kHz, receiver 0")
    for source in report["sources"]:
        row = source["measures"][0]
        cells = "  ".join(
            f"{centre}: {value:.3f} s"
            for centre, value in zip(row["bands_hz"], row["rt60_s"], strict=True)
            if centre in top and value is not None
        )
        print(f"{source['variant']:>10}  {cells}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
