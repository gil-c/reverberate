"""From W10's solved field to an ambisonic response, two ears, and the numbers.

The order of the chain is fixed and each step is here because leaving it out is
a mistake this project has already made or has written down:

1. **reduce**, one weight per receiver rather than eight, since W10's receivers
   are grid nodes;
2. **integrate and low cut**, because the single precision engine differentiates
   its source and reading the output without undoing that gives the derivative
   of the response;
3. **low pass at fmax**, because the stencil is dispersive at the top of its
   band and resampling first would fold exactly that part down into the audible
   range;
4. **resample to 48 kHz**, which is delivery and happens last of the four;
5. **air absorption**, ISO 9613-1 with its two parameters declared, which is
   new and which the roadmap's own table says is worth 38 per cent of T60 at
   16 kHz;
6. **encode**, per frequency, from the shells that frequency can use;
7. **decode**, through a head, with magnitude least squares above 2 kHz.

Steps 4 and 6 commute, which is why the encode may run at 48 kHz: resampling is
the same linear filter on every receiver, so it passes through a per frequency
linear fit untouched, and 24 000 bins is a third of the work of 145 638.

**What this module deliberately does not do is decide anything.** The encoder's
settings, the array and the head all arrive as arguments and all are recorded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import audio, metrics
from reverberate.audio import Atmosphere
from reverberate.experiments.engine import write_record
from reverberate.experiments.w10_ambisonic import COMMS_NAME
from reverberate.experiments.w20_render import (
    REALISED_ABSORPTION_FACTOR,
    dry_voice,
    publish,
    room_geometry,
    theory,
)
from reverberate.response import Provenance
from reverberate.spatial.array import ArrayDesign
from reverberate.spatial.binaural import (
    BinauralDecoder,
    coherence_floor,
    design_decoder,
    ild_db,
    interaural_coherence,
    itd_s,
    render,
)
from reverberate.spatial.encode import Ambisonic, EncoderSettings, encode
from reverberate.spatial.export import (
    AMBIX_NOTE,
    write_ambisonic_sofa,
    write_ambix_wav,
    write_brir_sofa,
)
from reverberate.spatial.hrtf import (
    HEAD_RADIUS_M,
    HrtfSet,
    ear_directions,
    measured_head,
    sphere_hrtf,
    woodworth_itd_s,
)
from reverberate.spatial.sh import quadrature, real_sh
from reverberate.spatial.validate import (
    direct_arrival_sample,
    direction_of_arrival,
    direction_to_scene_point,
    energy_per_order,
)
from reverberate.store import digest_of_file, shared_store
from reverberate.wave.voxelise import cache_root

__all__ = [
    "DELIVERY_RATE_HZ",
    "YAWS_DEG",
    "band_directions",
    "centre_identity",
    "binaural_measures",
    "decoders",
    "encode_run",
    "finish_run",
    "pressures_of_run",
    "spatial_report",
    "sphere_head",
    "write_audio",
    "main",
]

DELIVERY_RATE_HZ = 48000.0

#: What the artefacts are licensed as. HSSD is CC BY-NC, so anything derived
#: from its geometry inherits the non-commercial term, and so does the EARS
#: speech the audio is convolved with.
LICENCE = "CC BY-NC 4.0"

#: Head orientations rendered. Not a sweep: four angles a listener can check by
#: ear, and enough to show that the rotation is the exact one it claims to be.
YAWS_DEG = (0.0, 45.0, 90.0, -90.0)

#: The band the direction of arrival is measured in, per octave. Below 250 Hz a
#: 0.5 s response has too few cycles in a direct sound window to place one.
DOA_BANDS_HZ = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


def pressures_of_run(
    run_dir: Path,
    *,
    fmax_hz: float,
    comms_path: Path,
    lowcut_hz: float,
    delivery_rate_hz: float = DELIVERY_RATE_HZ,
    air: Atmosphere | None = None,
    sound_speed_m_s: float = 343.2,
    lowcut_order: int = 4,
) -> tuple[np.ndarray, float]:
    """One row per receiver at the delivery rate, air absorption applied or not.

    ``None`` skips the filter and is then an omission the report has to name.
    ``lowcut_order`` is the Butterworth order of the high pass; four leaves the
    modes just under the cut ringing, and a room whose lowest mode is real but
    whose damping is not may want eight.
    """
    reduced, differentiated = audio.read_engine_output(run_dir, comms_path)
    signals = audio.integrate_and_lowcut(
        reduced.signals,
        1.0 / reduced.sample_rate_hz,
        differentiated=differentiated,
        fcut=lowcut_hz,
        order=lowcut_order,
    )
    signals = audio.lowpass(signals, reduced.sample_rate_hz, fmax_hz)
    signals = audio.resample_to(signals, reduced.sample_rate_hz, delivery_rate_hz)
    if air is not None:
        signals = audio.apply_air_absorption(
            signals,
            delivery_rate_hz,
            sound_speed_m_s=sound_speed_m_s,
            atmosphere=air,
        )
    return signals, delivery_rate_hz


def sphere_head(
    sample_rate_hz: float, filter_length: int, *, quadrature_degree: int = 60
) -> HrtfSet:
    """The analytic head, sampled on the decoder's own frequency grid."""
    grid, weights = quadrature(quadrature_degree)
    frequency = np.fft.rfftfreq(filter_length, 1.0 / sample_rate_hz)
    sampled = sphere_hrtf(grid, frequency)
    return HrtfSet(sampled.responses, grid, frequency, sampled.description, weights=weights)


def decoders(
    order: int,
    sample_rate_hz: float,
    *,
    filter_length: int = 512,
    magls_cut_on_hz: float = 2000.0,
    measured_path: Path | None = None,
    plain: bool = False,
) -> tuple[dict[str, BinauralDecoder], dict[str, Any]]:
    """The decoders this run renders through, and what their heads are.

    One head is rendered by default, the analytic sphere, through magnitude
    least squares. A measured head joins it when ``measured_path`` names one.
    The sphere is never dropped: it is the one with a closed form to be checked
    against, where a measured set is a file that has to be fetched and whose
    conventions have to be believed.

    **The plain truncation is off by default, and that is not a shortcut.** It
    is the control that says what magnitude least squares buys, and for a long
    time this rendered it beside every run to keep that control honest. It does
    not have to: the difference between the two decoders is a property of the
    decoders against a head whose response is a closed form, not of any room, so
    :func:`decoder_accuracy` measures it in seconds without a solve. Rendering a
    knowingly worse decode through half a gigabyte of response added four files
    and four rows to every listening test and told nobody anything the closed
    form does not. ``plain=True`` brings it back for a listen.
    """
    heads: dict[str, Any] = {}
    head = sphere_head(sample_rate_hz, filter_length)
    heads["sphere"] = {"description": head.description, "licence": None}
    built = {
        "sphere_magls": design_decoder(
            head,
            order=order,
            sample_rate_hz=sample_rate_hz,
            filter_length=filter_length,
            magls_cut_on_hz=magls_cut_on_hz,
        ),
    }
    if plain:
        built["sphere_plain"] = design_decoder(
            head,
            order=order,
            sample_rate_hz=sample_rate_hz,
            filter_length=filter_length,
            magls_cut_on_hz=float("inf"),
            covariance_constraint=False,
        )
    if measured_path is not None:
        measured, metadata = measured_head(measured_path, sample_rate_hz, filter_length)
        heads["measured"] = {
            "description": measured.description,
            "file": Path(measured_path).name,
            "sha256": digest_of_file(Path(measured_path)),
            **{k: v for k, v in metadata.items() if k in ("licence", "author", "organisation")},
            "note": (
                "a measured head carries its own licence, which is not this "
                "project's; see the report's licence_conflict field before "
                "publishing anything decoded through it"
            ),
        }
        built["measured_magls"] = design_decoder(
            measured,
            order=order,
            sample_rate_hz=sample_rate_hz,
            filter_length=filter_length,
            magls_cut_on_hz=magls_cut_on_hz,
        )
    return built, heads


def decoder_accuracy(
    order: int,
    sample_rate_hz: float,
    *,
    filter_length: int = 512,
    magls_cut_on_hz: float = 2000.0,
    directions: int = 200,
    bands_hz: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
) -> dict[str, Any]:
    """What magnitude least squares buys, against a head with a closed form.

    This is the control for the decoder that ships, and it needs no room and no
    solve: the reference is :func:`sphere_hrtf`, which is exact, and the test
    directions are a Fibonacci sphere that was not used to fit either decoder.
    For each direction the ambisonic coefficients of a plane wave are the real
    spherical harmonics of that direction, so the decoded ear spectrum is the
    decoder's filters weighted by them.

    Reports, per octave, the mean level error in decibels and its spread across
    directions. The spread is the half that matters as much as the mean: a
    truncated decode does not lose the same level in every direction, so it
    moves a source's timbre as the head turns.
    """
    frequency = np.fft.rfftfreq(filter_length, 1.0 / sample_rate_hz)
    head = sphere_head(sample_rate_hz, filter_length)
    fitted = {
        "magls": design_decoder(
            head,
            order=order,
            sample_rate_hz=sample_rate_hz,
            filter_length=filter_length,
            magls_cut_on_hz=magls_cut_on_hz,
        ),
        "plain": design_decoder(
            head,
            order=order,
            sample_rate_hz=sample_rate_hz,
            filter_length=filter_length,
            magls_cut_on_hz=float("inf"),
            covariance_constraint=False,
        ),
    }
    index = np.arange(directions) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / directions)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    unit = np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], -1
    )
    truth = np.abs(sphere_hrtf(unit, frequency).responses)
    basis = real_sh(order, unit)

    rows: list[dict[str, Any]] = []
    errors = {
        name: 20.0
        * np.log10(
            np.abs(np.einsum("dc,ecf->edf", basis, np.fft.rfft(decoder.filters, axis=-1)))
            / np.maximum(truth, 1e-12)
        )
        for name, decoder in fitted.items()
    }
    for centre in bands_hz:
        band = (frequency >= centre / np.sqrt(2.0)) & (frequency < centre * np.sqrt(2.0))
        if not band.any():
            continue
        row: dict[str, Any] = {"band_hz": int(centre)}
        for name, error in errors.items():
            inside = error[:, :, band]
            row[f"{name}_level_db"] = round(float(np.mean(inside)), 2)
            row[f"{name}_spread_db"] = round(float(np.std(inside)), 2)
        rows.append(row)
    return {
        "what": (
            "level error of an order "
            f"{order} decode against the analytic head's own response, per octave, "
            f"over {directions} directions that did not fit either decoder"
        ),
        "why": (
            "the control for the decoder that ships. It is a property of the "
            "decoders and of a head with a closed form, not of a room, so it "
            "needs no solve and no rendered response to stand behind it"
        ),
        "magls_cut_on_hz": magls_cut_on_hz,
        "per_band": rows,
    }


def band_directions(
    ambisonic: Ambisonic,
    reference: np.ndarray,
    *,
    bands_hz: tuple[float, ...] = DOA_BANDS_HZ,
    window_s: float = 0.002,
    start: int | None = None,
) -> list[dict[str, Any]]:
    """Direction of arrival of the direct sound, per octave band, against the geometry.

    This is the check that catches every silent failure at once. A mirrored
    frame, an inverted odd order or the wrong kind of Hankel function all leave
    every level untouched and move this number by tens of degrees. At the top
    of the band it also measures something real rather than only guarding: the
    grid's own dispersion is direction dependent, so a residue here at 16 kHz
    that is absent at 1 kHz is the scheme and not the code.
    """
    rate = ambisonic.sample_rate_hz
    arrival = direct_arrival_sample(ambisonic) if start is None else start
    length = max(int(round(window_s * rate)), 16)
    rows = []
    for centre in bands_hz:
        low, high = centre / np.sqrt(2.0), min(centre * np.sqrt(2.0), 0.49 * rate)
        if low >= high:
            continue
        filtered = audio.lowpass(ambisonic.signals, rate, high)
        filtered = filtered - audio.lowpass(filtered, rate, low)
        banded = Ambisonic(filtered, rate, ambisonic.order, ambisonic.centre)
        measured = direction_of_arrival(banded, start=max(arrival - length // 4, 0), length=length)
        # A window shorter than a period cannot place a direction: the pressure
        # and velocity product integrates over less than a cycle and the answer
        # comes back reversed. A reader must not have to know that to read the
        # table, so each row says whether it may be read.
        #
        # The two thresholds are calibrated on W10's rehearsal rather than
        # chosen: there, half a cycle gave a 180 degree error at a
        # concentration of 0.53, one cycle gave 0.07 degrees at 0.88, and
        # everything above gave better than 0.02 degrees at 0.99 or more.
        cycles = length / rate * centre
        rows.append(
            {
                "band_hz": int(centre),
                "error_deg": round(measured.angle_to_deg(reference), 3),
                "cycles_in_window": round(float(cycles), 2),
                "usable": bool(cycles >= 1.0 and measured.concentration >= 0.6),
                **measured.record(),
            }
        )
    return rows


def binaural_measures(
    brir: np.ndarray,
    sample_rate_hz: float,
    source_azimuth_rad: float,
    *,
    coherence_bands_hz: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    late_from_s: float = 0.05,
) -> dict[str, Any]:
    """Interaural delay, level difference and coherence, beside what a sphere predicts.

    The coherence is measured per octave, because that is the only form in which
    it tests the roadmap's own claim: a room response's energy sits in the
    octaves below 1 kHz, where the ears are genuinely coherent because the head
    is small against the wavelength, so a broadband figure describes those and
    says nothing about the octaves the prediction is about.
    """
    per_band = []
    for band in coherence_bands_hz:
        row: dict[str, Any] = {"band_hz": int(band)}
        for lag in ("max", "zero"):
            times, coherence = interaural_coherence(brir, sample_rate_hz, band_hz=band, lag=lag)
            late = times > (times[0] + late_from_s) if times.size else np.zeros(0, dtype=bool)
            if not late.any():
                break
            row[f"{lag}_lag"] = round(float(np.mean(coherence[late])), 4)
            # The floor of the estimator itself, on ears that share nothing.
            # Without it a reader compares a measurement against a theoretical
            # value it cannot reach: the peak over lags reads 0.19 at 8 kHz when
            # the true answer is zero.
            row[f"{lag}_lag_floor"] = round(
                coherence_floor(sample_rate_hz, band_hz=band, lag=lag), 4
            )
        else:
            per_band.append(row)
    times, coherence = interaural_coherence(brir, sample_rate_hz)
    late = times > (times[0] + late_from_s) if times.size else np.zeros(0, dtype=bool)
    return {
        "itd_us": round(itd_s(brir, sample_rate_hz) * 1e6, 1),
        "woodworth_itd_us": round(float(woodworth_itd_s(np.array(source_azimuth_rad))) * 1e6, 1),
        "ild_db": round(ild_db(brir), 2),
        "late_coherence_broadband": (
            round(float(np.mean(coherence[late])), 4) if late.any() else None
        ),
        "late_coherence_per_band": per_band,
        "late_coherence_note": (
            "roadmap 5.5 predicts a diffuse tail is nearly incoherent between "
            "the ears above 1 kHz, about 0.04 at 8 kHz. Only the zero lag rows "
            "test that: the broadband figure is dominated by the octaves below "
            "1 kHz where two ears are coherent because the head is small "
            "against the wavelength, and the peak over lags is biased upwards "
            "by its own finite window, which is what each floor states"
        ),
    }


def _cache_entry(plan: dict[str, Any]) -> tuple[Path, Path] | None:
    """The voxelisation this run read and the model it was built from.

    The engine deletes ``vox_out.h5`` and ``sim_mats.h5`` from a run directory
    once it is finished with them, so the surface and the absorption the solver
    actually realised live in the cache entry and nowhere else. Absent is not an
    error: a run fetched onto another machine gets a report without a theory
    section rather than no report at all.

    The root falls back to this machine's cache when the plan does not name one,
    which is what a plan written before it recorded that looks like.
    """
    key = plan.get("cache_key")
    if not key:
        return None
    root = Path(str(plan["cache_root"])) if plan.get("cache_root") else cache_root()
    entry = root / str(key)
    manifest = entry / "manifest.json"
    if not (entry / "vox_out.h5").is_file() or not manifest.is_file():
        return None
    model = Path(str(json.loads(manifest.read_text()).get("model_json", "")))
    return (entry, model) if model.is_file() else None


def centre_identity(
    ambisonic: Ambisonic,
    array_rows: np.ndarray,
    design: ArrayDesign,
    *,
    max_frequency_hz: float | None = None,
) -> dict[str, Any] | None:
    """Check the encoded W channel against the pressure actually measured at the centre.

    **This is an identity, not an approximation.** At the expansion centre every
    radial term but the first vanishes, ``j_n(0) = 0`` for ``n > 0`` and
    ``j_0(0) = 1``, and ``Y_00 = 1``, so the field there is exactly ``a_00``.
    The plane wave convention divides by ``i^0``, which is one, so the W channel
    equals the pressure at the centre at every frequency.

    That makes it the one check on the real data that costs nothing and needs no
    reference run: a scaling error, a lost normalisation or a wrong plane wave
    convention all move it, and nothing else in this report would notice. It is
    available whenever the array holds its own centre node, which
    :func:`reverberate.spatial.array.design_array` always includes.
    """
    at_centre = np.flatnonzero(design.radii == 0.0)
    if at_centre.size != 1:
        return None
    measured = array_rows[int(at_centre[0])]
    encoded = ambisonic.signals[0]
    length = min(measured.size, encoded.size)
    spectrum_measured = np.fft.rfft(measured[:length])
    spectrum_encoded = np.fft.rfft(encoded[:length])
    frequency = np.fft.rfftfreq(length, 1.0 / ambisonic.sample_rate_hz)

    # **Per band, never as one number.** Ninety six per cent of a room
    # response's energy sits below 1 kHz, so a broadband residual is a
    # measurement of the bottom octave and hides whatever the top one is doing.
    # W29's own lesson, in the roadmap's words: a scalar hid a 29 per cent
    # omission behind a 5 per cent agreement.
    # Compared only where the encoder fitted. Above its band limit the encoded
    # signal is zero by construction and the measured one is not, so including
    # that region measures the band limit and not the fit: on this run it turned
    # -48 dB into -9 over an octave carrying 2.5 per cent of the energy.
    limit = float(max_frequency_hz) if max_frequency_hz else float(frequency[-1])
    rows = []
    for centre_hz in (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0):
        low, high = centre_hz / np.sqrt(2.0), centre_hz * np.sqrt(2.0)
        band = (frequency >= low) & (frequency < min(high, limit))
        reference = float(np.linalg.norm(spectrum_measured[band]))
        if not band.any() or reference == 0.0:
            continue
        residual = float(np.linalg.norm(spectrum_encoded[band] - spectrum_measured[band]))
        rows.append(
            {
                "band_hz": int(centre_hz),
                "residual_db": round(20.0 * np.log10(max(residual / reference, 1e-300)), 2),
                "share_of_energy": round(
                    float(
                        reference**2 / max(float(np.linalg.norm(spectrum_measured)) ** 2, 1e-300)
                    ),
                    5,
                ),
                "whole_band_fitted": bool(high <= limit),
            }
        )
    if not rows:
        return None
    return {
        "what": "the W channel against the pressure measured at the array's centre node",
        "why_exact": (
            "at the centre every radial term but the first vanishes, so the "
            "field there is exactly a_00, and the plane wave convention divides "
            "it by one; this is an identity and not an approximation"
        ),
        "per_band": rows,
        "worst_db": max(row["residual_db"] for row in rows),
    }


def _licence_conflict(heads: dict[str, Any]) -> dict[str, Any] | None:
    """Whether anything decoded here may be published under this project's licence.

    **Not a decision this module takes.** The room's geometry comes from HSSD
    under CC BY-NC, and a measured head may carry a share alike term, which asks
    a derivative to be licensed the same way while the non commercial term
    forbids exactly that. A response decoded through both is a derivative of
    both. The conflict is reported and the owner settles it; the sphere decode
    has no such question and can always be published.
    """
    measured = heads.get("measured")
    if measured is None:
        return None
    licence = str(measured.get("licence", ""))
    if "SA" not in licence.upper().replace("-", " ").split() and "BY-SA" not in licence.upper():
        return None
    return {
        "head_licence": licence,
        "project_licence": LICENCE,
        "what": (
            "a share alike head decoded into non commercial geometry is a "
            "derivative of two licences that do not compose"
        ),
        "so": (
            "the measured decode is written locally for listening and is not "
            "published; the sphere decode carries no such question"
        ),
    }


def _octave_rows(signal: np.ndarray, sample_rate_hz: float, fmax_hz: float) -> dict[str, Any]:
    band = metrics.measure(signal, int(sample_rate_hz))
    return {
        "bands_hz": [int(v) for v in band.bands],
        # **The whole band, not its centre.** An octave centred on 16 kHz runs
        # to 22.6 kHz, which the encoder zeroes, and that part carries 2.5 per
        # cent of a real response's energy: enough to turn a T30 of 0.22 s into
        # one of 0.98 s. Testing the centre marks such a band as measured when
        # three quarters of an octave of it is not.
        # bool(), because numpy 2's own boolean is not JSON serialisable and a
        # report that cannot be written is worse than one that is wrong.
        "in_band": [bool(float(v) * float(np.sqrt(2.0)) <= fmax_hz) for v in band.bands],
        "rt60_s": [_finite(v) for v in band.rt60],
        "edt_s": [_finite(v) for v in band.edt],
        "c50_db": [_finite(v) for v in band.c50],
        "drr_db": [_finite(v) for v in band.drr],
    }


def _finite(value: float) -> float | None:
    number = float(value)
    return None if not np.isfinite(number) else round(number, 4)


def rehearsal_report(
    run_dir: Path, settings: EncoderSettings, *, window_pad: int = 0
) -> dict[str, Any]:
    """What the box was run for: does the chain point the right way, and how far off is 16 kHz.

    The response is cut at the first wall reflection, which the plan computed
    from the geometry before anything was solved, so what is encoded is a free
    field monopole and its answer is known. The encode is done twice, with the
    ideal wavenumber and with the scheme's own, and the difference between the
    two direction errors is the whole question.
    """
    plan = json.loads((run_dir / "plan.json").read_text())
    rehearsal = plan["rehearsal"]
    positions = np.load(run_dir / "array_positions.npy")
    design = ArrayDesign(
        centre=np.asarray(plan["centre"], dtype=float),
        positions=positions,
        offsets=positions - np.asarray(plan["centre"], dtype=float),
        radii=np.linalg.norm(positions - np.asarray(plan["centre"], dtype=float), axis=1),
        shell=np.zeros(positions.shape[0], dtype=int),
        nominal_radii=(rehearsal["array_radius_cells"] * rehearsal["step_m"],),
        grid_step_m=rehearsal["step_m"],
    )
    grid_rate = float(plan["sample_rate_hz"])
    cut = int(rehearsal["first_reflection_sample"]) + window_pad

    reduced, differentiated = audio.read_engine_output(
        run_dir / "source0", run_dir / "comms" / COMMS_NAME
    )
    signals = audio.integrate_and_lowcut(
        reduced.signals, 1.0 / grid_rate, differentiated=differentiated, fcut=20.0
    )
    signals = audio.lowpass(signals, grid_rate, plan["encoder"]["max_frequency_hz"])
    direct = signals[:, :cut]

    reference = direction_to_scene_point(
        Ambisonic(np.zeros((4, 4)), grid_rate, 1, design.centre),
        np.asarray(plan["source"], dtype=float),
    )
    rows = {}
    for dispersion in ("ideal", "numerical"):
        variant = EncoderSettings(
            order=settings.order,
            fit_order=settings.fit_order,
            gain_cap_db=settings.gain_cap_db,
            gate_margin=settings.gate_margin,
            gate_taper=settings.gate_taper,
            max_frequency_hz=settings.max_frequency_hz,
            dispersion=dispersion,
        )
        ambisonic = encode(
            direct, grid_rate, design, sound_speed_m_s=plan["sound_speed_m_s"], settings=variant
        )
        rows[dispersion] = band_directions(
            ambisonic, reference, window_s=cut / grid_rate * 0.8, start=0
        )
    return {
        "run": run_dir.name,
        "what": (
            "the whole spatial chain on a box, over the window in which the "
            "array hears the direct sound alone"
        ),
        "rehearsal": rehearsal,
        "window_samples": cut,
        "reference_direction": [round(float(v), 5) for v in reference],
        "direction_of_arrival": rows,
        "reading": (
            "read only the rows marked usable: a direct sound window is short, "
            "and below a couple of periods an intensity vector can come back "
            "reversed. Where it is usable, the error is the chain's own at the "
            "bottom of the band and the grid's direction dependent dispersion "
            "at the top, which is the only thing that changes between the two "
            "encodes"
        ),
    }


@dataclass(frozen=True)
class EncodedRun:
    """A room run read, filtered and encoded, with everything the report reads."""

    run_dir: Path
    plan: dict[str, Any]
    design: ArrayDesign
    ambisonic: Ambisonic
    array_rows: np.ndarray
    extra_rows: np.ndarray
    extras: np.ndarray
    source: np.ndarray
    centre: np.ndarray
    rate: float
    fmax: float
    lowcut_hz: float
    lowcut_order: int
    room: dict[str, Any] | None
    theory_record: dict[str, Any] | None
    model_path: str | None


def room_report(
    run_dir: Path,
    settings: EncoderSettings,
    *,
    air: Atmosphere | None,
    lowcut_hz: float | None = None,
    yaws_deg: tuple[float, ...] = YAWS_DEG,
    filter_length: int = 512,
    measured_path: Path | None = None,
    plain_decode: bool = False,
    rendered: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode the room run, decode it through the heads, and measure all of it.

    ``rendered`` is filled in with the encoded response and the decoded ear
    signals when a mapping is passed, so a caller that wants to write audio or
    artefacts does not have to encode a second time. Nothing is written here.
    """
    return spatial_report(
        encode_run(run_dir, settings, air=air, lowcut_hz=lowcut_hz),
        settings,
        air=air,
        yaws_deg=yaws_deg,
        filter_length=filter_length,
        measured_path=measured_path,
        plain_decode=plain_decode,
        rendered=rendered,
    )


def encode_run(
    run_dir: Path,
    settings: EncoderSettings,
    *,
    air: Atmosphere | None,
    lowcut_hz: float | None = None,
    lowcut_order: int = 4,
) -> EncodedRun:
    """Read one room run and encode it on its own array. Nothing is written."""
    plan = json.loads((run_dir / "plan.json").read_text())
    positions = np.load(run_dir / "array_positions.npy")
    requested = np.asarray(plan["centre"], dtype=float)
    # The field is expanded about the grid node, not about the point that was
    # asked for. They differ by up to half a cell diagonal, and expanding about
    # the request would put the centre receiver at a non zero radius and cost
    # the exact identity between the W channel and the pressure measured there.
    centre = positions[int(np.argmin(np.linalg.norm(positions - requested, axis=1)))]
    source = np.asarray(plan["source"], dtype=float)
    extras = np.asarray(plan["extra_receivers"], dtype=float).reshape(-1, 3)
    design = ArrayDesign(
        centre=centre,
        positions=positions,
        offsets=positions - centre,
        radii=np.linalg.norm(positions - centre, axis=1),
        shell=np.zeros(positions.shape[0], dtype=int),
        nominal_radii=(float(np.linalg.norm(positions - centre, axis=1).max()),),
        grid_step_m=float(plan["array"]["grid_step_m"]),
        requested_centre=requested,
    )
    fmax = float(settings.max_frequency_hz or 16000.0)
    # Sabine and Eyring beside the measurement, from the solver's own boundary
    # rather than from the mesh. A decay time with nothing to compare it against
    # is a number and not a result, and the roadmap asks for both bounds.
    room: dict[str, Any] | None = None
    theory_record: dict[str, Any] | None = None
    model_path: str | None = None
    found = _cache_entry(plan)
    if found is not None:
        entry, model_json = found
        # The viewer draws the exact surface list the solver was handed, which
        # is roadmap constraint 9, so the report has to name it. The cache entry
        # is where a run's model can still be found once the engine has deleted
        # its copy of the grid.
        model_path = str(model_json)
        geometry = room_geometry(entry, model_json, sound_speed_m_s=343.0)
        room = geometry.record()
        theory_record = theory(
            geometry.volume_m3, geometry.surface_area_m2, geometry.mean_absorption
        ).record()
        if lowcut_hz is None:
            # The room's own first axial mode. Nothing below it is a mode of
            # this room, so nothing below it in the response is reverberation:
            # on W20's first run 99.7 per cent of the energy in the last half
            # second sat under it, where the fitted boundaries absorb almost
            # nothing, and it dominated every decay measure taken from the tail.
            lowcut_hz = geometry.first_axial_mode_hz
    if lowcut_hz is None:
        raise ValueError(
            "no low cut given and the voxelisation is not on this machine to "
            "derive one from; pass the room's first axial mode explicitly"
        )

    signals, rate = pressures_of_run(
        run_dir / "source0",
        fmax_hz=fmax,
        comms_path=run_dir / "comms" / COMMS_NAME,
        lowcut_hz=lowcut_hz,
        air=air,
        sound_speed_m_s=float(plan["sound_speed_m_s"]),
        lowcut_order=lowcut_order,
    )
    array_rows, extra_rows = signals[: design.count], signals[design.count :]

    ambisonic = encode(
        array_rows, rate, design, sound_speed_m_s=float(plan["sound_speed_m_s"]), settings=settings
    )
    return EncodedRun(
        run_dir=run_dir,
        plan=plan,
        design=design,
        ambisonic=ambisonic,
        array_rows=array_rows,
        extra_rows=extra_rows,
        extras=extras,
        source=source,
        centre=centre,
        rate=rate,
        fmax=fmax,
        lowcut_hz=lowcut_hz,
        lowcut_order=lowcut_order,
        room=room,
        theory_record=theory_record,
        model_path=model_path,
    )


def spatial_report(
    encoded: EncodedRun,
    settings: EncoderSettings,
    *,
    air: Atmosphere | None,
    yaws_deg: tuple[float, ...] = YAWS_DEG,
    filter_length: int = 512,
    measured_path: Path | None = None,
    plain_decode: bool = False,
    rendered: dict[str, Any] | None = None,
    ambisonic: Ambisonic | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode an encoded run through the heads and measure all of it.

    ``ambisonic`` replaces the run's own encoding when given, which is how a
    response assembled from several bands is reported against the plan of the
    band whose grid it is centred on. ``extra`` is merged into the report.
    """
    run_dir, plan, design = encoded.run_dir, encoded.plan, encoded.design
    array_rows, extra_rows, extras = encoded.array_rows, encoded.extra_rows, encoded.extras
    source, centre, rate, fmax = encoded.source, encoded.centre, encoded.rate, encoded.fmax
    room, theory_record, model_path = encoded.room, encoded.theory_record, encoded.model_path
    lowcut_hz, lowcut_order = encoded.lowcut_hz, encoded.lowcut_order
    ambisonic = encoded.ambisonic if ambisonic is None else ambisonic
    reference = direction_to_scene_point(ambisonic, source)
    times, order_energy = energy_per_order(ambisonic)
    identity = centre_identity(ambisonic, array_rows, design, max_frequency_hz=fmax)

    built, heads = decoders(
        settings.order,
        rate,
        filter_length=filter_length,
        measured_path=measured_path,
        plain=plain_decode,
    )
    azimuth = float(np.arctan2(reference[1], reference[0]))
    binaural: dict[str, Any] = {}
    responses: dict[str, dict[float, np.ndarray]] = {}
    for name, decoder in built.items():
        rows = {}
        responses[name] = {}
        for yaw in yaws_deg:
            # The listener turns left by yaw, so the field turns right by it.
            brir = render(ambisonic, decoder, field_yaw_rad=np.radians(-yaw))
            responses[name][yaw] = brir
            rows[f"yaw_{int(yaw)}"] = binaural_measures(brir, rate, azimuth - np.radians(yaw))
        binaural[name] = {"decoder": decoder.record(), "measures": rows}

    if rendered is not None:
        rendered["ambisonic"] = ambisonic
        rendered["brirs"] = responses
        rendered["source"] = source
        rendered["centre"] = centre
        rendered["plan"] = plan

    return {
        "run": run_dir.name,
        "scene_id": plan.get("scene_id"),
        "model_json": model_path,
        "room_geometry": room,
        "theory": theory_record,
        "theory_note": (
            "Sabine and Eyring on the surface and absorption the solver's own "
            "boundary nodes realise, with W3's measured factor of "
            f"{REALISED_ABSORPTION_FACTOR} applied. Both are diffuse field "
            "statements and neither describes a band below the room's Schroeder "
            "frequency"
        ),
        "room": plan.get("room"),
        "cache_key": plan.get("cache_key"),
        "geometry_sha256": plan.get("geometry_sha256"),
        "reference_run": plan.get("reference_run"),
        "reference_note": plan.get("reference_note"),
        "binaural": True,
        "sample_rate_hz": rate,
        "encoder": settings.record(),
        "array": plan["array"],
        "conditioning": plan["conditioning"],
        "air_absorption": (
            {"applied": True, **air.record()}
            if air is not None
            else {
                "applied": False,
                "why": (
                    "not requested, so the treble of the tail is overstated; at "
                    "16 kHz that is 38 per cent of T60 on this room's own decay"
                ),
            }
        ),
        "low_cut_hz": lowcut_hz,
        "low_cut_order": lowcut_order,
        "centre_identity": identity,
        "direction_of_arrival": band_directions(ambisonic, reference),
        "reference_direction": [round(float(v), 5) for v in reference],
        "order_energy": {
            "times_s": [round(float(v), 5) for v in times[:: max(len(times) // 40, 1)]],
            "energy": [
                [round(float(v), 8) for v in row]
                for row in order_energy[:: max(len(times) // 40, 1)]
            ],
            "note": (
                "a diffuse tail carries the same energy per channel in every "
                "order, so these curves should converge as the tail develops"
            ),
        },
        "omnidirectional": _octave_rows(ambisonic.signals[0], rate, fmax),
        "extra_receivers": [
            {"receiver": index, "position": extras[index].tolist(), **_octave_rows(row, rate, fmax)}
            for index, row in enumerate(extra_rows)
        ],
        "binaural_decodes": binaural,
        # The control for the decoder that ships, measured rather than rendered.
        "decoder_accuracy": decoder_accuracy(settings.order, rate, filter_length=filter_length),
        "heads": heads,
        "licence_conflict": _licence_conflict(heads),
        **(extra or {}),
    }


def write_audio(
    ambisonic: Ambisonic,
    brirs: dict[str, dict[float, np.ndarray]],
    dry: np.ndarray,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """The ambisonic WAV, and the anechoic clip through every head and every yaw.

    **One gain over every file written here, and that is the point.** Level is a
    measurement. A gain per file would make a shadowed ear as loud as a near one
    and destroy the interaural level difference, which is the cue this dataset
    exists to carry; within a single ambisonic file it would destroy the
    direction itself, which is nothing but the ratios between the channels. So
    the peak is taken over everything and the same gain is handed to each write,
    recorded beside the realised peak of each file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wet = {
        f"{head}_yaw{int(yaw)}": np.stack([audio.convolve(dry, response[ear]) for ear in range(2)])
        for head, angles in brirs.items()
        for yaw, response in angles.items()
    }
    gain = audio.peak_gain(*wet.values(), ambisonic.signals)

    written: list[dict[str, Any]] = []
    # The shared gain, handed over rather than left to be recomputed. Asking for
    # zero headroom does not disable the normalisation, it asks for a peak of
    # exactly one, which put this file at full scale beside binaural files at a
    # fifth of it while the report claimed one gain for all of them.
    ambix, _ = write_ambix_wav(ambisonic, out_dir / "ambisonic_acn_sn3d.wav", gain=gain)
    written.append({"path": ambix.name, "what": AMBIX_NOTE, "channels": ambisonic.signals.shape[0]})
    for name, block in wet.items():
        path = out_dir / f"binaural_{name}.wav"
        import soundfile

        soundfile.write(
            str(path),
            (block * gain).T,
            int(round(ambisonic.sample_rate_hz)),
            subtype="FLOAT",
            format="WAV",
        )
        written.append(
            {
                "path": path.name,
                "what": "two ears, the anechoic clip through this head at this yaw",
                "peak": round(float(np.max(np.abs(block * gain))), 5),
                "seconds": round(block.shape[1] / ambisonic.sample_rate_hz, 3),
            }
        )
    for entry in written:
        entry["write_gain"] = gain
    return written


def _rehearsal(args: argparse.Namespace) -> int:
    settings = EncoderSettings(
        order=args.order, fit_order=args.fit_order, max_frequency_hz=args.fmax
    )
    write_record(args.run, "report.json", rehearsal_report(args.run, settings))
    return 0


def finish_run(
    run_dir: Path,
    report: dict[str, Any],
    rendered: dict[str, Any],
    settings: EncoderSettings,
    *,
    fmax_hz: float,
    band: str = "high",
    audio_wanted: bool,
    publish_wanted: bool,
    seed: int,
) -> dict[str, Any]:
    """Write the SOFA artefacts, the audio and the report, and publish if asked."""
    ambisonic = rendered["ambisonic"]
    provenance = Provenance(
        scene_sha256=str(rendered["plan"].get("geometry_sha256") or ""),
        mats_hash=str(rendered["plan"].get("cache_key") or ""),
        engine="cuda",
        band=band,
        fmax_hz=float(fmax_hz),
        grid_step_m=float(rendered["plan"]["array"]["grid_step_m"]),
        points_per_wavelength=10.5,
        sound_speed_m_s=float(rendered["plan"]["sound_speed_m_s"]),
        seed=0,
        run_id=run_dir.name,
        notes=(
            "ambisonic order "
            f"{settings.order}, N3D, ACN, about one point; the effective order "
            "per frequency is in report.json and is lower below 1 kHz"
        ),
    )
    artefacts = [
        write_ambisonic_sofa(
            ambisonic,
            run_dir / "responses" / "ambisonic.sofa",
            source_position=rendered["source"],
            provenance=provenance,
            title=f"{run_dir.name}: ambisonic room impulse response",
            licence=LICENCE,
        )
    ]
    ears = ear_directions()
    for head, angles in rendered["brirs"].items():
        yaws = sorted(angles)
        artefacts.append(
            write_brir_sofa(
                np.stack([angles[yaw] for yaw in yaws]),
                np.asarray(yaws, dtype=float),
                run_dir / "responses" / f"binaural_{head}.sofa",
                sample_rate_hz=ambisonic.sample_rate_hz,
                listener_position=rendered["centre"],
                source_position=rendered["source"],
                ear_positions=HEAD_RADIUS_M * ears,
                provenance=provenance,
                title=f"{run_dir.name}: binaural room impulse responses, {head}",
                licence=LICENCE,
            )
        )
    report["artefacts"] = [path.name for path in artefacts]

    if audio_wanted:
        store = shared_store()
        if store is None:
            print("no store credentials, so no anechoic clip and no audio")
        else:
            dry, dry_record = dry_voice(store, seed)
            report["dry_voice"] = dry_record
            report["audio"] = write_audio(ambisonic, rendered["brirs"], dry, run_dir / "audio")
            audio.write_wav(run_dir / "audio" / "dry_voice.wav", dry, ambisonic.sample_rate_hz)
    write_record(run_dir, "report.json", report)

    if publish_wanted:
        store = shared_store()
        if store is None:
            raise SystemExit("no store credentials, so nothing can be published")
        files = [*artefacts, run_dir / "report.json", run_dir / "plan.json"]
        files += sorted((run_dir / "audio").glob("*.wav"))
        conflict = report.get("licence_conflict")
        if conflict:
            # Written locally for listening, kept out of the store. Publishing
            # is what a licence governs, and this one is not ours to resolve.
            files = [path for path in files if "measured" not in path.name]
            print(f"not publishing the measured decode: {conflict['what']}")
        report["published"] = publish(store, run_dir.name, files)
        write_record(run_dir, "report.json", report)
    return report


def _room(args: argparse.Namespace) -> int:
    settings = EncoderSettings(
        order=args.order, fit_order=args.fit_order, max_frequency_hz=args.fmax
    )
    air = (
        None
        if args.no_air
        else Atmosphere(temperature_c=args.temperature, humidity_percent=args.humidity)
    )
    rendered: dict[str, Any] = {}
    report = room_report(
        args.run,
        settings,
        air=air,
        lowcut_hz=args.low_cut,
        measured_path=args.measured_head,
        plain_decode=args.plain_decode,
        rendered=rendered,
    )
    finish_run(
        args.run,
        report,
        rendered,
        settings,
        fmax_hz=float(args.fmax),
        audio_wanted=bool(args.audio),
        publish_wanted=bool(args.publish),
        seed=args.seed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rehearsal = sub.add_parser("rehearsal", help="analyse the box run")
    rehearsal.add_argument("--run", type=Path, required=True)
    rehearsal.add_argument("--order", type=int, default=7)
    rehearsal.add_argument("--fit-order", type=int, default=10)
    rehearsal.add_argument("--fmax", type=float, default=16000.0)
    rehearsal.set_defaults(func=_rehearsal)

    room = sub.add_parser("room", help="encode, decode and measure the room run")
    room.add_argument("--run", type=Path, required=True)
    room.add_argument("--order", type=int, default=7)
    room.add_argument("--fit-order", type=int, default=10)
    room.add_argument("--fmax", type=float, default=16000.0)
    room.add_argument(
        "--low-cut",
        type=float,
        default=None,
        help="high pass in Hz; defaults to the room's own first axial mode",
    )
    room.add_argument("--temperature", type=float, default=20.0)
    room.add_argument("--humidity", type=float, default=50.0)
    room.add_argument("--no-air", action="store_true", help="skip air absorption and say so")
    room.add_argument("--audio", action="store_true", help="render the anechoic clip through it")
    room.add_argument(
        "--plain-decode",
        action="store_true",
        help=(
            "also decode without magnitude least squares, to listen to what it "
            "buys; the measurement of that is in the report either way"
        ),
    )
    room.add_argument("--publish", action="store_true", help="push the artefacts to the store")
    room.add_argument("--seed", type=int, default=20250101)
    room.add_argument(
        "--measured-head",
        type=Path,
        default=None,
        help="a SimpleFreeFieldHRIR file to decode through as well as the sphere",
    )
    room.set_defaults(func=_room)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
