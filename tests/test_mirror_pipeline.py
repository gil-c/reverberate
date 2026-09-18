"""The pipeline end to end on a box: the field in the reference's format, the same on any devices.

A solver model of a closed box and a mock reference field of a few points in
it: :func:`run` must derive the scene, write the mirror field on the
reference's lattice and clock, the paths, the signature and the report; and
the field must not depend on how many cores rendered it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reverberate.compute import Devices
from reverberate.mirror.direct import apply_signature
from reverberate.mirror.files import load_paths
from reverberate.mirror.ism import IsmSettings
from reverberate.mirror.pipeline import MirrorSettings, run
from reverberate.mirror.rays import RaySettings
from reverberate.viz.field_payload import check, mock_field
from test_accel_scene import write_scene

SOURCE = np.array([0.3, 0.5, 0.4])


def a_run(tmp_path: Path) -> tuple[Path, Path]:
    models = tmp_path / "models"
    models.mkdir(parents=True)
    write_scene(models / "apartment_full.json")
    run_dir = tmp_path / "run"
    (run_dir / "field").mkdir(parents=True)
    reference = mock_field(
        run_dir / "field" / "S1.h5",
        source_id="S1",
        source_position=SOURCE.tolist(),
        box_lo=[0.2, 0.0, 0.2],
        box_hi=[0.8, 1.0, 0.8],
        step_m=0.3,
        margin_m=0.0,
        heights_m=(0.5,),
        order=3,
        total_s=0.15,
    )
    # A direct sound at the path length, so the alignment has something to read.
    import h5py

    with h5py.File(reference, "a") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        direct = np.asarray(handle["direct_path_m"][...])
        for point in range(handle["ir"].shape[0]):
            block = np.asarray(handle["ir"][point]) * 1e-3
            at = int(round((direct[point] / 343.2 + 0.002) * rate))
            block[0, at] += 1.0
            block[3, at] += 1.0
            handle["ir"][point] = block
    return run_dir, models


SETTINGS = MirrorSettings(
    ism=IsmSettings(max_order=2, flutter_order=2, furniture_bounces=2),
    rays=RaySettings(rays=300, duration_s=0.15, bin_s=0.002, receiver_radius_m=0.15),
)


@pytest.mark.slow
def test_run_writes_the_field_and_the_same_field_on_one_core_or_three(tmp_path: Path) -> None:
    fields = []
    for cores in (1, 3):
        run_dir, models = a_run(tmp_path / f"cores{cores}")
        report = run(
            run_dir,
            "S1",
            SOURCE,
            models=models,
            settings=SETTINGS,
            devices=Devices.host(cores),
            say=lambda _: None,
        )
        field = run_dir / "field_mirror" / "S1.h5"
        assert check(field) == []
        assert report["trace"]["images"] == 1 + 6 + 30
        assert report["alignment"]["points_used"] > 0
        assert json.loads((run_dir / "mirror" / "report_S1.json").read_text())["source"] == "S1"
        assert (run_dir / "mirror" / "signature_S1.npy").is_file()
        assert len(load_paths(run_dir / "mirror" / "paths_S1.npz")) > 0
        assert not (run_dir / "mirror" / "render_S1").exists()
        import h5py

        with h5py.File(field, "r") as handle:
            fields.append(np.asarray(handle["ir"][...]))
    np.testing.assert_array_equal(fields[0], fields[1])


def test_a_calibration_s_rays_and_tail_are_the_render_s() -> None:
    from reverberate.mirror.parameters import Parameters

    fitted = Parameters(
        rendered_with={
            "rays": RaySettings(rays=1234).record(),
            "render": {"tail_from_s": 0.02, "tail_bursts": 6, "order": 3},
        }
    )
    pinned = MirrorSettings(parameters=fitted).pinned()
    assert pinned.rays.rays == 1234
    assert pinned.render.tail_from_s == 0.02 and pinned.render.tail_bursts == 6
    assert pinned.render.order == MirrorSettings().render.order


def test_the_card_spectrum_is_the_signature_and_the_low_cut() -> None:
    from scipy.signal import butter, sosfilt

    from reverberate.mirror.render import _through_spectrum

    rng = np.random.default_rng(4)
    rate = 48000.0
    decay = np.exp(-np.arange(24000) / 4000.0)
    signals = rng.standard_normal((4, 24000)) * decay
    taps = rng.standard_normal(64) * np.exp(-np.arange(64) / 8.0)
    sos = butter(8, 40.0, btype="high", fs=rate, output="sos")
    recursive = sosfilt(sos, apply_signature(signals, taps), axis=-1)
    spectral = _through_spectrum(signals, rate, taps, sos, np)
    assert np.max(np.abs(spectral - recursive)) < 1e-9 * np.max(np.abs(recursive))
