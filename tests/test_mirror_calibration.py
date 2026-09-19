"""The calibration: its parameters apply as stated, and the fixed point finds a planted absorption.

The parameters must round trip through their record under the same key, and
apply to the materials exactly as documented. The fixed point, started at
half the true absorption of a box, must move the absorption back up; the run
must read what the pipeline wrote and write a calibration that pins the rays
and the tail it was fitted with.
"""

from __future__ import annotations

import json
from dataclasses import replace as swap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.mirror.calibration.criteria import CriteriaSettings
from reverberate.mirror.calibration.fit import (
    _mean_absorption,
    calibrate,
    choose_points,
    fixed_point,
    read_references,
    write_reference_subset,
)
from reverberate.mirror.files import load_paths
from reverberate.mirror.geometry import MaterialTable
from reverberate.mirror.ism import IsmSettings, grow_tree, paths_for
from reverberate.mirror.parameters import Parameters, apply_parameters, image_scene
from reverberate.mirror.pipeline import MirrorSettings, run
from reverberate.mirror.rays import RaySettings, trace
from reverberate.mirror.render import RenderSettings, render_point
from test_mirror_ism import RECEIVER, SOURCE, box_scene
from test_mirror_pipeline import SETTINGS, a_run
from test_mirror_pipeline import SOURCE as RUN_SOURCE

C = 343.2


def with_labels(scene: Any, labels: tuple[str, ...]) -> Any:
    m = scene.materials
    return swap(
        scene,
        labels=labels,
        materials=MaterialTable(labels, m.absorption, m.scattering, m.bands_hz, m.source),
    )


def test_parameters_round_trip_through_their_record_under_one_key() -> None:
    parameters = Parameters(
        absorption_scale=(0.9, 0.8, 1.1, 1.0, 1.0, 1.2, 0.7),
        scattering_scale=1.5,
        tail_gain_db=(1.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0),
        image_absorption_scale=tuple(0.5 for _ in range(7)),
        shell_scattering=0.4,
        rendered_with={"rays": RaySettings().record(), "render": RenderSettings().record()},
    )
    again = Parameters.from_record(parameters.record())
    assert again == parameters and again.key == parameters.key
    # The optional fields appear only when set, so older files keep their keys.
    plain = Parameters().record()
    assert not {"image_absorption_scale", "shell_scattering", "rendered_with"} & set(plain)


def test_the_images_read_their_own_absorption_and_the_class_scattering() -> None:
    scene = with_labels(box_scene(alpha=0.2, scattering=0.05), ("shell",))
    parameters = Parameters(
        image_absorption_scale=tuple(0.5 for _ in range(7)), shell_scattering=0.5
    )
    assert np.allclose(image_scene(scene, parameters).materials.absorption, 0.1)
    assert np.allclose(image_scene(scene, parameters).materials.scattering, 0.05)
    assert np.allclose(apply_parameters(scene, parameters).materials.absorption, 0.2)
    assert np.allclose(apply_parameters(scene, parameters).materials.scattering, 0.5)
    assert np.allclose(_mean_absorption(scene), 0.2)


def test_the_shell_scattering_replaces_the_shell_class_only() -> None:
    scene = box_scene(alpha=0.2, scattering=0.1)
    set_to = Parameters(scattering_scale=2.0, shell_scattering=0.6)
    assert np.allclose(
        apply_parameters(with_labels(scene, ("shell",)), set_to).materials.scattering, 0.6
    )
    assert np.allclose(apply_parameters(scene, set_to).materials.scattering, 0.2)


@pytest.mark.slow
def test_the_fixed_point_brings_the_decay_back_to_the_planted_absorption() -> None:
    truth = box_scene(alpha=0.25, scattering=0.3)
    ism = IsmSettings(max_order=2, flutter_order=2)
    paths = paths_for(truth, grow_tree(truth, SOURCE, ism), RECEIVER, ism)
    settings = RenderSettings(order=3, duration_s=0.6)
    rays = RaySettings(rays=3000, duration_s=0.6)

    def render(index: int, point_paths: Any, histogram: Any, parameters: Parameters) -> Any:
        response, _ = render_point(
            index,
            point_paths,
            histogram,
            settings,
            tail_gain_db=np.asarray(parameters.tail_gain_db),
            receiver_radius_m=rays.receiver_radius_m,
            sound_speed_m_s=C,
            seed=1,
        )
        return response

    def tracer(scene: Any) -> Any:
        return trace(scene, SOURCE, RECEIVER[None, :], rays)

    reference = render(0, paths, tracer(truth), Parameters())
    # Half the absorption rings twice as long; the step must raise the scale back.
    start = Parameters(absorption_scale=tuple(0.5 for _ in range(7)))
    best, evaluations = fixed_point(
        truth,
        {0: paths},
        {0: reference},
        render,
        tracer,
        start=start,
        criteria=CriteriaSettings(low_hz=1000.0),
        iterations=4,
        say=lambda _: None,
    )
    assert len(evaluations) == 4
    assert evaluations[-1].cost < evaluations[0].cost
    assert np.median(evaluations[-1].parameters.absorption_scale[3:]) > 0.7
    assert best.image_absorption_scale is not None


@pytest.mark.slow
def test_the_calibration_reads_the_run_and_pins_its_rays_and_tail(tmp_path: Path) -> None:
    run_dir, models = a_run(tmp_path)
    run(run_dir, "S1", RUN_SOURCE, models=models, settings=SETTINGS, say=lambda _: None)
    every = load_paths(run_dir / "mirror" / "paths_S1.npz")
    chosen = choose_points(every, 2)
    assert len(chosen) == 2 and all(np.any(every[i].order == 0) for i in chosen)

    # The same references from the field and from the subset that can travel.
    field = run_dir / "field" / "S1.h5"
    write_reference_subset(field, run_dir / "mirror" / "reference_subset_S1.h5", chosen, 1)
    from_field = read_references(run_dir, "S1", chosen, 1)
    field.rename(field.with_name("away.h5"))
    from_subset = read_references(run_dir, "S1", chosen, 1)
    for i in chosen:
        np.testing.assert_allclose(from_subset[i].signals, from_field[i].signals, rtol=1e-6)
    field.with_name("away.h5").rename(field)

    settings = MirrorSettings(ism=SETTINGS.ism, rays=swap(SETTINGS.rays, rays=60))
    best, target = calibrate(
        run_dir,
        "S1",
        RUN_SOURCE,
        settings=settings,
        points=2,
        iterations=2,
        order=1,
        say=lambda _: None,
    )
    record = json.loads(target.read_text())
    assert target.name == f"{best.key}.json" and record["points"] == chosen
    assert len(record["trajectory"]) == 2
    assert best.rendered_with is not None and best.rendered_with["rays"]["rays"] == 60
    assert MirrorSettings(parameters=best).pinned().rays.rays == 60
