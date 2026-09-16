"""The calibration moves towards a planted answer, and its record round trips.

The reference is the mirror itself rendered on a box at one absorption; the
search starts from twice that absorption and must lower its cost and its
absorption scale within a handful of evaluations. Small, on the twins.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.mirror.calibrate import (
    CostWeights,
    Parameters,
    apply_parameters,
    calibrate,
    regain,
    write_calibration,
)
from reverberate.mirror.criteria import CriteriaSettings
from reverberate.mirror.ism import IsmSettings, grow_tree, paths_for
from reverberate.mirror.rays import RaySettings, trace
from reverberate.mirror.render import RenderSettings, render, render_paths
from test_mirror_ism import RECEIVER, SOURCE, box_scene

C = 343.2


def test_parameters_round_trip_through_their_record_and_vector() -> None:
    parameters = Parameters(
        absorption_scale=(0.9, 0.8, 1.1, 1.0, 1.0, 1.2, 0.7),
        scattering_scale=1.5,
        tail_gain_db=(1.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0),
    )
    again = Parameters.from_record(parameters.record())
    assert again.key == parameters.key
    np.testing.assert_allclose(
        Parameters.from_vector(parameters.to_vector()).to_vector(), parameters.to_vector()
    )
    assert Parameters().absorption_scale == tuple(1.0 for _ in range(7))
    tied = Parameters.from_vector(parameters.to_vector(tied=True), tied=True)
    assert len(set(tied.absorption_scale)) == 1 and tied.scattering_scale == 1.5
    assert tied.to_vector(tied=True).shape == (3,)


@pytest.mark.slow
def test_the_search_lowers_the_cost_towards_the_planted_absorption(tmp_path) -> None:  # type: ignore[no-untyped-def]
    truth = box_scene(alpha=0.25, scattering=0.1)
    settings = IsmSettings(max_order=2, flutter_order=2)
    tree = grow_tree(truth, SOURCE, settings)
    paths = paths_for(truth, tree, RECEIVER, settings)
    render_settings = RenderSettings(order=3, duration_s=0.3, tail_from_s=0.015)
    reference, _ = render(paths, truth, render_settings, sound_speed_m_s=C, seed=1)

    def render_point(index, point_paths, histogram, scene, parameters):  # type: ignore[no-untyped-def]
        del index, histogram, parameters
        response, _ = render(point_paths, scene, render_settings, sound_speed_m_s=C, seed=1)
        return response

    def tracer(scene):  # type: ignore[no-untyped-def]
        return trace(scene, SOURCE, RECEIVER[None, :], RaySettings(rays=30, duration_s=0.05))

    # Start from half the absorption the reference was made with, in the tied
    # search: the first vertex of the simplex steps up towards the truth.
    start = Parameters(absorption_scale=tuple(0.5 for _ in range(7)))
    softer = apply_parameters(truth, start)
    assert np.allclose(softer.materials.absorption, 0.125)
    assert regain(paths, softer).gain[1:, 0].max() > paths.gain[1:, 0].max()
    best, evaluations = calibrate(
        truth,
        {0: paths},
        {0: reference},
        render_point,
        tracer,
        start=start,
        weights=CostWeights(),
        criteria=CriteriaSettings(),
        iterations=8,
        tied=True,
        say=lambda _: None,
    )
    assert len(evaluations) >= 6
    costs = [e.cost for e in evaluations]
    assert min(costs) < costs[0]
    assert (
        best.absorption_scale == min(evaluations, key=lambda e: e.cost).parameters.absorption_scale
    )
    assert np.mean(best.absorption_scale) > 0.5
    path = write_calibration(tmp_path, best, evaluations)
    assert path.name == f"{best.key}.json"
    assert render_paths(paths, render_settings, C).signals.shape[0] == 16
