"""Tests for the solver run viewer.

The viewer's whole claim is that it shows the file the solver read, so the
tests that matter are the ones tying the payload back to that file rather than
to anything re-derived. The rest guard the two reductions the page depends on,
the envelope and the decay curve, because a plot that silently loses its shape
is worse than no plot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.response import Provenance, ResponseSet, write_raw
from reverberate.viz import run_view
from reverberate.viz.run_view import (
    COLOUR_BAND_HZ,
    MATERIAL_BANDS_HZ,
    PLOT_POINTS,
    SPECTROGRAM_BINS,
    SPECTROGRAM_FRAMES,
    absorption_colour,
    build_site,
    decay_curve_points,
    envelope,
    spectrogram,
    surface_groups,
)


def test_envelope_keeps_the_peak_a_stride_would_have_missed() -> None:
    """Decimating by sampling would draw a waveform quieter than it is."""
    signal = np.zeros(10_000)
    signal[4_001] = 1.0  # deliberately not on any round stride

    reduced = envelope(signal, points=64)

    assert len(reduced) == 64
    assert max(high for _, high in reduced) == pytest.approx(1.0)


def test_envelope_reports_both_extremes_of_each_bucket() -> None:
    signal = np.array([-3.0, 3.0, -1.0, 1.0])

    assert envelope(signal, points=2) == [[-3.0, 3.0], [-1.0, 1.0]]


def test_envelope_of_nothing_is_nothing_rather_than_a_crash() -> None:
    assert envelope(np.array([])) == []


def test_decay_curve_falls_and_never_reports_an_infinity() -> None:
    """The tail of a Schroeder integration is -inf, which cannot be plotted.

    It is clamped to the floor the axis draws rather than dropped, so the curve
    keeps its time base and a reader can see where the response ran out.
    """
    rng = np.random.default_rng(0)
    samples = 48_000
    decay = rng.normal(size=samples) * np.exp(-6.9078 * np.arange(samples) / 48_000 / 0.5)

    curve = decay_curve_points(decay, 48_000.0, points=128)

    assert len(curve["db"]) == len(curve["seconds"]) == 128
    assert all(np.isfinite(curve["db"]))
    assert curve["db"][0] == pytest.approx(0.0, abs=1e-6)
    assert curve["db"][-1] < -20.0
    assert curve["seconds"][-1] == pytest.approx(samples / 48_000, rel=0.01)


def test_surface_groups_pass_the_solver_file_through_untouched() -> None:
    """Constraint 9: the viewer draws the list the solver read, not a re-derivation."""
    model = {
        "mats_hash": {
            "shell": {
                "pts": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "tris": [[0, 1, 2]],
                "color": [200, 190, 180],
                "sides": [2],
            }
        }
    }

    groups = surface_groups(model)

    assert len(groups) == 1
    assert groups[0]["label"] == "shell"
    assert groups[0]["positions"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert groups[0]["indices"] == [0, 1, 2]
    assert groups[0]["triangles"] == 1


def test_the_exporters_grey_is_dropped_in_favour_of_the_measured_absorption() -> None:
    """Thirteen materials came out as one flat grey.

    ``scene_export`` writes ``[128, 128, 128]`` into every group because PFFDTD
    ignores the field, so passing it through renders every surface identically.
    The absorption is the quantity an acoustic view exists to show, so that is
    what the colour carries.
    """
    model = {
        "mats_hash": {
            "curtain": {"pts": [[0.0, 0.0, 0.0]], "tris": [], "color": [128, 128, 128]},
            "mirror": {"pts": [[0.0, 0.0, 0.0]], "tris": [], "color": [128, 128, 128]},
        }
    }
    materials = {"curtain": [0.62] * 11, "mirror": [0.03] * 11}

    groups = surface_groups(model, materials)

    assert [g["absorption"] for g in groups] == [0.62, 0.03]
    assert groups[0]["colour"] != groups[1]["colour"]
    assert [128, 128, 128] not in [g["colour"] for g in groups]


def test_a_material_with_no_measured_absorption_stays_grey() -> None:
    """Grey has to keep meaning "not known", so nothing on the ramp may be grey."""
    model = {"mats_hash": {"mystery": {"pts": [[0.0, 0.0, 0.0]], "tris": []}}}

    groups = surface_groups(model, {})

    assert groups[0]["colour"] == [128, 128, 128]
    assert groups[0]["absorption"] is None
    ramp = [absorption_colour(a / 20) for a in range(21)]
    assert all(max(c) - min(c) > 40 for c in ramp)


def test_the_material_band_centres_match_the_table_they_index() -> None:
    """The centres are copied rather than imported, so pin the copy to the source."""
    from reverberate.experiments.scene_export import BANDS

    assert list(MATERIAL_BANDS_HZ) == [float(v) for v in BANDS]
    assert COLOUR_BAND_HZ in MATERIAL_BANDS_HZ


def _provenance() -> Provenance:
    return Provenance(
        scene_sha256="a" * 64,
        mats_hash="b" * 32,
        engine="gpu",
        band="wave",
        fmax_hz=4000.0,
        grid_step_m=0.0082,
        points_per_wavelength=10.5,
        sound_speed_m_s=343.0,
        seed=1,
        run_id="test",
        solver_commit="c" * 40,
        wall_clock_s=1.0,
        notes="two bare points in a room are not a pair of ears",
    )


def _write_run(root: Path) -> Path:
    """A miniature run with the same shape as a real one."""
    run = root / "run"
    (run / "responses").mkdir(parents=True)
    (run / "audio").mkdir()

    model = {
        "mats_hash": {
            "shell": {
                "pts": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                "tris": [[0, 1, 2]],
                "color": [200, 190, 180],
                "sides": [2],
            }
        }
    }
    model_path = root / "model.json"
    model_path.write_text(json.dumps(model))

    rate = 8000.0
    ir = np.zeros((2, 800))
    ir[:, 10] = 1.0
    write_raw(
        ResponseSet(
            ir=ir,
            sample_rate_hz=rate,
            source_position=np.array([0.5, 1.2, 0.5]),
            receiver_positions=np.array([[1.0, 1.2, 1.0], [1.5, 1.2, 1.0]]),
            provenance=_provenance(),
            room_volume_m3=30.0,
        ),
        run / "responses" / "source0.h5",
    )
    for name in ("dry_voice.wav", "source0_receiver0_wet.wav"):
        (run / "audio" / name).write_bytes(b"RIFF____WAVEfmt ")

    measures: list[dict[str, Any]] = [
        {
            "receiver": index,
            "bands_hz": [125, 250],
            "in_band": [True, False],
            "rt60_s": [0.5, None],
            "edt_s": [0.4, None],
            "c50_db": [3.0, None],
            "drr_db": [1.0, None],
        }
        for index in (0, 1)
    ]
    report = {
        "run": "run",
        "scene_sha256": "a" * 64,
        "model_json": str(model_path),
        "cache_key": "b" * 32,
        "responses": 2,
        "binaural": False,
        "binaural_note": "two bare points in a room are not a pair of ears",
        "placement_seed": 1,
        "placement": {
            "seed": 1,
            "sources": [{"position": [0.5, 1.2, 0.5], "archetype": "speech"}],
            "receivers": [{"position": [1.0, 1.2, 1.0]}, {"position": [1.5, 1.2, 1.0]}],
        },
        "cost": {},
        "room": {
            "volume_m3": 30.0,
            "surface_area_m2": 60.0,
            "shell_area_m2": 40.0,
            "mean_absorption": 0.3,
            "first_axial_mode_hz": 40.0,
            "per_class": [],
        },
        "low_cut_hz": 40.0,
        "theory": {"sabine_rt60_s": 0.2, "eyring_rt60_s": 0.18},
        "theory_shell_only": {"sabine_rt60_s": 0.6, "eyring_rt60_s": 0.55},
        "theory_note": "two bounds, not one prediction",
        "band_note": "bands above fmax are the filter, not the room",
        "dry_voice": {"member_name": "p001/x.wav", "licence": "CC BY-NC 4.0", "anechoic": True},
        "omissions": ["no air absorption"],
        "sources": [{"source_index": 0, "measures": measures}],
    }
    (run / "report.json").write_text(json.dumps(report))
    (run / "plan.json").write_text(json.dumps({"scene_id": "102344022", "room": "bedroom.001"}))
    return run


def test_build_site_writes_a_payload_that_needs_nothing_else(tmp_path: Path) -> None:
    """Offline and self-contained: the data and its audio, nothing more.

    No markup is written here. The viewer is the apartment browser, which owns
    the page; this half owns the numbers.
    """
    run = _write_run(tmp_path)

    view = build_site(run, tmp_path / "site")

    assert (tmp_path / "site" / "run.json").is_file()
    assert not (tmp_path / "site" / "run.html").exists()
    assert (tmp_path / "site" / "audio" / "dry_voice.wav").is_file()
    assert view.groups == 1
    assert view.triangles == 1
    assert view.samples == 2
    assert view.audio_files == 2
    assert "2 samples" in view.summary()


def test_the_payload_carries_every_pair_with_its_own_plots(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    build_site(run, tmp_path / "site")
    payload = json.loads((tmp_path / "site" / "run.json").read_text())

    assert [sample["id"] for sample in payload["samples"]] == ["s0r0", "s0r1"]
    assert len(payload["sources"]) == 1
    assert len(payload["receivers"]) == 2
    for sample in payload["samples"]:
        assert 0 < len(sample["envelope"]) <= PLOT_POINTS
        assert len(sample["decay"]["db"]) == len(sample["decay"]["seconds"])
        assert sample["measures"]["receiver"] == sample["receiver_index"]


def test_a_pair_without_audio_says_so_rather_than_pointing_at_a_missing_file(
    tmp_path: Path,
) -> None:
    """Only receiver 0 has a rendered WAV, so receiver 1 must offer none."""
    run = _write_run(tmp_path)

    build_site(run, tmp_path / "site")
    payload = json.loads((tmp_path / "site" / "run.json").read_text())

    by_id = {sample["id"]: sample for sample in payload["samples"]}
    assert by_id["s0r0"]["wet_audio"] == "audio/source0_receiver0_wet.wav"
    assert by_id["s0r1"]["wet_audio"] is None


def test_the_page_keeps_the_caveats_the_report_carried(tmp_path: Path) -> None:
    """The viewer must not be the place the honesty gets dropped."""
    run = _write_run(tmp_path)

    build_site(run, tmp_path / "site")
    payload = json.loads((tmp_path / "site" / "run.json").read_text())

    assert "not a pair of ears" in payload["binaural_note"]
    assert payload["omissions"] == ["no air absorption"]
    assert payload["theory_shell_only"]["sabine_rt60_s"] == 0.6
    assert payload["dry_voice"]["licence"] == "CC BY-NC 4.0"


def _spectrogram_image(spec: dict[str, Any]) -> np.ndarray:
    import base64

    raw = np.frombuffer(base64.b64decode(spec["data"]), dtype=np.uint8)
    return raw.reshape(spec["bins"], spec["frames"])


def test_spectrogram_puts_a_tone_in_the_row_that_tone_belongs_in() -> None:
    """The whole point of the picture is that frequency maps to height."""
    rate = 8000.0
    time = np.arange(int(rate)) / rate
    tone = np.sin(2 * np.pi * 1000.0 * time)

    image = _spectrogram_image(spectrogram(tone, rate))

    assert image.shape == (SPECTROGRAM_BINS, SPECTROGRAM_FRAMES)
    # Row index is linear in frequency from 0 to Nyquist, so 1 kHz of 4 kHz
    # sits a quarter of the way up.
    loudest = int(np.argmax(image.mean(axis=1)))
    assert loudest == pytest.approx(SPECTROGRAM_BINS / 4, abs=2)


def test_spectrogram_shows_a_decay_as_a_fade_from_left_to_right() -> None:
    rate = 8000.0
    samples = int(rate)
    rng = np.random.default_rng(0)
    decay = rng.normal(size=samples) * np.exp(-6.9078 * np.arange(samples) / rate / 0.2)

    image = _spectrogram_image(spectrogram(decay, rate))

    columns = image.mean(axis=0)
    assert columns[0] > columns[-1]


def test_spectrogram_of_silence_is_the_floor_and_not_a_divide_by_zero() -> None:
    spec = spectrogram(np.zeros(4000), 8000.0)

    assert not _spectrogram_image(spec).any()


def test_spectrogram_pads_a_response_shorter_than_one_window() -> None:
    """A short response must still produce an image of the stated size."""
    spec = spectrogram(np.array([1.0, 0.0, -1.0]), 8000.0)

    assert _spectrogram_image(spec).shape == (SPECTROGRAM_BINS, SPECTROGRAM_FRAMES)


def test_every_sample_carries_a_spectrogram(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    build_site(run, tmp_path / "site")
    payload = json.loads((tmp_path / "site" / "run.json").read_text())

    for sample in payload["samples"]:
        assert sample["spectrogram"]["bins"] == SPECTROGRAM_BINS
        # fmax is 4 kHz in the fixture, so the axis stops half an octave above
        # it rather than at the 4 kHz Nyquist of the delivery rate.
        assert 4000.0 <= sample["spectrogram"]["max_hz"] <= 6000.0
        assert _spectrogram_image(sample["spectrogram"]).shape == (
            SPECTROGRAM_BINS,
            SPECTROGRAM_FRAMES,
        )


def test_the_spectrogram_axis_stops_above_the_simulated_band_not_at_nyquist() -> None:
    """A 48 kHz file of a 4 kHz simulation is five sixths empty at Nyquist."""
    full = spectrogram(np.zeros(4096), 48_000.0)
    bounded = spectrogram(np.zeros(4096), 48_000.0, max_hz=6000.0)

    assert full["max_hz"] == pytest.approx(24_000.0, rel=0.01)
    assert bounded["max_hz"] <= 6000.0
    assert _spectrogram_image(bounded).shape == _spectrogram_image(full).shape


def test_a_ceiling_above_nyquist_is_clamped_rather_than_inventing_bandwidth() -> None:
    spec = spectrogram(np.zeros(4096), 8000.0, max_hz=99_000.0)

    assert spec["max_hz"] == pytest.approx(4000.0, rel=0.01)


def _web(name: str) -> str:
    return (Path(run_view.__file__).parent / "web" / name).read_text()


def _page() -> str:
    return _web("index.html")


def test_a_run_knows_which_apartment_it_was_simulated_in(tmp_path: Path) -> None:
    """The link that lets a run be a mode of an apartment rather than an island."""
    run = _write_run(tmp_path)

    reference = run_view.run_scene(run)

    assert reference.scene_id == "102344022"
    assert reference.room == "bedroom.001"
    assert reference.name == "run"


def test_the_payload_carries_the_apartment_so_the_panel_can_name_it(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    build_site(run, tmp_path / "site")
    payload = json.loads((tmp_path / "site" / "run.json").read_text())

    assert payload["scene_id"] == "102344022"
    assert payload["room_name"] == "bedroom.001"
    # The acoustic figures keep the key they already had.
    assert "volume_m3" in payload["room"]


def test_discovery_skips_a_run_that_was_never_rendered(tmp_path: Path) -> None:
    """A half-finished directory must not be offered and then fail to open."""
    root = tmp_path / "runs"
    complete = _write_run(root / "complete")
    (root / "complete" / "run").rename(root / "complete_run")
    (root / "started").mkdir(parents=True)
    (root / "started" / "plan.json").write_text(json.dumps({"scene_id": "1", "room": "r"}))

    found = run_view.discover_runs(root)

    assert [r.name for r in found] == ["complete_run"]
    assert complete.parent.exists()


def test_discovery_of_a_missing_directory_is_empty_rather_than_an_error() -> None:
    assert run_view.discover_runs(Path("/nonexistent/runs")) == []


def test_the_solver_button_is_released_before_the_apartment_is_fetched() -> None:
    """Assembly takes seconds, and a stale button offers the wrong room.

    Switching apartments awaits a manifest the server builds on demand. If the
    run were only pointed at the incoming apartment after that wait, the Solver
    button would keep offering the outgoing apartment's run and open it over a
    room it was never simulated in.
    """
    page = _page()

    release = page.index("releaseRun(sceneId);")
    fetch = page.index("await fetch(`scenes/${sceneId}/manifest.json`)")

    assert release < fetch


def test_one_walkable_outline_governs_every_mode() -> None:
    """The solver mesh is the apartment's own bedroom, cropped and not moved.

    Measured: the exported apartment spans (-21.07, -0.12, -13.11) to (2.51,
    2.86, 5.33) and the bedroom the solver read spans (-2.46, -0.02, -2.41) to
    (1.92, 2.80, 0.74), which is inside it. An earlier version fenced each mode
    with its own region on the belief that the frames differed; they do not, and
    two regions made walking jump at a mode switch.
    """
    page = _page()

    assert 'if (activeMode === "acoustic") {' not in page
    outline = page.index("if (!manifest) {")
    assert page.index("runView.bounds", outline) < page.index("outlineContains(x, z)", outline)


def test_switching_mode_leaves_the_camera_where_the_viewer_put_it() -> None:
    """Three representations of one room, so a switch is not a move."""
    page = _page()

    assert "savedPosition" not in page
    body = page[page.index("async function setMode(") : page.index("function animate(")]
    assert "camera.position" not in body


def test_each_simulated_room_leads_the_selector_and_opens_on_its_run() -> None:
    """A room with a run is the only entry where all three modes have something.

    It also opens without waiting: the apartment assembly is started but not
    awaited, so the run is on screen while the rest of the flat is still being
    built.
    """
    page = _page()

    options = page.index("select.innerHTML = [")
    assert page.index("runs.map(", options) < page.index("apartments.map(", options)
    body = page[page.index("async function openSelection(") :]
    assert body.index("const assembling = loadApartment(") < body.index('await setMode("acoustic")')
    assert body.index('await setMode("acoustic")') < body.index("await assembling;")


def test_discover_runs_accepts_a_single_run_directory(tmp_path: Path) -> None:
    """Pointing --runs at one run is the easy mistake, and the alternative is an
    empty selector that explains nothing."""
    root = tmp_path / "runs"
    _write_run(root / "w20")

    direct = run_view.discover_runs(root / "w20" / "run")
    parent = run_view.discover_runs(root / "w20")

    assert [r.name for r in direct] == [r.name for r in parent] == ["run"]


def test_the_deliberate_mode_is_recorded_before_the_first_await() -> None:
    """A warm apartment cache finishes inside setMode's awaits. Assigning
    ``chosen`` at the end let that arrival read it as unset and pull the view
    back to colour, so opening on the simulated room worked only on a cold
    cache. Intent is known before the work, so it is recorded before it.
    """
    body = _page()
    head = body.index("async function setMode(mode)")
    assert body.index("chosen = mode;", head) < body.index("await buildMode(mode)", head)
    assert body.count("chosen = mode;") == 1


# The viewer's behaviour lives in a browser and this suite has none: it must stay
# offline and finish in under a minute. What is pinned cheaply is that the line
# each past defect was fixed by is still there, or that the construct each defect
# came from is still gone. These are regression pins, not behavioural tests. The
# reasoning behind each fix is at the fix itself and in ADR 0006; here a row only
# has to name the guarantee so a failure is readable.
PAGE_MUST_HAVE = [
    ("position: fixed; top: 0; right: 0; bottom: 0;", "panel overlays, never in the layout"),
    ("overflow-y: auto;", "the panel scrolls, the document does not"),
    ("renderer.setSize(innerWidth, innerHeight);", "sized from the window, so no resize loop"),
    ("requestAnimationFrame(() => runShow && runShow(0));", "plots drawn once the panel is wide"),
    ('if (mode === "acoustic" && !apartmentRuns.length) return;', "no run, nothing to draw"),
    ('const MODES = [...SCENE_MODES, "acoustic"];', "one selector governs every mode"),
    ('id="btn-acoustic"', "the run is reached from the acoustic button"),
    ('from "./solver_mode.js"', "the run panel belongs to the run's module"),
    ('if (!manifest && mode !== "acoustic") return;', "the run does not wait on the assembly"),
    ("for (const mode of SCENE_MODES) groups[mode] = null;", "assembly invalidates only its modes"),
    ('if (activeMode !== "acoustic") activeMode = null;', "assembly does not evict a shown run"),
    ("if (chosen === null) {", "assembly claims the view only if unclaimed"),
    ("setBusy(assemblyMessage);", "a mode switch keeps the build message up"),
    ('? "loading solver run…" : assemblyMessage', "the run's wait is named without losing it"),
    ('if (button.disabled) button.classList.remove("active");', "no highlight on a dead button"),
]
PAGE_MUST_NOT_HAVE = [
    ("ResizeObserver", "measuring, resizing and remeasuring ran to 26 million pixels"),
    ('id="btn-solver"', "the separate solver button was folded into acoustic"),
]


@pytest.mark.parametrize(("needle", "why"), PAGE_MUST_HAVE)
def test_the_page_still_carries_its_regression_fixes(needle: str, why: str) -> None:
    assert needle in _page(), why


@pytest.mark.parametrize(("needle", "why"), PAGE_MUST_NOT_HAVE)
def test_the_page_stays_clear_of_what_caused_a_defect(needle: str, why: str) -> None:
    assert needle not in _page(), why


def test_the_run_module_owns_the_panel_and_the_geometry() -> None:
    module = _web("solver_mode.js")

    assert "renderRunPanel" in module
    assert "buildRunGroup" in module
