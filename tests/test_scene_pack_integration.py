"""The three halves of a moving scene meet in one pack: what one writes, the other reads.

The moving mirror (``mirror.moving``, ``mirror.tails``, ``mirror.directivity``)
makes the tables above the crossover, the low band library
(``spatial.translate``, ``spatial.lowband``) chooses the cells and stores the
responses under it, and the signal engine (``reverberate.render``) owns the
format. Each was written against the format's document alone. Here a small
room is traced, written through :class:`PackWriter`, read back and rendered,
and the things two of them define are shown to be one thing: a path's
identity, the bands, the tail's model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.compute import Devices
from reverberate.metrics import band_centres, octave_bank, octave_filter_rows
from reverberate.mirror import moving
from reverberate.mirror.directivity import Directivity, directivity_gain, omni, voice_v1
from reverberate.mirror.hybrid import Crossover
from reverberate.mirror.moving import prepare, trace_early
from reverberate.mirror.moving_onset import onset_field
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import RaySettings
from reverberate.mirror.render import RenderSettings as MirrorRender
from reverberate.mirror.render import _band_map, tail_from_histogram
from reverberate.mirror.tails import TailCache, histograms, tail_sites, tail_table
from reverberate.render import pack as pack_format
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.pack import (
    KIND_DIFFRACTED,
    KIND_DIRECT,
    KIND_SPECULAR,
    Cells,
    Early,
    Header,
    Level,
    Listener,
    Low,
    Mirror,
    PackWriter,
    Source,
    Tail,
    band_map,
    default_fusion,
    path_id,
    read_pack,
    synthetic_recipe,
    tail_seed,
)
from reverberate.spatial import translate
from reverberate.spatial.field import monopole_coefficients
from reverberate.spatial.lowband import LOW_RATE_HZ, LOW_SAMPLES, onset_s, pair_key, to_stored
from reverberate.spatial.sh import degrees_of, scene_to_ambisonic
from reverberate.spatial.translate import (
    MODE_FUSED,
    MODE_INAUDIBLE,
    choose_cells,
    clearance_m,
    serving_radius_m,
)
from test_mirror_diffract import walled_box

C = 343.2
FS = 48000
STEP_S = 0.05
SETTINGS = MirrorSettings(
    rays=RaySettings(rays=120, duration_s=0.12, bin_s=0.002, receiver_radius_m=0.3)
)
#: A rail the source walks, solved every 8 cm, and the cells the head walks between.
RAIL_START = np.array([0.5, 1.2, 0.5])
RAIL_PITCH_M = 0.08
CELLS = np.array([[1.10 + 0.15 * i, 1.2, 1.0] for i in range(5)])


def test_the_trace_names_its_paths_as_the_format_does() -> None:
    assert moving.path_id is pack_format.path_id
    assert (moving.KIND_DIRECT, moving.KIND_SPECULAR) == (KIND_DIRECT, KIND_SPECULAR)
    catalogue = walled_box()
    ms = prepare(catalogue, SETTINGS)
    # One listener in view of the source and one behind the wall, whose onset is diffracted.
    source = np.repeat(np.array([[1.0, 1.2, 0.8]]), 2, axis=0)
    listener = np.array([[1.5, 1.4, 1.8], [3.0, 1.2, 0.8]])
    held = onset_field(catalogue, np.concatenate([source, listener]), sound_speed_m_s=C)
    table = trace_early(ms, source, listener, onsets=held)
    assert set(table.kind.tolist()) >= {KIND_DIRECT, KIND_SPECULAR, KIND_DIFFRACTED}
    for identity, kind, order, sequence in zip(
        table.path_id, table.kind, table.order, table.sequence, strict=True
    ):
        if kind in (KIND_DIRECT, KIND_SPECULAR):
            assert int(identity) == path_id(int(kind), sequence[:order])
    # The format's bytes: the kind, the facets, and for a diffracted path -1 then its edges.
    words = np.asarray([3, -1, 5], dtype="<i4").tobytes()
    named = int.from_bytes(hashlib.sha256(bytes([2]) + words).digest()[:8], "little")
    assert path_id(KIND_DIFFRACTED, (3,), (5,)) == named
    assert path_id(KIND_DIFFRACTED, (3,), (5,)) != path_id(KIND_DIFFRACTED, (3, 5), ())
    words = np.asarray([-1, -1, 2], dtype="<i4").tobytes()
    nameless = int.from_bytes(hashlib.sha256(bytes([2]) + words).digest()[:8], "little")
    assert path_id(KIND_DIFFRACTED, rank=2) == nameless
    # What the engine validates: a step's rows sorted by identity, none twice.
    packed = Early(**table.pack())
    for step in range(table.steps):
        ids = packed.path_id[packed.rows(step)]
        assert np.all(ids[1:] > ids[:-1])


def test_the_bands_and_the_directivity_are_one_definition() -> None:
    centres, picks = _band_map(48000.0)
    assert Header("trace", "", "", "", 1.0, 21).bank == tuple(centres) == band_centres(48000)
    np.testing.assert_array_equal(band_map(OCTAVE_BANDS, centres), picks)
    # The pack's directivity is the mirror's class, and the engine reads its gain.
    assert pack_format.Directivity is Directivity
    voice = voice_v1()
    assert pack_format.Directivity(**voice.pack(), name="voice_v1").digest == voice.digest
    behind = directivity_gain(voice, np.array([[-1.0, 0.0, 0.0]]), 0.0)
    assert behind.shape == (1, 7) and np.all(np.diff(behind[0]) < 0.0)


def wave_response(source: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """A solve's response in the dwelling's cache: a free monopole at 4 kHz, before its masks.

    ``[64, 4800]``, the samples of the 48 kHz response
    (:func:`reverberate.spatial.lowband.decimate`'s scale), band limited as
    a low only solve is.
    """
    freqs = np.fft.rfftfreq(LOW_SAMPLES, 1.0 / LOW_RATE_HZ)
    keep = freqs > 80.0
    k = 2.0 * np.pi * freqs[keep] / C
    seen = scene_to_ambisonic((source - cell)[None, :])[0]
    spectrum = np.zeros((freqs.size, 64), dtype=complex)
    spectrum[keep] = 4.0 * np.pi * monopole_coefficients(seen, k, 7) / (1j ** degrees_of(7))[None]
    edges = np.clip((freqs - 80.0) / 80.0, 0, 1) * np.clip((1800.0 - freqs) / 200.0, 0, 1)
    spectrum *= (0.5 - 0.5 * np.cos(np.pi * edges))[:, None]
    return np.asarray(np.fft.irfft(spectrum.T, LOW_SAMPLES, axis=-1) * (LOW_RATE_HZ / FS))


def traced_pack(target: Path, *, with_low: bool = True) -> Path:
    """A pack as the scene trace will write one: every table from the lot that owns it."""
    catalogue = walled_box()
    ms = prepare(catalogue, SETTINGS)
    steps = 11
    duration = (steps - 1) * STEP_S
    k = np.arange(steps)[:, None]
    source = RAIL_START + k * np.array([0.0, 0.0, 0.05])  # 1 m/s along the rail
    listener = np.array([1.13, 1.2, 1.0]) + k * np.array([0.03, 0.0, 0.0])
    yaw = np.full(steps, -90.0, dtype=np.float32)
    audible = np.ones(steps, dtype=bool)
    audible[-2:] = False
    crossover = Crossover()
    recipe = synthetic_recipe(duration, ["voice"])

    # Above the crossover: the moving mirror's tables, in the pack's types.
    onsets = onset_field(catalogue, np.concatenate([source, listener]), sound_speed_m_s=C)
    early = trace_early(ms, source, listener, audible=audible, onsets=onsets)
    arcs = np.arange(0.0, 0.5 + 1e-9, 0.5)
    rail = RAIL_START + arcs[:, None] * np.array([0.0, 0.0, 1.0])
    sites = tail_sites(np.zeros((0, 3)), [rail])
    tail = tail_table(
        ms, SETTINGS, source, listener, sites, CELLS, audible=audible, devices=Devices.host(1)
    )

    # Under it: the cells by the library's rule, the pairs in the library's stored form.
    clearance = clearance_m(CELLS, np.asarray(catalogue.occluder_vertices).reshape(-1, 3, 3))
    arc = np.linalg.norm(source - RAIL_START, axis=1)
    lower = np.floor(arc / RAIL_PITCH_M + 1e-9).astype(int)
    weight = arc / RAIL_PITCH_M - lower
    solved = RAIL_START + (np.arange(lower.max() + 2) * RAIL_PITCH_M)[:, None] * np.array(
        [0.0, 0.0, 1.0]
    )
    mode = np.full(steps, MODE_INAUDIBLE, dtype=np.uint8)
    cell = np.full((steps, 2), -1, dtype=np.int32)
    pair = np.full((steps, 2, 2), -1, dtype=np.int32)
    rows: dict[tuple[int, int], int] = {}
    for step in np.flatnonzero(audible):
        radius = serving_radius_m(CELLS, clearance, source[step])
        mode[step], chosen = choose_cells(listener[step], CELLS, radius)
        cell[step] = chosen
        for a in range(2 if weight[step] > 1e-9 else 1):
            for b in range(2 if mode[step] == MODE_FUSED else 1):
                pair[step, a, b] = rows.setdefault((int(lower[step]) + a, chosen[b]), len(rows))
    weight = np.where(audible & (weight > 1e-9), weight, 0.0)
    stored = [
        to_stored(wave_response(solved[j], CELLS[c]), LOW_RATE_HZ, crossover) for j, c in rows
    ]
    low = Low(
        ir=np.stack(stored),
        pair_position=np.stack([solved[j] for j, _ in rows]),
        pair_cell=np.array([c for _, c in rows], dtype=np.int32),
        pair_key=np.array(
            [
                pair_key("grid", solved[j], CELLS[c], encoder={"order": 7}, solver="test")
                for j, c in rows
            ],
            dtype="S64",
        ),
        seam_db=np.zeros(len(rows), dtype=np.float32),
        onset_s=np.array([onset_s(ir[0], LOW_RATE_HZ) for ir in stored]),
        pair=pair,
        position_weight=weight.astype(np.float32),
        cell=cell,
        mode=mode,
    )
    first = np.array(
        [early.delay_s[early.rows(s)].min() if audible[s] else 0.0 for s in range(steps)]
    )
    header = Header(
        profile="trace",
        recipe_sha256=hashlib.sha256(recipe).hexdigest(),
        dwelling="box",
        scene_id="walled-box",
        duration_s=duration,
        steps=steps,
        sound_speed_m_s=C,
        bank_bands_hz=band_centres(FS),
        has_low=with_low,
        has_tail=True,
        fusion=default_fusion(),
        provenance={"code_version": "test", "cost": []},
    )
    rays = SETTINGS.traced_rays()
    with PackWriter(
        target,
        header,
        recipe,
        Listener(listener, np.zeros((steps, 3), dtype=np.float32)),
        Cells(
            position=CELLS,
            kind=np.full(len(CELLS), 2, dtype=np.uint8),
            lattice_index=np.full((len(CELLS), 3), -1, dtype=np.int32),
            clearance_m=clearance.astype(np.float32),
            room=("room",) * len(CELLS),
            layers_y_m=(1.2,),
        ),
        mirror=Mirror(
            tail_from_s=SETTINGS.render.tail_from_s,
            tail_bursts=SETTINGS.render.tail_bursts,
            tail_gain_db=SETTINGS.parameters.tail_gain_db,
            histogram_bin_s=rays.bin_s,
            histogram_order=rays.order,
            receiver_radius_m=rays.receiver_radius_m,
        ),
        crossover=crossover,
        directivity={"voice_v1": voice_v1(), "omni": omni()},
    ) as writer:
        writer.add_source(
            Source(
                id="voice",
                kind="near_voice",
                position=source,
                yaw_deg=yaw,
                audible=audible,
                early=Early(**early.pack()),
                tail=Tail(**tail.pack()),
                low=low if with_low else None,
                level=Level(high_gain_db=np.zeros(steps, dtype=np.float32), onset_s=first),
                directivity_model="voice_v1",
                directivity_enabled=True,
                tail_seed=tail_seed(7, "voice"),
            )
        )
    return target


def test_the_mirrors_tables_and_the_low_bands_cells_are_a_pack_the_engine_renders(
    tmp_path: Path,
) -> None:
    path = traced_pack(tmp_path / "pack.h5")
    dry = np.random.default_rng(0).standard_normal(int(0.4 * FS))
    with read_pack(path, deep=True) as pack:
        source = pack.sources["voice"]
        assert source.low is not None and source.tail is not None
        # The rule of the library chose: two cells either side where both may serve, the
        # nearer alone where the source is too close for the further one, a cell itself on it.
        modes = set(source.low.mode[source.audible].tolist())
        assert MODE_FUSED in modes and MODE_INAUDIBLE not in modes
        assert pack.header.fusion["fuse_within_m"] == translate.FUSE_WITHIN_M
        assert pack.directivity["voice_v1"].digest == voice_v1().digest
        engine = Engine(pack, {"voice": dry})
        parts = {name: engine.stem("voice", parts=(name,)) for name in ("early", "low", "tail")}
        whole = engine.render()
        flat = Engine(pack, {"voice": dry}, settings=RenderSettings(directivity=False)).stem(
            "voice", parts=("early",)
        )
    assert whole.shape == (64, 10 * 2400) and np.all(np.isfinite(whole))
    for name, part in parts.items():
        assert np.all(np.isfinite(part)) and float(np.abs(part).max()) > 0.0, name
    np.testing.assert_allclose(whole, sum(parts.values()), atol=1e-12 * np.abs(whole).max())
    # The voice faces away from the head along its rail: its pattern changes the early part.
    assert float(np.abs(parts["early"] - flat).max()) > 1e-3 * float(np.abs(flat).max())


def test_the_engines_tail_is_the_mirrors_tail_band_by_band(tmp_path: Path) -> None:
    """One histogram, at rest: the two tails are different noise at the same energies."""
    catalogue = walled_box()
    ms = prepare(catalogue, SETTINGS)
    steps = 9
    station = np.array([1.0, 1.2, 0.8])
    source = np.repeat(station[None, :], steps, axis=0)
    listener = np.repeat(CELLS[2][None, :], steps, axis=0)
    early = trace_early(ms, source, listener)
    cache = TailCache()
    table = tail_table(
        ms,
        SETTINGS,
        source,
        listener,
        tail_sites(station[None, :]),
        CELLS,
        devices=Devices.host(1),
        cache=cache,
    )
    assert table.energy.shape[0] == 1 and float(table.energy.sum()) > 0.0
    recipe = synthetic_recipe((steps - 1) * STEP_S, ["voice"])
    rays = SETTINGS.traced_rays()
    held = pack_format.ScenePack(
        header=Header(
            profile="trace",
            recipe_sha256=hashlib.sha256(recipe).hexdigest(),
            dwelling="box",
            scene_id="walled-box",
            duration_s=(steps - 1) * STEP_S,
            steps=steps,
            bank_bands_hz=band_centres(FS),
            has_tail=True,
        ),
        recipe=recipe,
        listener=Listener(listener, np.zeros((steps, 3), dtype=np.float32)),
        cells=Cells(
            CELLS,
            np.zeros(len(CELLS), np.uint8),
            np.full((len(CELLS), 3), -1, np.int32),
            np.ones(len(CELLS), np.float32),
            ("room",) * len(CELLS),
        ),
        sources={
            "voice": Source(
                id="voice",
                kind="near_voice",
                position=source,
                yaw_deg=np.zeros(steps, dtype=np.float32),
                audible=np.ones(steps, dtype=bool),
                early=Early(**early.pack()),
                tail=Tail(**table.pack()),
                level=Level(np.zeros(steps, dtype=np.float32), np.zeros(steps)),
                tail_seed=tail_seed(7, "voice"),
            )
        },
        mirror=Mirror(
            tail_from_s=SETTINGS.render.tail_from_s,
            tail_bursts=SETTINGS.render.tail_bursts,
            histogram_bin_s=rays.bin_s,
            histogram_order=rays.order,
            receiver_radius_m=rays.receiver_radius_m,
        ),
    )
    pack_format.validate(held)
    click = np.zeros(int(0.05 * FS))
    click[1200] = 1.0
    mine = Engine(held, {"voice": click}).stem("voice", parts=("tail",))[0, 1200:]
    histogram = histograms(catalogue, SETTINGS, station[None, :], CELLS, cache=cache)[0]
    assert cache.misses == 1, "the mirror's histogram is the one the table was made from"
    theirs, _ = tail_from_histogram(
        histogram,
        2,
        MirrorRender(duration_s=mine.size / FS),
        sound_speed_m_s=C,
        start_s=float(np.linalg.norm(CELLS[2] - station)) / C,
        seed=3,
        bursts=SETTINGS.render.tail_bursts,
        scale_per_band=table.scale[0],
    )
    bands = octave_bank(FS).filters.shape[1]

    def read(signal: np.ndarray) -> np.ndarray:
        rows = octave_filter_rows(np.repeat(signal[None], bands, 0), FS, np.arange(bands))
        return np.asarray((rows**2).sum(axis=1))

    level = 10.0 * np.log10(read(mine) / read(np.asarray(theirs)[0]))
    # Measured: 0.56 dB at worst, two draws of noise read through a leaking bank.
    assert np.abs(level).max() < 1.0, level
