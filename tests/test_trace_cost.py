"""What the trace's stages cost, and that cutting it moved nothing.

Each saving of ``docs/adr/0016-appendix-trace-cost.md`` has its identity here: the distance
fields solved several to a call are the fields solved one at a time; the
chain pulled in one pass keeps the corners the scalar walk kept; the
levelling's channel 0 is the whole response's; a scene's key and upload are
read once. Then the options: what the ``check`` stage does and what its
record proves, what stays on the machine, and the estimate's constants with
the card they were measured on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.signal import butter, sosfreqz
from scipy.sparse.csgraph import dijkstra

from reverberate.audio import Atmosphere
from reverberate.compute import array_module_name
from reverberate.mirror import moving_onset
from reverberate.mirror import tails as tails_module
from reverberate.mirror.hybrid import Crossover
from reverberate.mirror.moving_onset import onset_field
from reverberate.mirror.occupancy import Occupancy, _pull, _sees_many
from reverberate.mirror.parameters import apply_parameters
from reverberate.mirror.render import LOWCUT_HZ, LOWCUT_ORDER, _lowcut_response
from reverberate.mirror.tails import TailCache, histograms, tail_key
from reverberate.render.pack import read_pack
from reverberate.trace.bundle import build_bundle
from reverberate.trace.level import pair_low, pair_omni
from reverberate.trace.plan import MEASURED_ON, Profile, estimate, make_plan
from reverberate.trace.run import CHECK_FULL, CHECK_READ, Trace, render_check
from test_mirror_diffract import walled_box
from test_trace import assets, moving_recipe, resting_recipe, traced

# --------------------------------------------------------------------------
# the onsets: fields by the call, the chain in one pass
# --------------------------------------------------------------------------


def test_distance_fields_solved_together_are_those_solved_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = walled_box()
    points = np.array([[0.6, 1.7, 0.5], [3.2, 1.7, 1.0], [1.3, 1.7, 0.9], [2.4, 1.2, 1.6]])
    held = onset_field(scene, points, sound_speed_m_s=343.2)
    starts = [int(s) for s in held.occupancy.flat(held.occupancy.cell_of(points))]
    calls: list[int] = []
    real = dijkstra

    def counted(graph: Any, **given: Any) -> Any:
        calls.append(len(np.atleast_1d(given["indices"])))
        return real(graph, **given)

    monkeypatch.setattr(moving_onset, "dijkstra", counted)
    held.solve(starts)
    # One call for the four, and the fields are each start's own search.
    assert calls == [len(set(starts))] and held.solved == len(set(starts))
    for start in starts:
        predecessor = held.root(start)
        alone = dijkstra(held.graph, directed=False, indices=start, return_predecessors=True)
        assert np.array_equal(predecessor, alone[1])
        # A field is kept as its predecessors: a cell is reached when it has one, or is the start.
        reached = predecessor >= 0
        reached[start] = True
        assert np.array_equal(reached, np.isfinite(alone[0]))
    # Held: asking again solves nothing; what was just asked for is never dropped.
    held.solve(starts)
    assert calls == [len(set(starts))]
    monkeypatch.setattr(moving_onset, "ROOTS_KEPT", 2)
    monkeypatch.setattr(moving_onset, "ROOTS_PER_CALL", 3)
    fresh = onset_field(scene, points, sound_speed_m_s=343.2)
    calls.clear()
    fresh.solve(starts)
    assert calls == [3, 1] and all(start in fresh._roots for start in starts)
    fresh.solve(starts[:1])
    assert len(fresh._roots) == 2 and starts[0] in fresh._roots


def scalar_pull(occupancy: Occupancy, chain: np.ndarray) -> list[np.ndarray]:
    """The walk :func:`_pull` replaced: one segment tested at a time, from the far end."""
    corners = [chain[0]]
    at = 0
    last = len(chain) - 1
    while at < last:
        step = last
        while step > at + 1 and not occupancy.sees(chain[at], chain[step]):
            step -= 1
        corners.append(chain[step])
        at = step
    return corners


def test_the_chain_pulled_in_one_pass_keeps_the_corners_of_the_scalar_walk() -> None:
    rng = np.random.default_rng(3)
    occupancy = Occupancy(rng.random((40, 20, 40)) < 0.12, np.array([-1.0, 0.0, -2.0]), 0.1)
    for trial in range(40):
        count = int(rng.integers(2, 150))
        chain = np.cumsum(rng.normal(0, 0.08, (count, 3)), axis=0)
        chain += rng.uniform([0, 0.5, -1], [2, 1.5, 1])
        if trial % 3 == 0:
            # Lengths that are whole numbers of samples: where a count could round apart.
            chain = np.round(chain / 0.04) * 0.04
        ends = chain[1:]
        seen = _sees_many(occupancy, chain[0], ends)
        assert np.array_equal(seen, [occupancy.sees(chain[0], end) for end in ends])
        pulled, walked = _pull(occupancy, chain), scalar_pull(occupancy, chain)
        assert len(pulled) == len(walked)
        assert all(np.array_equal(a, b) for a, b in zip(pulled, walked, strict=True))


# --------------------------------------------------------------------------
# the rays and the levelling: nothing identical computed twice
# --------------------------------------------------------------------------


def test_a_scene_is_keyed_once_for_all_its_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    held = assets(rays=16)
    cells = np.array([[1.3, 1.7, 0.9], [1.3, 1.7, 1.2]])
    sites = np.array([[0.6, 1.7, 0.5], [0.6, 1.7, 0.9]])
    applied: list[int] = []
    real = apply_parameters

    def counted(*given: Any, **named: Any) -> Any:
        applied.append(1)
        return real(*given, **named)

    monkeypatch.setattr(tails_module, "apply_parameters", counted)
    cache = TailCache()
    from reverberate.compute import Devices

    first = histograms(
        held.catalogue, held.settings, sites, cells, devices=Devices.host(1), cache=cache
    )
    again = histograms(
        held.catalogue, held.settings, sites, cells, devices=Devices.host(1), cache=cache
    )
    # The scene with its materials and its key are made once a cache, not once a call or a site.
    assert len(applied) == 1 and (cache.misses, cache.hits) == (2, 2)
    assert all(a is b for a, b in zip(first, again, strict=True))
    scene = cache._shared["scene"]
    rays = held.settings.traced_rays()
    one = cells[0]
    assert tail_key(scene, sites[0], one, rays) == tail_key(scene.key, sites[0], one, rays)
    assert tail_key(scene, sites[0], one, rays) != tail_key(scene, sites[1], one, rays)


def test_the_levelling_reads_channel_0_and_gets_the_whole_responses_answers() -> None:
    rng = np.random.default_rng(11)
    decay = np.exp(-np.arange(4800) / 900.0)
    cached = (rng.standard_normal((64, 4800)) * decay).astype(np.float32)
    cached[:, :40] = 0.0
    atmosphere = Atmosphere()
    for lead_s in (0.0, 0.0106744):
        _, onset, aired = pair_low(
            cached, Crossover(), atmosphere, sound_speed_m_s=343.2, lead_s=lead_s, unit_at_1m=0.08
        )
        onset_0, aired_0 = pair_omni(cached[:1], atmosphere, sound_speed_m_s=343.2, lead_s=lead_s)
        assert onset_0 == onset and np.array_equal(aired_0, aired)


def test_the_low_cut_is_read_once_a_transform() -> None:
    sos = butter(LOWCUT_ORDER, LOWCUT_HZ, btype="high", fs=48000.0, output="sos")
    _, direct = sosfreqz(sos, worN=np.fft.rfftfreq(4096, 1.0 / 48000.0), fs=48000.0)
    assert np.array_equal(_lowcut_response(4096, 48000.0), direct)
    assert _lowcut_response(4096, 48000.0) is _lowcut_response(4096, 48000.0)


# --------------------------------------------------------------------------
# the check: an option, and a record that says which device computed
# --------------------------------------------------------------------------


def test_the_whole_scene_reads_its_pack_and_a_smoke_run_renders_it(tmp_path: Path) -> None:
    trace, _, _ = traced(tmp_path / "whole", resting_recipe(), rays=16)
    # ``traced`` asks for the full check; left to itself the whole scene reads.
    left = Trace(
        tmp_path / "whole" / "bundle",
        tmp_path / "whole" / "out",
        engine=trace.engine,
        gpu=False,
        quiet=True,
    )
    assert left.check_mode == CHECK_READ and trace.check_mode == CHECK_FULL
    report = left.run()
    assert report["check"] == CHECK_READ and "render" not in report
    assert report["read_back"] == {"sources": 1, "steps": 11, "cells": 1, "deep": False}
    # The stages' parts are timed: what the ledger's units are read from.
    assert set(report["seconds"]["paths"]) >= {"prepare", "onset_field", "trace"}
    assert set(report["seconds"]["level"]) >= {"tails", "pairs"}
    assert "low_ir" in report["seconds"]["write"] and "read_back" in report["seconds"]["check"]
    # Nothing is traced twice, and the field of the onsets is not built for cached tables.
    again = Trace(
        tmp_path / "whole" / "bundle",
        tmp_path / "whole" / "out",
        engine=trace.engine,
        gpu=False,
        quiet=True,
    )
    resumed = again.run()
    assert "onset_field" not in resumed["seconds"]["paths"] and resumed["distance_fields"] == 0
    # A smoke run checks in full unless its bundle says otherwise; the bundle's word is kept.
    held = assets(16)
    recipe = resting_recipe()
    smoke = make_plan(recipe, held.triangles, Profile(seconds=0.3, sources=1))
    build_bundle(tmp_path / "smoke", recipe, held, smoke, allow_asset_mismatch=True)
    build_bundle(tmp_path / "told", recipe, held, smoke, allow_asset_mismatch=True, check="read")
    assert "check" not in json.loads((tmp_path / "smoke" / "campaign.json").read_text())["trace"]
    assert json.loads((tmp_path / "told" / "campaign.json").read_text())["trace"]["check"] == "read"
    modes = [
        Trace(tmp_path / name, tmp_path / "o" / name, engine=trace.engine, gpu=False, quiet=True)
        for name in ("smoke", "told")
    ]
    assert [m.check_mode for m in modes] == [CHECK_FULL, CHECK_READ]
    with pytest.raises(ValueError, match="checks"):
        Trace(tmp_path / "smoke", tmp_path / "x", engine=trace.engine, gpu=False, check_mode="no")


def test_the_render_check_says_which_module_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace, _, _ = traced(tmp_path, resting_recipe(), rays=16)
    report = trace.run()
    render = report["render"]
    assert render["card"] is False and render["finite"] and render["peak"] > 0.0
    assert (render["host_module"], render["host_arrays"]) == ("numpy", "numpy")
    assert "passed" not in render and "card_module" not in render
    assert array_module_name(np.zeros(2)) == "numpy"
    with read_pack(tmp_path / "out" / "pack.h5") as pack:
        assert pack.header.provenance["timings_s"]["level"] >= 0.0
    # A card that was promised and is not there is an error, not a host render called a pass.
    monkeypatch.setenv("REVERBERATE_NO_GPU", "0")
    import reverberate.compute as compute

    def no_card(gpu: bool | None = None) -> Any:
        if gpu:
            raise RuntimeError("a GPU was asked for and cupy sees no CUDA device")
        return np

    monkeypatch.setattr(compute, "xp_for", no_card)
    with pytest.raises(RuntimeError, match="GPU was asked"):
        render_check(tmp_path / "out" / "pack.h5", seconds=0.1, card=True)


# --------------------------------------------------------------------------
# what comes home
# --------------------------------------------------------------------------


def test_the_pair_cache_stays_on_the_machine_when_told_and_samples_are_not_deflated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reverberate.gpu import onebox
    from reverberate.wave import remote_voxelise
    from reverberate.wave.remote import Machine

    present = ["pack.h5", "pairs", "trace_report.json", "level.jsonl", "tails", "early"]
    assert onebox.fetch_items(present) == ["pairs", "pack.h5", "trace_report.json", "level.jsonl"]
    assert "pairs" not in onebox.fetch_items(present, leave=("pairs",))
    sent: list[tuple[list[str], bool]] = []

    def copied(machine: Any, sources: list[str], destination: str, **given: Any) -> None:
        sent.append(([Path(s).name for s in sources], bool(given.get("compress", True))))

    def chunked(machine: Any, remote: str, local: Path, **given: Any) -> dict[str, Any]:
        sent.append(([Path(remote).name], False))
        return {"bytes": 1, "seconds": 1.0, "bytes_per_s": 1.0, "failures": 0}

    monkeypatch.setattr(onebox, "rsync", copied)
    monkeypatch.setattr(onebox, "fetch_file", chunked)
    monkeypatch.setattr(onebox, "run_on", lambda *a, **k: "\n".join(present))
    said: list[str] = []
    onebox.fetch(None, tmp_path, tmp_path, fetch_cache=False, say=said.append, leave=("pairs",))
    # The pack first, as it is and in chunks; then the reports, deflated.
    assert sent == [(["pack.h5"], False), (["trace_report.json", "level.jsonl"], True)]
    assert said == ["left on the machine: pairs"]
    argv: list[list[str]] = []
    monkeypatch.setattr(remote_voxelise, "_run", lambda command, what: argv.append(command))
    machine = Machine(host="h", port=22, user="root", identity=None)
    remote_voxelise.rsync(machine, ["a"], "b", download=True, compress=False)
    remote_voxelise.rsync(machine, ["a"], "b", download=True)
    assert argv[0][1] == "-a" and argv[1][1] == "-az"


def test_a_smoke_run_leaves_its_pairs_and_the_scene_brings_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reverberate.experiments.run as run_module
    from reverberate.gpu import onebox
    from reverberate.scenes import Recipe
    from reverberate.trace.driver import launch

    class Spec:
        key, ppw, nh = "key1500", 10.5, 4

    monkeypatch.setattr(run_module, "scene_spec", lambda models, scene, fmax: (Spec(), None, 0))
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    export = tmp_path / "earlier" / "storey"
    export.mkdir(parents=True)
    (export / "manifest.json").write_text("{}")
    (export / "apartment_full.json").write_text("{}")
    (export.parent / "materials").mkdir()
    tree = moving_recipe().to_dict()
    tree["dwelling"]["scene_id"] = "104862621_172226772"
    recipe = Recipe.from_dict(tree)
    seen: list[dict[str, Any]] = []

    def rented(bundle: Path, where: Path, **given: Any) -> dict[str, Any]:
        told = json.loads((bundle / "campaign.json").read_text())["trace"]
        seen.append({**given, "check": told.get("check")})
        return {}

    monkeypatch.setattr(onebox, "run", rented)
    common: dict[str, Any] = {
        "models_from": export,
        "allow_asset_mismatch": True,
        "yes": True,
        "say": lambda _: None,
    }
    launch(recipe, assets(), tmp_path / "a", profile=Profile(seconds=0.3, sources=2), **common)
    launch(recipe, assets(), tmp_path / "b", **common)
    launch(recipe, assets(), tmp_path / "c", fetch_pairs=False, check="full", **common)
    launch(recipe, assets(), tmp_path / "d", fetch_pairs=False, publish_pairs=True, **common)
    assert [tuple(given["leave"]) for given in seen] == [("pairs",), (), ("pairs",), ()]
    assert [given["check"] for given in seen] == [None, None, "full", None]


# --------------------------------------------------------------------------
# the estimate
# --------------------------------------------------------------------------


def test_the_estimate_names_the_card_of_each_constant_and_what_is_still_projected() -> None:
    held = assets()
    plan = make_plan(moving_recipe(), held.triangles)
    priced = estimate(plan, rate_usd_per_hour=0.174)
    seconds = priced["seconds"]
    assert set(seconds) >= {"fixed", "low", "paths", "rays", "level", "write", "check"}
    assert {"transfer_pack", "transfer_pairs"} <= set(seconds)
    # Every stage is either measured, with its card, or projected: never unsaid.
    assert set(priced["measured"]) | set(priced["projected"]) == set(seconds)
    assert not set(priced["measured"]) & set(priced["projected"])
    assert all(priced["measured_on_by_stage"][name] for name in priced["measured"])
    assert priced["measured_on_by_stage"]["rays"] == MEASURED_ON["rays"]
    assert priced["total_s"] == pytest.approx(sum(seconds.values()), abs=1.0)
    assert priced["usd"]["rays"] == pytest.approx(seconds["rays"] / 3600.0 * 0.174, abs=1e-3)
    # The pairs brought home are a line of their own, and leaving them takes it off the total.
    without = estimate(plan, rate_usd_per_hour=0.174, fetch_pairs=False)
    assert without["seconds"]["transfer_pairs"] == 0.0
    saved = priced["total_s"] - without["total_s"]
    assert saved == pytest.approx(seconds["transfer_pairs"], abs=1.0)
    # The check is the whole scene's read unless a full one is asked for.
    full = estimate(plan, rate_usd_per_hour=0.174, check="full")
    assert full["seconds"]["check"] > seconds["check"]
    assert priced["non_solve_s"] == pytest.approx(priced["total_s"] - seconds["low"], abs=1.0)


# --------------------------------------------------------------------------
# the host's cores
# --------------------------------------------------------------------------


def test_the_cores_counted_are_the_containers_share_and_not_the_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reverberate.compute as compute
    from reverberate.render.early import workers_of

    monkeypatch.setattr("os.sched_getaffinity", lambda _: set(range(72)), raising=False)
    quota, period = tmp_path / "quota", tmp_path / "period"
    monkeypatch.setattr(compute, "_CGROUP_MAX", str(tmp_path / "none"))
    monkeypatch.setattr(compute, "_CGROUP_QUOTA", str(quota))
    monkeypatch.setattr(compute, "_CGROUP_PERIOD", str(period))
    # No quota written: the affinity. Then the first card box's: 72 cores seen, 13.8 given.
    assert compute.usable_cores() == 72
    quota.write_text("1382400\n")
    period.write_text("100000\n")
    assert compute.usable_cores() == 13
    quota.write_text("-1\n")
    assert compute.usable_cores() == 72
    # The unified hierarchy says both on one line, or that there is no limit.
    unified = tmp_path / "cpu.max"
    monkeypatch.setattr(compute, "_CGROUP_MAX", str(unified))
    unified.write_text("250000 100000\n")
    assert compute.usable_cores() == 2
    unified.write_text("max 100000\n")
    assert compute.usable_cores() == 72
    unified.write_text("50000 100000\n")
    assert compute.usable_cores() == 1
    # "Every core" of the engine's transforms is that count; a card takes no such word.
    unified.write_text("400000 100000\n")
    monkeypatch.setattr("reverberate.render.early.usable_cores", compute.usable_cores)
    assert workers_of(np, -1) == {"workers": 4} and workers_of(np, 3) == {"workers": 3}
    assert workers_of(object(), -1) == {}


# --------------------------------------------------------------------------
# the grid a trace takes unless told
# --------------------------------------------------------------------------


def test_a_trace_is_asked_at_7_2_points_unless_told_and_10_5_is_the_reference() -> None:
    """The command line and a recipe's cost name the grid; a record that names none is 10.5."""
    from reverberate.scenes import cost
    from reverberate.trace import machines
    from reverberate.trace.cli import parse
    from reverberate.trace.plan import LOW_PPW, REFERENCE_PPW

    assert (LOW_PPW, REFERENCE_PPW) == (7.2, 10.5)
    words = ["bundle", "--recipe", "R.json", "--out", "B"]
    assert parse(words).low_ppw == 7.2
    assert parse([*words, "--low-ppw", "10.5"]).low_ppw == 10.5
    # PFFDTD has the bundle's grid and no other.
    assert parse([*words, "--low-engine", "pffdtd"]).low_ppw is None
    record = {
        "source_positions": 7,
        "pairs": 559,
        "cells_a_position": [80] * 6 + [79],
        "tail_sites": 7,
        "tail_cells": 12,
        "step_pairs": 9000,
        "pairs_of_the_patch": 0,
        "profile": {},
    }
    one = {"gpu_name": "RTX 3090", "num_gpus": 1, "gpu_ram_gb": 24.0, "dph_total": 0.173}
    coarse = machines.predict(record, low_ppw=7.2, **one)
    fine = machines.predict(record, low_ppw=10.5, **one)
    assert coarse is not None and fine is not None
    priced = cost.predict(record, num_gpus=1, dph_total=0.173)
    assert priced["seconds"] == coarse["seconds"]
    assert cost.predict(record, num_gpus=1, dph_total=0.173, low_ppw=10.5)["seconds"] == (
        fine["seconds"]
    )
    # A record that names no grid is the reference's, as every run's before 2026-10-07.
    unnamed = machines.predict(record, **one)
    assert unnamed is not None and unnamed["seconds"] == fine["seconds"]
    # Half the card seconds a source position, and a longer preparation.
    assert coarse["work"]["solve_card_s"] < 0.55 * fine["work"]["solve_card_s"]
    assert coarse["seconds"]["prepare"] == 240.0 and fine["seconds"]["prepare"] == 110.0
