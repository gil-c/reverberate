"""The trace as the jobs of one queue: any machine, the same pack.

The queue is proved alone first, on jobs that do nothing: what waits is not
started early, a job answered in parts is waited for through its parts, a
job that fails is given to another worker, one refused its memory is split.
Then the trace of ``tests/test_trace.py``'s moving scene is run on machines
that are not there (two cards and two cores, in one process), with jobs
failed on purpose, and in real processes, and its pack must be the one a
single process writes: every dataset and every attribute but the date and
the seconds (:func:`reverberate.trace.run.pack_digest`).

The wave solver's two ways round a small card are held to the bit on
``numpy``: the records brought to the host block by block, and one solve cut
in slabs with a plane exchanged a cut a step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.mirror.moving import MovingSettings, prepare, trace_early
from reverberate.trace import run as run_module
from reverberate.trace.pool import CARD, HOST, PARENT, Job, Pool, PoolError, WorkerSpec
from reverberate.trace.resources import Machine, measure, predict
from reverberate.trace.run import Trace, _blocks, _joined, pack_digest
from reverberate.wave.lowband.problem import build_problem
from reverberate.wave.lowband.scheme import CARTESIAN, FCC, Scheme
from reverberate.wave.lowband.slabs import bounds_for, exchange_bytes, slabs_of, solve_slabbed
from reverberate.wave.lowband.solver import NumpyStepper, drive_for, solve
from test_trace import assets, moving_recipe, traced
from test_wave_lowband import HEARD_AT, a_campaign, lossy_room, machine, seeds_of

__all__ = ["machine"]

# --------------------------------------------------------------------------
# the queue alone
# --------------------------------------------------------------------------


class Told:
    """A handler that writes down what it is asked, and answers as it was told to."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.asked: list[str] = []
        self.answers = answers or {}

    def run_job(self, stage: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.asked.append(f"{stage}/{name}")
        return dict(self.answers.get(f"{stage}/{name}", {"made": name}))


def specs(cards: int, hosts: int) -> list[WorkerSpec]:
    held = [WorkerSpec(k, card=k, free_bytes=8e9) for k in range(cards)]
    return held + [WorkerSpec(cards + k) for k in range(hosts)]


def test_a_job_waits_for_what_it_reads_and_a_merge_runs_between_two_hand_outs() -> None:
    told = Told()
    merged: list[str] = []
    pool = Pool(specs(1, 2), inline=told)
    pool.add(
        Job("level", "0", after=("solve/a", "paths/0"), on=HOST),
        Job("merge", "all", on=PARENT, after=("level/0",), work=lambda: merged.append("yes")),
        Job("solve", "a", on=CARD),
        Job("paths", "0", on=HOST),
    )
    done = pool.run()
    assert told.asked.index("level/0") > max(told.asked.index(k) for k in ("solve/a", "paths/0"))
    assert merged == ["yes"] and set(done) == {"level/0", "merge/all", "solve/a", "paths/0"}
    account = pool.account()
    assert account["stages"]["solve"]["jobs"] == 1 and len(account["workers"]) == 3
    # A card's work goes to the card; the host's goes to the cores first.
    by_worker = {line["name"]: line["card"] for line in pool.ledger if line["stage"] == "solve"}
    assert by_worker == {"a": 0}


def test_a_job_answered_in_parts_is_waited_for_through_its_parts() -> None:
    parts = [{"name": "a.0", "payload": {"half": 0}}, {"name": "a.1", "payload": {"half": 1}}]
    told = Told({"solve/a": {"parted": parts}})
    pool = Pool(specs(2, 0), inline=told)
    pool.add(Job("level", "0", after=("solve/a",), on=HOST), Job("solve", "a", on=CARD))
    pool.run()
    assert told.asked[0] == "solve/a" and told.asked[-1] == "level/0"
    assert set(told.asked[1:3]) == {"solve/a.0", "solve/a.1"}
    assert pool.finished("solve/a")


def test_a_failed_job_goes_to_another_worker_and_a_refused_one_is_split() -> None:
    told = Told()

    def halves(job: Job) -> list[Job]:
        return [Job("solve", f"{job.name}.{k}", {"half": k}, on=CARD) for k in range(2)]

    pool = Pool(
        specs(2, 1),
        inline=told,
        faults={"paths/0": ("error", 1), "solve/big": ("memory", 1)},
    )
    pool.add(Job("paths", "0", on=HOST), Job("solve", "big", on=CARD, split=halves))
    pool.run()
    tried = [line for line in pool.ledger if line["name"] == "0"]
    assert len(tried) == 1 and tried[0]["tries"] == 2, "failed once, then made"
    assert told.asked.count("paths/0") == 1, "the failure a test asks for is before the work"
    assert {"solve/big.0", "solve/big.1"} <= set(told.asked) and "solve/big" not in told.asked
    # A job that fails every time ends the run with what the worker said.
    stubborn = Pool(specs(0, 2), inline=Told(), faults={"paths/0": ("error", 99)})
    stubborn.add(Job("paths", "0", on=HOST))
    with pytest.raises(PoolError, match="failed 3 times"):
        stubborn.run()
    # A job that waits for one that does not exist is said, not waited for.
    lost = Pool(specs(0, 1), inline=Told())
    lost.add(Job("level", "0", after=("solve/never",)))
    with pytest.raises(PoolError, match="no job of this pool"):
        lost.run()


def test_a_launch_too_large_for_a_card_waits_for_one_that_holds_it() -> None:
    told = Told()
    pool = Pool(
        [WorkerSpec(0, card=0, free_bytes=4e9), WorkerSpec(1, card=1, free_bytes=20e9)],
        inline=told,
    )
    pool.add(*[Job("solve", str(k), on=CARD, bytes=10e9) for k in range(4)])
    pool.run()
    assert {line["card"] for line in pool.ledger} == {1}
    # One that no card holds is given to the largest, which parts it.
    larger = Pool([WorkerSpec(0, card=0, free_bytes=4e9)], inline=Told())
    larger.add(Job("solve", "0", on=CARD, bytes=10e9))
    assert set(larger.run()) == {"solve/0"}


# --------------------------------------------------------------------------
# the machine
# --------------------------------------------------------------------------


def test_a_machine_is_a_process_a_card_and_a_process_a_core_left() -> None:
    eight = Machine.fake((24.0,) * 8, cores=14, ram_gb=120.0)
    held = eight.workers()
    assert [w.card for w in held[:8]] == list(range(8)) and len(held) == 8 + 6
    assert all(w.card is None for w in held[8:])
    assert [w.card for w in Machine.fake((), cores=4).workers()] == [None] * 3
    assert len(Machine.fake((16.0,), cores=1).workers()) == 2
    said = measure(None, np, seconds=0.05)
    assert said["updates_per_s"] > 0 and said["transform_s"] > 0
    measured = Machine.fake((), cores=4).measured([said])
    assert measured.host_updates_per_s == said["updates_per_s"]


def test_the_prediction_goes_as_the_cards_and_the_cores_it_is_given() -> None:
    counts = {
        "node_updates": 1529 * 47.4e6 * 32769,
        "pairs": 16887,
        "pair_s": 0.31,
        "sites": 202,
        "site_s": 18.2,
        "paths_jobs": 62751,
        "level_pairs": 16887,
        "row_pairs": 16887,
        "write_s": 170.0,
    }
    one = predict(counts, Machine.fake((24.0,), cores=14), host_workers=13)
    eight = predict(
        counts, Machine.fake((24.0,) * 8, cores=14), host_workers=6, rate_usd_per_hour=1.38
    )
    assert one["seconds"]["solve"] == pytest.approx(8 * eight["seconds"]["solve"], rel=1e-3)
    assert eight["hours"] < one["hours"] / 7.0, "the host's stages hide under the solves"
    assert eight["usd"] == pytest.approx(eight["hours"] * 1.38, abs=0.01)
    assert not eight["calibrated"], "until a card says what the grid costs against the box"
    # With no card the host's cores bound it, and twice the cores is half the time.
    few = predict(
        {**counts, "node_updates": 0.0, "pairs": 0}, Machine.fake((), cores=4), host_workers=4
    )
    more = predict(
        {**counts, "node_updates": 0.0, "pairs": 0}, Machine.fake((), cores=8), host_workers=8
    )
    assert few["seconds"]["host"] == pytest.approx(2 * more["seconds"]["host"], abs=0.2)


# --------------------------------------------------------------------------
# the trace: one pack, whatever computed it
# --------------------------------------------------------------------------


def test_a_table_cut_in_blocks_is_the_table_traced_whole() -> None:
    heard = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert _blocks(heard, 2) == [(0, 4), (4, 6), (6, 10)]
    assert _blocks(heard, 100) == [(0, 10)] and _blocks(heard[:0], 2) == [(0, 0)]
    assert _blocks(np.zeros(4, dtype=bool), 2) == [(0, 4)]
    held = assets()
    ms = prepare(held.catalogue, held.settings, MovingSettings())
    steps = 9
    source = np.linspace([0.6, 1.7, 0.5], [0.6, 1.7, 0.98], steps)
    head = np.linspace([1.3, 1.7, 0.9], [1.3, 1.7, 1.2], steps)
    audible = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    whole = trace_early(ms, source, head, audible=audible)
    parts = [
        trace_early(ms, source[a:b], head[a:b], audible=audible[a:b])
        for a, b in _blocks(audible, 2)
    ]
    joined = _joined(parts, source, head)
    assert whole.offsets[-1] > 0 and len(parts) == 4
    for name in run_module._EARLY_FIELDS:
        assert np.array_equal(getattr(whole, name), getattr(joined, name)), name


def quick(tmp: Path) -> tuple[Trace, Any, Any]:
    """The moving scene's trace, not yet run; its pack read back and not rendered."""
    trace, pairs, plan = traced(tmp, moving_recipe(), rays=16)
    trace.check_mode = "read"
    return trace, pairs, plan


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The moving scene's pack as one process writes it, every table one block."""
    tmp = tmp_path_factory.mktemp("reference")
    trace, _, _ = quick(tmp)
    report = trace.run()
    return {"digest": pack_digest(tmp / "out" / "pack.h5"), "report": report, "tmp": tmp}


def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks of a few steps and a few pairs: several jobs a table on a half second scene."""
    monkeypatch.setattr(run_module, "PATHS_BLOCK", 3)
    monkeypatch.setattr(run_module, "PAIRS_BLOCK", 2)


@pytest.mark.parametrize(
    ("cards", "hosts"), [((), 0), ((8.0, 16.0), 2)], ids=["one process", "2 cards and 2 cores"]
)
def test_the_pack_is_the_same_on_any_machine(
    reference: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cards: tuple[float, ...],
    hosts: int,
) -> None:
    small_blocks(monkeypatch)
    trace, _, _ = quick(tmp_path)
    trace.machine = Machine.fake(cards, cores=len(cards) + hosts + 1)
    trace.workers = hosts
    report = trace.run()
    assert pack_digest(tmp_path / "out" / "pack.h5") == reference["digest"]
    stages = report["pool"]["stages"]
    assert stages["paths"]["jobs"] > 6 and stages["level"]["jobs"] > 2
    assert {"solve", "paths", "rays", "tails", "level", "rows"} <= set(stages)
    assert len(report["pool"]["workers"]) == max(1, len(cards) + hosts)
    if cards:
        # Every worker of the machine took jobs: none waited while another stage had work.
        assert all(w["jobs"] > 0 for w in report["pool"]["workers"])
        assert report["predicted"]["cards"] == len(cards)


def test_a_job_made_again_after_a_failure_leaves_the_pack_as_it_was(
    reference: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_blocks(monkeypatch)
    trace, _, _ = quick(tmp_path)
    trace.machine = Machine.fake((8.0, 16.0), cores=4)
    trace.workers = 1
    trace.faults = {
        "paths/voice.1": ("error", 2),
        "paths/_pairs.0": ("error", 1),
        "level/1": ("memory", 1),
        "rows/0": ("error", 1),
        "solve/p000000": ("error", 1),
    }
    report = trace.run()
    assert pack_digest(tmp_path / "out" / "pack.h5") == reference["digest"]
    again = [line for line in trace.pool.ledger if line["tries"] > 1]
    assert {f"{line['stage']}/{line['name']}" for line in again} == set(trace.faults)
    assert report["level"]["made"] == report["pairs"]["scene"]


def test_the_pack_is_the_same_from_processes_of_their_own(
    reference: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_blocks(monkeypatch)
    trace, _, _ = quick(tmp_path)
    trace.machine = Machine.fake((), cores=3)
    trace.workers = 2
    trace.engine_told = {"kind": "free-field", "measure_s": 0.0}
    report = trace.run()
    assert pack_digest(tmp_path / "out" / "pack.h5") == reference["digest"]
    workers = report["pool"]["workers"]
    assert len(workers) == 2 and sum(w["jobs"] for w in workers) > 10
    assert (tmp_path / "out" / "state" / "centres.json").is_file()
    # A run stopped before the pack is taken up from the jobs' files: nothing is made twice.
    (tmp_path / "out" / "pack.h5").unlink()
    second, _, _ = quick(tmp_path)
    again = second.run()
    assert pack_digest(tmp_path / "out" / "pack.h5") == reference["digest"]
    assert again["rays"]["histograms_traced"] == 0 and again["level"]["made"] == 0
    assert all(record == {"cached": True} for record in again["paths"].values())


def test_the_scaling_is_each_counts_own_run_and_says_whether_the_packs_are_one(
    reference: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from reverberate.trace import scaling
    from reverberate.trace.cli import main

    bundle = reference["tmp"] / "bundle"
    code = main(
        ["scaling", "--bundle", str(bundle), "--out", str(tmp_path / "s"), "--free-field"]
        + ["--cpu", "--workers", "0", "--keep"]
    )
    said = capsys.readouterr().out
    assert code == 0 and "the pack is the same at every count" in said
    held = json.loads((tmp_path / "s" / "scaling.json").read_text())
    (run,) = held["runs"]
    assert held["by"] == "workers" and run["pack"] == reference["digest"]
    assert run["processes"] == 1 and run["stages"]["level"]["work_s"] > 0
    assert (tmp_path / "s" / "0" / "pool.json").is_file() is False
    assert json.loads((tmp_path / "s" / "0w" / "pool.json").read_text())["ledger"]
    # The table: the wall against the first count's over the workers, and Karp and Flatt's
    # serial fraction of what is left.
    two = {**run, "workers": 2, "work_wall_s": 0.75 * run["work_wall_s"], "pack": "other"}
    lines = scaling.table([{**run, "workers": 1}, two], "workers").splitlines()
    assert lines[2].split()[:5] == ["2", *lines[2].split()[1:3], "1.33", "0.500"]
    # Two packs that differ are said, by the command that reads what they hold.
    pack = reference["tmp"] / "out" / "pack.h5"
    assert main(["digest", str(pack), str(pack)]) == 0
    assert reference["digest"] in capsys.readouterr().out


def test_a_run_predicted_over_its_hours_stops_before_its_work(tmp_path: Path) -> None:
    trace, pairs, _ = quick(tmp_path)
    trace.max_hours = 1e-9
    with pytest.raises(RuntimeError, match="stopped before it started"):
        trace.run()
    assert pairs.solved == [] and (tmp_path / "out" / "campaign.failed").is_file()
    assert (tmp_path / "out" / "prediction.json").is_file()


# --------------------------------------------------------------------------
# the wave solver on a small card, and on several
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", [CARTESIAN, FCC], ids=lambda s: s.name)
def test_records_brought_to_the_host_block_by_block_are_the_records(scheme: Scheme) -> None:
    arrays, grid, sources, receivers = lossy_room(scheme)
    problem = build_problem(arrays, seeds_of(grid, sources))
    drive = drive_for(problem, grid, sources, receivers, 90 * grid.Ts)
    kept = solve(problem, drive, np)
    timing: dict[str, Any] = {}
    for block in (1, 7, 64, 10_000):
        spooled = solve(problem, drive, np, records_on="host", spool_steps=block, timing=timing)
        assert spooled.dtype == np.float32 and np.array_equal(kept, spooled), block
    assert timing["records_on"] == "host" and np.abs(kept).max() > 0
    with pytest.raises(ValueError, match="on the device or on the host"):
        solve(problem, drive, np, records_on="disk")


@pytest.mark.parametrize("scheme", [CARTESIAN, FCC], ids=lambda s: s.name)
@pytest.mark.parametrize("reach", ["box", "reached"])
def test_a_solve_cut_in_slabs_is_the_solve_to_the_bit(scheme: Scheme, reach: str) -> None:
    arrays, grid, sources, receivers = lossy_room(scheme)
    problem = build_problem(arrays, None if reach == "box" else seeds_of(grid, sources))
    drive = drive_for(problem, grid, sources, receivers, 70 * grid.Ts)
    stepper = NumpyStepper(problem, drive)
    state, out = stepper.state(), stepper.records()
    for n in range(drive.steps):
        stepper.step(state, n, out)
    assert np.abs(out).max() > 0 and np.abs(state.u1).max() > 0
    for slabs in (2, 3):
        fields: dict[str, Any] = {}
        timing: dict[str, Any] = {}
        cut = solve_slabbed(
            problem, drive, np, slabs=slabs, spool_steps=11, fields=fields, timing=timing
        )
        assert np.array_equal(out, cut), "the records"
        assert np.array_equal(
            state.u1[: problem.column_count * problem.nz],
            fields["u1"][: problem.column_count * problem.nz],
        ), "the field"
        parts = slabs_of(problem, drive, slabs)
        planes = bounds_for(problem, slabs)
        assert [p.planes for p in parts] == planes and planes[0][0] == 0
        assert planes[-1][1] == problem.shape[0] and len(planes) == slabs
        # Every column is one slab's own, and the lossy nodes are shared out whole.
        assert sum(p.own[1] - p.own[0] for p in parts) == problem.column_count
        assert sum(p.problem.lossy for p in parts) == problem.lossy
        assert sum(p.record_rows.size for p in parts) == drive.record_index.size
        # Two planes a cut, every source of the batch, four bytes a node.
        assert timing["exchange_bytes_a_step"] == exchange_bytes(problem, drive.batch, parts)
        assert (
            0
            < timing["exchange_bytes_a_step"]
            <= ((slabs - 1) * 2 * problem.shape[1] * problem.shape[2] * drive.batch * 4)
        )
    with pytest.raises(ValueError, match="too thin"):
        bounds_for(problem, 9)


def test_the_launches_of_a_queue_make_the_pairs_the_campaign_makes(
    machine: dict[str, Path],
) -> None:
    together = a_campaign(machine)
    together.run()
    queued = a_campaign(machine, out="queued")
    queued.voxelise()
    queued.place()
    queued.heard_at = [sorted(cells) for cells in HEARD_AT]
    # Sized for the smallest card of the machine: a card that holds one source is given
    # launches of one, and so is the 24 GB card beside it, which then never waits.
    assert len(queued.launches([24e9, 1.0], host_bytes=8e9)) == 3
    assert len(queued.launches([24e9, 24e9], host_bytes=8e9)) == 1
    # The records of a launch are on the host: what the host has bounds a launch's rows.
    rows = sum(int(queued.cell_nodes(c).size) for c in (0, 1))
    tight = queued.launches([24e9], host_bytes=2.0 * 4.0 * queued.steps * (rows - 1))
    assert len(tight) > 1
    launches = queued.launches([], host_bytes=8e9)
    assert sum(len(held["items"]) for held in launches) == 3
    assert sorted(pair for held in launches for pair in held["pairs"]) == [
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 0),
    ]
    assert all(held["bytes"] > 0 and held["updates"] > 0 for held in launches)
    # A worker's own campaign: the same files, the launch handed over by its name alone.
    worker = a_campaign(machine, out="queued")
    records = [worker.run_launch(held["items"]) for held in launches]
    assert sum(r["pairs"] for r in records) == 4
    assert [r["cells"] for r in worker.gathered(records)] == [2, 1, 1]
    for source, cells in enumerate(HEARD_AT):
        for cell in cells:
            key = together.key_of(source, cell)
            assert np.array_equal(together.cache.read(key), worker.cache.read(key))
    # Made again, a launch finds its pairs and solves nothing.
    assert worker.run_launch(launches[0]["items"])["pairs"] == 0
    assert queued.launches([], host_bytes=8e9) == []
    # A launch the device does not hold comes back as two that ask for less.
    assert worker.launch_bytes(worker._ready[0], []) == pytest.approx(
        worker._ready[0].bytes_shared() + worker.fit_bytes()
    )
