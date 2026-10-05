"""What the solver is measured with: against itself, against the present engine, and for cost.

Three measurements, each one command of ``python -m reverberate.wave.lowband``:

- **verify**: on a small lossy room of both grids, the card's kernels
  against the ``numpy`` step bit for bit, a batch against its sources solved
  alone, the grid cut to its sources' reach against the whole box and, where
  PFFDTD is built, the present engine's binary on the same four files;
- **compare**: the pairs of one campaign against a reference, error per
  third octave to 1414 Hz and per ambisonic degree, the worst cell and the
  ninth decile. The reference is another campaign's pair cache (the present
  engine on the same bundle) or the low side of a field, the dense line of
  hssd_0076;
- **cost**: node updates a second, and seconds and USD per source position
  and per pair at an hourly rate, for several batch sizes.

**How an error is counted** is the project's own way
(``docs/open-questions/low-band-translation.md``): the energy of the
difference over the energy of the reference, over the 50 ms after the onset,
each channel weighted by ``j_n(k a)`` on a head's sphere, ``a = 0.10 m``; and
the level, the ratio of the two energies over the whole response. An
expansion about another centre is moved to the reference's first
(:func:`reverberate.spatial.translate.apply_translation`): an array stands
on a node of its own grid, and two grids' nodes are up to half a diagonal
apart.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.special import spherical_jn

from reverberate.audio import Atmosphere
from reverberate.spatial.lowband import LOW_RATE_HZ, LOW_SAMPLES, onset_s, to_stored
from reverberate.spatial.sh import degrees_of
from reverberate.spatial.translate import apply_translation
from reverberate.wave.comms import engine_indices, interp_weights, nearest_node
from reverberate.wave.lowband.box import box_arrays, write_entry
from reverberate.wave.lowband.problem import build_problem, load_problem, read_entry
from reverberate.wave.lowband.scheme import CARTESIAN, FCC, Scheme, numbers
from reverberate.wave.lowband.solver import drive_for, solve, steps_for

__all__ = [
    "THIRD_OCTAVES_HZ",
    "compare_caches",
    "compare_responses",
    "cost_table",
    "extract_line",
    "format_table",
    "grid_of_entry",
    "paper_numbers",
    "stored_of_cache",
    "verify",
]

#: Third octave centres whose band lies under the top of the crossover's ramp, 1414 Hz.
THIRD_OCTAVES_HZ = (
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
)
HEAD_RADIUS_M = 0.10
WINDOW_S = 0.05

#: Card memory a batch is sized for, GB, and the share of it a batch may take.
CARDS_GB = (16.0, 24.0, 48.0, 80.0)


# --------------------------------------------------------------------------
# the numbers on paper
# --------------------------------------------------------------------------


def paper_numbers(
    *,
    fmax_hz: float,
    duration_s: float,
    reference_nodes: float,
    reached_share: float,
    lossy_share: float,
    branches: int,
    rows: int,
) -> list[dict[str, Any]]:
    """Each scheme's solve on a storey: nodes, steps, memory a source, sources a card.

    ``reference_nodes`` is the box of the Cartesian grid at 10.5 points per
    wavelength; ``reached_share`` what a source reaches of a box and
    ``lossy_share`` its lossy boundary nodes over its nodes, both read on a
    voxelised grid (the second goes as the grid's step, the boundary being a
    surface). ``rows`` is the nodes a source is read at.
    """
    out = []
    reference = numbers(CARTESIAN, fmax_hz=fmax_hz, duration_s=duration_s)
    for scheme, ppw in ((CARTESIAN, 10.5), (CARTESIAN, 7.2), (FCC, 7.7)):
        held = numbers(
            scheme, fmax_hz=fmax_hz, duration_s=duration_s, ppw=ppw, reference_nodes=reference_nodes
        )
        box = held["nodes"]
        coarser = held["grid_step_m"] / reference["grid_step_m"]
        # A wall's nodes per area: one a cell on the Cartesian grid, one in two on the other.
        lossy = lossy_share * reference_nodes / coarser**2 * scheme.nodes_per_cell
        stored = box * min(1.0, reached_share + 0.15)
        per_source = 8.0 * stored + 8.0 * lossy * branches
        records = 4.0 * held["steps"] * rows
        held.update(
            {
                "reached_nodes": box * reached_share,
                "lossy_nodes": lossy,
                "air_updates": box * reached_share * held["steps"],
                "boundary_branch_updates": lossy * branches * held["steps"],
                "bytes_per_source": per_source,
                "record_bytes_per_source": records,
                "sources_per_card": {
                    f"{int(gb)} GB": int(max(1, (0.8 * gb * 1e9 - 2.5e9) // (per_source + records)))
                    for gb in CARDS_GB
                },
            }
        )
        out.append(held)
    return out


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def _small_room(scheme: Scheme) -> tuple[Any, Any, np.ndarray, list[np.ndarray]]:
    """A lossy room on ``scheme``'s grid, two sources and three receivers a source."""
    arrays, grid = box_arrays(scheme, (40, 36, 30), room=((4, 4, 4), (34, 30, 24)), lossy=True)
    sources = np.array([[10.3, 11.1, 9.6], [25.2, 20.4, 15.9]]) * grid.h
    points = np.array([[20, 18, 14], [30, 8, 8], [8, 26, 20]]) * grid.h
    nodes = np.array(
        [engine_indices(np.array([nearest_node(p, grid)[1]]), grid)[0] for p in points]
    )
    return arrays, grid, sources, [nodes, nodes]


def verify(
    out: Path, *, xp: Any, pffdtd_dir: Path | None = None, steps: int = 600, say: Any = print
) -> dict[str, Any]:
    """The solver against itself on ``xp`` and, with ``pffdtd_dir``, against the present engine."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for scheme in (CARTESIAN, FCC):
        arrays, grid, sources, receivers = _small_room(scheme)
        seeds = np.concatenate([engine_indices(interp_weights(s, grid)[1], grid) for s in sources])
        whole = build_problem(arrays, None)
        cut = build_problem(arrays, seeds)
        duration = steps * grid.Ts
        on_host = solve(cut, drive_for(cut, grid, sources, receivers, duration), np)
        record: dict[str, Any] = {
            "nodes": whole.box_nodes,
            "reached": cut.reached,
            "steps": steps,
            "cut_equals_whole_on_numpy": bool(
                np.array_equal(
                    on_host, solve(whole, drive_for(whole, grid, sources, receivers, duration), np)
                )
            ),
        }
        if xp is not np:
            on_card = xp.asnumpy(solve(cut, drive_for(cut, grid, sources, receivers, duration), xp))
            record["card_equals_numpy"] = bool(np.array_equal(on_card, on_host))
            record["card_max_difference_over_peak"] = float(
                np.abs(on_card - on_host).max() / np.abs(on_host).max()
            )
            alone = [
                xp.asnumpy(
                    solve(
                        cut, drive_for(cut, grid, sources[b : b + 1], [receivers[b]], duration), xp
                    )
                )
                for b in range(2)
            ]
            record["batch_equals_singles_on_card"] = bool(
                np.array_equal(np.concatenate(alone), on_card)
            )
        if pffdtd_dir is not None:
            record["present_engine"] = _against_engine(
                out / scheme.name, arrays, grid, sources, receivers, duration, on_host, pffdtd_dir
            )
        report[scheme.name] = record
        say(f"{scheme.name}: {json.dumps(record)}")
    if xp is not np:
        report["per_pair_path"] = _verify_filters(xp)
        say(f"per pair path: {json.dumps(report['per_pair_path'])}")
    (out / "verify.json").write_text(json.dumps(report, indent=1))
    return report


def _verify_filters(xp: Any) -> dict[str, Any]:
    """The kept resampler against the table's, and several cells a launch against one."""
    from reverberate.accel import dsp
    from reverberate.spatial.encode import EncoderSettings
    from reverberate.wave.lowband.fit import CellEncoder, Resampler

    rng = np.random.default_rng(7)
    rate, steps = 27307.107326536352, 6000
    signals = xp.asarray(rng.standard_normal((40, steps)))
    kept = Resampler.prepare(steps, rate, LOW_RATE_HZ, xp).apply(signals, xp)
    table = dsp.resample(signals, rate, LOW_RATE_HZ, xp)
    offsets = rng.uniform(-0.3, 0.3, size=(40, 3))
    offsets[0] = 0.0
    encoder = CellEncoder(
        scheme=CARTESIAN,
        grid_rate_hz=rate,
        grid_step_m=0.0218,
        sound_speed_m_s=343.2,
        fmax_hz=1500.0,
        settings=EncoderSettings(order=2, fit_order=3, max_frequency_hz=1500.0),
        xp=xp,
        samples=round(steps / rate * LOW_RATE_HZ),
        scale=1.0,
    )
    records = xp.asarray(rng.standard_normal((120, steps)).astype(np.float32))
    together = encoder.cells(records, [offsets] * 3)
    alone = [encoder.cell(records[40 * i : 40 * i + 40], offsets) for i in range(3)]
    peak = max(float(np.abs(a).max()) for a in alone)
    return {
        "resampler_equals_table": bool(xp.array_equal(kept, table)),
        "resampler_max_difference": float(xp.abs(kept - table).max()),
        "cells_against_cell_over_peak": max(
            float(np.abs(t - a).max()) for t, a in zip(together, alone, strict=True)
        )
        / peak,
    }


def _against_engine(
    work: Path,
    arrays: Any,
    grid: Any,
    sources: np.ndarray,
    receivers: list[np.ndarray],
    duration: float,
    mine: np.ndarray,
    pffdtd_dir: Path,
) -> dict[str, Any]:
    """PFFDTD's single precision card binary on the same room, a source at a time."""
    from reverberate.accel.solve import prepare_job, run_engine
    from reverberate.wave.comms import write_comms

    entry = write_entry(arrays, grid, work / "entry")
    # The receivers again as positions: a node's own coordinate is read back as that node.
    nx, ny, nz = grid.shape
    worst = 0.0
    rows = 0
    for b, source in enumerate(sources):
        job = work / f"job{b}"
        job.mkdir(parents=True, exist_ok=True)
        positions = _positions_of(receivers[b], grid)
        comms = write_comms(
            entry,
            source,
            positions,
            duration,
            diff_source=True,
            out_path=job / "comms_out.h5",
            interpolation="nearest",
        )
        prepare_job(job, entry, comms)
        result = run_engine(job, pffdtd_dir=pffdtd_dir, poll_s=2.0)
        with h5py.File(result.output, "r") as handle:
            theirs = np.asarray(handle["u_out"][...], dtype=np.float64)
        count = positions.shape[0]
        ours = mine[rows : rows + count, : theirs.shape[1]].astype(np.float64)
        rows += count
        worst = max(worst, float(np.abs(theirs - ours).max() / np.abs(theirs).max()))
    return {"max_difference_over_peak": worst}


def _positions_of(engine_nodes: np.ndarray, grid: Any) -> np.ndarray:
    """The coordinates of nodes given in the engine's space, for a box that is not rotated."""
    nx, ny, nz = grid.shape
    out = []
    for node in np.asarray(engine_nodes, dtype=np.int64):
        if grid.fcc:
            half = ny // 2 + 1
            ix, rest = divmod(int(node), half * nz)
            iy, iz = divmod(rest, nz)
            if (ix + iy + iz) % 2:
                iy = ny - 1 - iy
        else:
            ix, rest = divmod(int(node), ny * nz)
            iy, iz = divmod(rest, nz)
        out.append([grid.xv[ix], grid.yv[iy], grid.zv[iz]])
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def extract_line(field: Path, plan: Path, out: Path, *, every: int = 1) -> dict[str, Any]:
    """The low side of a field's points in the stored form, with where each array stood.

    ``field`` is a three band field (the dense line of hssd_0076) and
    ``plan`` its ``plan.json``. Its low side is
    :func:`reverberate.spatial.lowband.to_stored` of each response: the
    crossover's masks taken at 48 kHz, kept at 4 kHz. Under 800 Hz a field is
    its low grid's expansion and above it its mid grid's, each about its own
    node, and both centres are kept.
    """
    held = json.loads(Path(plan).read_text())
    with h5py.File(field, "r") as handle:
        index = np.asarray(handle["point_index"][...])[::every]
        positions = np.asarray(handle["positions"][...])[::every]
        rate = float(handle.attrs["sample_rate_hz"])
        source = np.asarray(handle.attrs["source_position"], dtype=float)
        stored = np.zeros((index.size, 64, LOW_SAMPLES), dtype=np.float32)
        for row, point in enumerate(range(0, handle["ir"].shape[0], every)):
            stored[row] = to_stored(np.asarray(handle["ir"][point], dtype=np.float64), rate)
    centres = {
        band: np.asarray([held["bands"][band]["centres"][int(i)] for i in index], dtype=float)
        for band in ("low", "mid")
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        stored=stored,
        cells=np.asarray([held["points"][int(i)] for i in index], dtype=float),
        positions=positions,
        centres_low=centres["low"],
        centres_mid=centres["mid"],
        source=source,
        seam_hz=np.float64(800.0),
    )
    return {"points": int(index.size), "source": source.tolist(), "out": str(out)}


def stored_of_cache(cache: Any, key: str, *, atmosphere: Atmosphere | None = None) -> np.ndarray:
    """A cached pair as the pack keeps it: its air and its masks taken, ``[64, 4800]``."""
    response = np.asarray(cache.read(key), dtype=np.float64)
    return np.asarray(
        to_stored(response, LOW_RATE_HZ, atmosphere=atmosphere or Atmosphere()), dtype=np.float64
    )


def compare_caches(reference: Path, candidate: Path) -> dict[str, Any]:
    """Two campaigns' pair caches against each other, whatever engine or grid made each.

    ``reference`` and ``candidate`` are campaigns' output directories. Their
    pairs are matched by the two positions each record names, so the keys,
    which name the solver and the grid, need not agree.
    """
    from reverberate.accel.pairs import PairCache

    def held(out: Path) -> tuple[Any, dict[tuple[float, ...], tuple[str, list[float]]]]:
        grids = sorted(p.name for p in (Path(out) / "pairs").iterdir() if p.is_dir())
        if len(grids) != 1:
            raise ValueError(f"{out} holds the pairs of {len(grids)} grids, not of one")
        cache = PairCache(Path(out) / "pairs", grids[0])
        found = {
            (*record["source_m"], *record["cell_m"]): (key, record["centre_m"])
            for key, record in cache.records().items()
        }
        return cache, found

    theirs, there = held(reference)
    ours, here = held(candidate)
    shared = sorted(set(there) & set(here))
    if not shared:
        raise ValueError("the two caches share no pair")
    table = compare_responses(
        np.stack([stored_of_cache(theirs, there[pair][0]) for pair in shared]),
        np.stack([stored_of_cache(ours, here[pair][0]) for pair in shared]),
        reference_centres=[(0.0, np.asarray([there[pair][1] for pair in shared], dtype=float))],
        candidate_centres=np.asarray([here[pair][1] for pair in shared], dtype=float),
    )
    table["reference"], table["candidate"] = str(reference), str(candidate)
    return table


def _band_edges(centre: float) -> tuple[float, float]:
    return centre / 2.0 ** (1.0 / 6.0), centre * 2.0 ** (1.0 / 6.0)


def compare_responses(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    reference_centres: list[tuple[float, np.ndarray]],
    candidate_centres: np.ndarray,
    sound_speed_m_s: float = 343.2,
    order: int = 7,
) -> dict[str, Any]:
    """Error and level of ``candidate`` against ``reference``, per third octave and per degree.

    Both are ``[cell, channel, sample]`` in the stored form at 4 kHz.
    ``reference_centres`` is a list of ``(from_hz, centres)``: the centre the
    reference's expansion stands on from that frequency up; the candidate,
    about ``candidate_centres``, is moved to it band by band.
    """
    cells, channels, samples = reference.shape
    freqs = np.fft.rfftfreq(samples, 1.0 / LOW_RATE_HZ)
    degree = degrees_of(order)[:channels]
    k = 2.0 * np.pi * freqs / sound_speed_m_s
    weight = np.asarray(spherical_jn(degree[:, None], k[None, :] * HEAD_RADIUS_M)) ** 2
    time_s = np.arange(samples) / LOW_RATE_HZ
    bands = {centre: _band_edges(centre) for centre in THIRD_OCTAVES_HZ}
    error = np.zeros((cells, len(bands)))
    level = np.zeros((cells, len(bands)))
    by_degree = np.zeros((cells, order + 1))
    for cell in range(cells):
        spectrum = np.fft.rfft(candidate[cell], axis=-1)
        moved = np.zeros_like(spectrum)
        for position, (from_hz, centres) in enumerate(reference_centres):
            upto = (
                reference_centres[position + 1][0]
                if position + 1 < len(reference_centres)
                else np.inf
            )
            chosen = (freqs >= from_hz) & (freqs < upto)
            offset = np.asarray(centres[cell], dtype=float) - candidate_centres[cell]
            if np.abs(offset).max() < 1e-9:
                moved[:, chosen] = spectrum[:, chosen]
            else:
                moved[:, chosen] = apply_translation(
                    spectrum[:, chosen],
                    offset,
                    freqs[chosen],
                    order,
                    sound_speed_m_s=sound_speed_m_s,
                    xp=np,
                )
        ours = np.fft.irfft(moved, n=samples, axis=-1)
        theirs = reference[cell]
        onset = onset_s(theirs[0], LOW_RATE_HZ)
        window = ((time_s >= onset - 0.005) & (time_s <= onset + WINDOW_S)).astype(float)
        early_ours = np.fft.rfft(ours * window, axis=-1)
        early_theirs = np.fft.rfft(theirs * window, axis=-1)
        whole_theirs = np.fft.rfft(theirs, axis=-1)
        difference = np.abs(early_ours - early_theirs) ** 2
        for column, (low, high) in enumerate(bands.values()):
            chosen = (freqs >= low) & (freqs < high)
            w = weight[:, chosen]
            error[cell, column] = (difference[:, chosen] * w).sum() / max(
                (np.abs(early_theirs[:, chosen]) ** 2 * w).sum(), 1e-300
            )
            level[cell, column] = (np.abs(moved[:, chosen]) ** 2 * w).sum() / max(
                (np.abs(whole_theirs[:, chosen]) ** 2 * w).sum(), 1e-300
            )
        inside = (freqs >= _band_edges(THIRD_OCTAVES_HZ[0])[0]) & (freqs < 1414.0)
        for n in range(order + 1):
            rows = degree == n
            by_degree[cell, n] = difference[rows][:, inside].sum() / max(
                (np.abs(early_theirs[rows][:, inside]) ** 2).sum(), 1e-300
            )

    def db(values: np.ndarray) -> np.ndarray:
        return np.asarray(10.0 * np.log10(np.maximum(values, 1e-30)))

    error_db, level_db, degree_db = db(error), db(level), db(by_degree)
    table = {
        "cells": cells,
        "third_octaves_hz": list(bands),
        "error_db": {
            "median": np.median(error_db, axis=0).round(1).tolist(),
            "p90": np.percentile(error_db, 90, axis=0).round(1).tolist(),
            "worst": error_db.max(axis=0).round(1).tolist(),
        },
        "level_db": {
            "median": np.median(level_db, axis=0).round(2).tolist(),
            "p90_abs": np.percentile(np.abs(level_db), 90, axis=0).round(2).tolist(),
            "worst_abs": np.abs(level_db).max(axis=0).round(2).tolist(),
        },
        "degree_error_db": {
            "median": np.median(degree_db, axis=0).round(1).tolist(),
            "p90": np.percentile(degree_db, 90, axis=0).round(1).tolist(),
            "worst": degree_db.max(axis=0).round(1).tolist(),
        },
    }
    table["worst_band"] = {
        "error_db": float(error_db.max()),
        "at_hz": float(list(bands)[int(np.argmax(error_db.max(axis=0)))]),
        "p90_error_db": float(np.percentile(error_db, 90, axis=0).max()),
        "level_abs_db": float(np.abs(level_db).max()),
        "p90_level_abs_db": float(np.percentile(np.abs(level_db), 90, axis=0).max()),
    }
    return table


def format_table(table: dict[str, Any]) -> str:
    """The comparison as text: a row a third octave, then a row a degree."""
    lines = [
        f"{table['cells']} cells; error over the 50 ms after the onset on a head's sphere,"
        " level over the whole response",
        "third octave | error dB: median  p90  worst | level dB: median  p90|.|  worst|.|",
    ]
    for i, centre in enumerate(table["third_octaves_hz"]):
        e, lv = table["error_db"], table["level_db"]
        lines.append(
            f"{centre:9.0f} Hz | {e['median'][i]:7.1f} {e['p90'][i]:6.1f} {e['worst'][i]:6.1f}"
            f" | {lv['median'][i]:7.2f} {lv['p90_abs'][i]:6.2f} {lv['worst_abs'][i]:6.2f}"
        )
    lines.append("degree | error dB, 90 to 1414 Hz, every channel alike: median  p90  worst")
    d = table["degree_error_db"]
    for n in range(len(d["median"])):
        lines.append(f"{n:6d} | {d['median'][n]:7.1f} {d['p90'][n]:6.1f} {d['worst'][n]:6.1f}")
    w = table["worst_band"]
    lines.append(
        f"worst band: error {w['error_db']:.1f} dB at {w['at_hz']:.0f} Hz"
        f" (p90 {w['p90_error_db']:.1f} dB), level {w['level_abs_db']:.2f} dB"
        f" (p90 {w['p90_level_abs_db']:.2f} dB)"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def cost_table(
    entry_dir: Path,
    sources: np.ndarray,
    cell_nodes: list[np.ndarray],
    *,
    xp: Any,
    duration_s: float,
    batches: tuple[int, ...],
    measured_steps: int,
    rate_usd_per_hour: float,
    cards: int,
    encoder: Any = None,
    offsets: np.ndarray | None = None,
    say: Any = print,
) -> dict[str, Any]:
    """Node updates a second and what a source position and a pair cost, per batch size.

    ``measured_steps`` steps are timed and the solve's full length is scaled
    from them; the stepping does not depend on the step. ``rate_usd_per_hour``
    is the machine's and ``cards`` how many cards it holds, each taken to run
    its own batches at the measured rate. With ``encoder`` one cell's records
    of the full length are filtered and fitted, which is a pair's cost.
    """
    from reverberate.wave.comms import load_grid

    grid = load_grid(entry_dir)
    sources = np.asarray(sources, dtype=float).reshape(-1, 3)
    seeds = np.concatenate([engine_indices(interp_weights(s, grid)[1], grid) for s in sources])
    t0 = time.time()
    problem = load_problem(entry_dir, seeds)
    prepare_s = time.time() - t0
    steps = steps_for(duration_s, grid.Ts)
    rows = np.concatenate(cell_nodes) if cell_nodes else np.zeros(0, dtype=np.int64)
    table: dict[str, Any] = {
        "grid": {
            **problem.record,
            "updated_nodes": problem.updated,
            "stored_nodes": problem.nodes,
            "bytes_per_source": problem.bytes_per_source(),
            "steps": steps,
            "sample_rate_hz": 1.0 / grid.Ts,
            "prepare_s": round(prepare_s, 2),
        },
        "billed_rate_usd_per_hour": rate_usd_per_hour,
        "cards": cards,
        "batches": [],
    }
    per_second = rate_usd_per_hour / 3600.0 / max(1, cards)
    for batch in batches:
        chosen = sources[np.arange(batch) % sources.shape[0]]
        timing: dict[str, Any] = {}
        try:
            drive = drive_for(problem, grid, chosen, [rows] * batch, measured_steps * grid.Ts)
            # Once to compile and to warm the card, then the timed run.
            if xp is not np:
                solve(problem, drive_for(problem, grid, chosen[:1], [rows], 8 * grid.Ts), xp)
            solve(problem, drive, xp, timing=timing)
        except Exception as error:  # noqa: BLE001 - a batch too large for the card is a row
            table["batches"].append({"batch": batch, "error": repr(error)[:200]})
            say(f"batch {batch}: {error!r}"[:200])
            continue
        seconds_a_source = timing["seconds"] / measured_steps * steps / batch
        row = {
            "batch": batch,
            "updates_per_s": timing["updates_per_s"],
            "box_updates_per_s": timing["updates_per_s"] * problem.box_nodes / problem.updated,
            "card_s_per_source": seconds_a_source,
            "machine_s_per_source": seconds_a_source / max(1, cards),
            "usd_per_source": seconds_a_source * per_second,
        }
        table["batches"].append(row)
        say(
            f"batch {batch:4d}: {row['updates_per_s']:.3g} node updates/s,"
            f" {seconds_a_source:.3f} card s a source position,"
            f" {100 * row['usd_per_source']:.4f} US cents"
        )
    if encoder is not None and offsets is not None and cell_nodes:
        count = int(cell_nodes[0].size)
        noise = xp.asarray(
            np.random.default_rng(0).standard_normal((count, steps)).astype(np.float32)
        )
        encoder.cell(noise, offsets)
        t0 = time.time()
        repeats = 5
        for _ in range(repeats):
            encoder.cell(noise, offsets)
        pair_s = (time.time() - t0) / repeats
        table["pair"] = {
            "nodes": count,
            "card_s_per_pair": pair_s,
            "usd_per_pair": pair_s * per_second,
            "prepare_s": round(encoder.prepare_s, 1),
            "operator_bytes": int(sum(o.bytes_on_device for o in encoder.operators.values())),
        }
        say(f"a pair: {pair_s:.3f} card s, {100 * pair_s * per_second:.5f} US cents")
    return table


def grid_of_entry(entry_dir: Path) -> dict[str, Any]:
    """An entry's grid as the solver reads it, without a source: the whole box."""
    arrays = read_entry(entry_dir)
    return {
        "scheme": arrays.scheme.name,
        "shape": list(arrays.shape),
        "box_nodes": arrays.points,
        "boundary_nodes": int(arrays.bn_ixyz.size),
        "lossy_nodes": int((arrays.mat_bn >= 0).sum()),
        "grid_step_m": arrays.h,
        "sample_rate_hz": 1.0 / arrays.ts,
        "branches": max((int(m.shape[0]) for m in arrays.materials), default=0),
    }
