"""The early trace's compiled text, its shared store and the levelling's kept noise: the same bits.

Each of the three makes the trace cheaper and none may change a table. The
compiled text (:mod:`reverberate.mirror.native`) is held to its ``numpy``
twin field by field, on a room with a doorway, off the axes, and with the
diffracted onsets; the text a card is given is built for the host with a
loop in place of the card's threads, and gives the host's rows. A store
(:mod:`reverberate.mirror.shared`) filled by one trace is read by the next
and by two processes at once, which make each entry once between them. The
tail of a kept seed, the bank's kept transforms and the air's frames taken
at once are the arrays they were.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate import audio, compute
from reverberate.metrics import octave_filter_rows
from reverberate.mirror import native
from reverberate.mirror.moving import EarlyTable, MovingSettings, prepare, trace_early
from reverberate.mirror.moving_onset import onset_field
from reverberate.mirror.rays import Histogram
from reverberate.mirror.render import RenderSettings, TailNoise, band_rows, tail_from_histogram
from reverberate.mirror.shared import Store, name_of
from test_mirror_diffract import walled_box
from test_mirror_moving import SETTINGS, a_walk, tilted

C = 343.2
FIELDS = (
    "offsets",
    "path_id",
    "delay_s",
    "arrival",
    "departure",
    "gain",
    "order",
    "kind",
    "length_m",
    "sequence",
    "rank",
)
SMALL = MovingSettings(
    source_pitch_m=0.3, listener_pitch_m=0.3, pairs_per_block=600, pairs_per_validation=200
)

compiled = pytest.mark.skipif(not native.available(), reason=f"no C compiler: {native.why_not()}")


def same(a: EarlyTable, b: EarlyTable) -> None:
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def walked(catalogue: Any, store: Store | None = None, steps: int = 24) -> EarlyTable:
    """A walk through the doorway and back, with its onsets: many anchors, lists and blocks."""
    ms = prepare(catalogue, SETTINGS, SMALL, store=store)
    source, listener = a_walk(steps)
    both = np.concatenate([source, listener])
    region = (both.min(axis=0) - 0.5, both.max(axis=0) + 0.5)
    onsets = onset_field(catalogue, both, sound_speed_m_s=C, store=store)
    return trace_early(ms, source, listener, region=region, onsets=onsets)


# --------------------------------------------------------------------------
# the compiled text against its twin
# --------------------------------------------------------------------------


@compiled
@pytest.mark.parametrize("off_the_axes", [False, True], ids=["on the axes", "off the axes"])
def test_the_compiled_text_writes_the_twin_s_table_to_the_bit(off_the_axes: bool) -> None:
    catalogue = tilted(walled_box())[0] if off_the_axes else walled_box()
    if off_the_axes:
        # The walk is the room's own, turned with it.
        turn = tilted(walled_box())[1]
        ms = prepare(catalogue, SETTINGS, SMALL)
        source, listener = (points @ turn.T for points in a_walk(16))
        with native.disabled():
            twin = trace_early(prepare(catalogue, SETTINGS, SMALL), source, listener)
        mine = trace_early(ms, source, listener)
    else:
        with native.disabled():
            twin = walked(catalogue)
        mine = walked(catalogue)
        # Steps behind the wall have their onsets, and those went through the text too.
        assert twin.record["without_direct"] > 0 and twin.record["diffraction"]["found"] > 0
    assert twin.record["engine"] == "numpy" and mine.record["engine"] == "compiled"
    assert mine.offsets[-1] > 100
    same(mine, twin)
    # What the sieve was given and what it kept: the same counts, a pair at a time or all at once.
    assert mine.record["sieved"] == twin.record["sieved"]
    assert mine.record["pairs"] == twin.record["pairs"]


FAKE_CARD = (
    r"""
#include <math.h>
struct RvDim { unsigned int x; };
static RvDim blockIdx, blockDim, threadIdx;
#define __device__ static
#define __forceinline__ inline
#define __global__
static inline double __longlong_as_double(long long) { return HUGE_VAL; }
"""
    + native.DEVICE_SOURCE
    + r"""
extern "C" void fake_sieve(
    long long threads, RV_SCENE_ARGS,
    long long count, long long width, const double* positions, const int* order,
    const int* parent, const int* sequence, const double* rotation,
    const double* translation, const double* source, const double* listener,
    const long long* pair_job, const long long* pair_image, long long pairs, double margin,
    unsigned char* kept)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (long long p = 0; p < threads; ++p) {
        blockIdx.x = (unsigned int)p;
        rv_sieve_pairs(
            RV_SCENE_PASS, count, width, positions, order, parent, sequence, rotation,
            translation, source, listener, pair_job, pair_image, pairs, margin, kept);
    }
}

extern "C" void fake_validate(
    long long threads, RV_SCENE_ARGS,
    long long width, const int* order, const int* sequence,
    const double* source, const double* listener,
    const long long* pair_job, const long long* pair_image, long long pairs,
    int has_region, const double* lo, const double* hi, double reach, double epsilon,
    unsigned char* alive, double* out_at, double* out_first, double* out_chain)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (long long p = 0; p < threads; ++p) {
        blockIdx.x = (unsigned int)p;
        rv_validate_kept(
            RV_SCENE_PASS, width, order, sequence, source, listener, pair_job, pair_image,
            pairs, has_region, lo, hi, reach, epsilon, alive, out_at, out_first, out_chain);
    }
}
"""
)


class Card:
    """An array module that is not ``numpy`` and computes as it does: a card that is not there."""

    __name__ = "a card that is not there"

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


def fake_kernels(tmp: Path) -> Any:
    """The card's text built for the host, a loop for the card's threads; ``None`` without C++."""
    compiler = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    if compiler is None:
        return None
    (tmp / "card.cpp").write_text(FAKE_CARD)
    flags = ["-std=c++14", "-O1", "-fPIC", "-shared", "-ffp-contract=off"]
    built = subprocess.run(
        [compiler, *flags, str(tmp / "card.cpp"), "-o", str(tmp / "card.so")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr[-2000:]
    held = ctypes.CDLL(str(tmp / "card.so"))
    names = {"rv_sieve_pairs": held.fake_sieve, "rv_validate_kept": held.fake_validate}

    def kernel(source: str, name: str) -> Any:
        assert source == native.DEVICE_SOURCE

        def launch(grid: tuple[int, ...], block: tuple[int, ...], given: tuple[Any, ...]) -> None:
            passed: list[Any] = [ctypes.c_longlong(grid[0] * block[0])]
            for value in given:
                if isinstance(value, np.ndarray):
                    assert value.flags.c_contiguous
                    passed.append(ctypes.c_void_p(value.ctypes.data))
                elif isinstance(value, np.floating):
                    passed.append(ctypes.c_double(float(value)))
                elif isinstance(value, np.int32):
                    passed.append(ctypes.c_int(int(value)))
                else:
                    passed.append(ctypes.c_longlong(int(value)))
            names[name](*passed)

        return launch

    return kernel


@compiled
def test_the_text_a_card_is_given_gives_the_host_s_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = fake_kernels(tmp_path)
    if kernel is None:
        pytest.skip("no C++ compiler to build the card's text with")
    catalogue = walled_box()
    source, listener = a_walk(16)
    region = (listener.min(axis=0) - 2.5, listener.max(axis=0) + 2.5)
    on_host = trace_early(prepare(catalogue, SETTINGS, SMALL), source, listener, region=region)
    monkeypatch.setattr(compute, "raw_kernel", kernel)
    monkeypatch.setenv(native.CARD_VARIABLE, "1")
    on_card = trace_early(
        prepare(catalogue, SETTINGS, SMALL), source, listener, region=region, xp=Card()
    )
    assert on_card.record["engine"] == "compiled, on a card"
    assert on_card.record["pairs"] == on_host.record["pairs"] > 0
    same(on_card, on_host)


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


def test_a_store_changes_no_table_and_the_next_trace_makes_nothing(tmp_path: Path) -> None:
    catalogue = walled_box()
    alone = walked(catalogue)
    first = Store(tmp_path / "store")
    same(walked(catalogue, first), alone)
    assert first.made["scene"] == first.made["onsets"] == 1
    assert first.made["trees"] > 3 and first.made["lists"] > 10 and first.made["fields"] > 0
    # A resumed run, or a second scene on these positions: every entry is read.
    second = Store(tmp_path / "store")
    same(walked(catalogue, second), alone)
    assert second.made == {} and second.found["scene"] == second.found["onsets"] == 1
    assert second.found["trees"] == first.made["trees"]
    assert second.found["fields"] == first.made["fields"]
    # An entry is named by what it was made from: another scene, another name.
    other = Store(tmp_path / "store")
    walked(tilted(catalogue)[0], other, steps=4)
    assert other.made["scene"] == 1 and "scene" not in other.found
    assert name_of(np.zeros(3), "a") != name_of(np.zeros(3, dtype=np.float32), "a")


def _walk_in_a_process(root: str, link: Any) -> None:
    store = Store(Path(root))
    table = walked(walled_box(), store)
    link.send(({name: getattr(table, name) for name in FIELDS}, store.made, store.found))


def test_two_processes_on_one_store_make_each_entry_once_between_them(tmp_path: Path) -> None:
    alone = walked(walled_box(), Store(tmp_path / "alone"))
    made_alone = Store(tmp_path / "count")
    walked(walled_box(), made_alone)
    context = multiprocessing.get_context("spawn")
    links, processes = [], []
    for _ in range(2):
        here, there = context.Pipe()
        process = context.Process(
            target=_walk_in_a_process, args=(str(tmp_path / "both"), there), daemon=True
        )
        process.start()
        links.append(here)
        processes.append(process)
    answers = [link.recv() for link in links]
    for process in processes:
        process.join(timeout=20)
    made: dict[str, int] = {}
    for fields, mine, _ in answers:
        for name in FIELDS:
            np.testing.assert_array_equal(fields[name], getattr(alone, name), err_msg=name)
        for kind, count in mine.items():
            made[kind] = made.get(kind, 0) + count
    # What costs is made under its own lock, or claimed: once, whoever asked first.
    for kind in ("scene", "onsets", "trees", "fields"):
        assert made[kind] == made_alone.made[kind], kind
    assert made["lists"] >= made_alone.made["lists"]


# --------------------------------------------------------------------------
# the levelling's arrays
# --------------------------------------------------------------------------


def a_histogram(seed: int, bins: int = 60, order: int = 3) -> Histogram:
    rng = np.random.default_rng(seed)
    channels = (order + 1) ** 2
    energy = rng.random((1, bins, 7)) * np.exp(-np.arange(bins) / 12.0)[None, :, None]
    moments = rng.standard_normal((1, bins, 7, channels)) * energy[..., None]
    moments[..., 0] = energy
    return Histogram(
        energy=energy,
        moments=moments,
        hits=np.ones((1, bins), dtype=np.int64),
        bin_s=0.002,
        bands_hz=(125, 250, 500, 1000, 2000, 4000, 8000),
        order=order,
        rays=1000,
    )


@pytest.mark.parametrize("order", [0, 2])
def test_a_tail_of_a_kept_seed_is_the_tail(order: int) -> None:
    settings = RenderSettings(order=order, duration_s=0.1, sample_rate_hz=48000.0)
    given: dict[str, Any] = {
        "sound_speed_m_s": C,
        "start_s": 0.004,
        "bursts": 6,
        "scale_per_band": np.linspace(0.5, 2.0, 8),
    }
    noise = TailNoise(keep=1)
    for seed, histogram in ((7, a_histogram(1)), (7, a_histogram(2)), (9, a_histogram(1))):
        fresh, _ = tail_from_histogram(histogram, 0, settings, seed=seed, **given)
        kept, _ = tail_from_histogram(histogram, 0, settings, seed=seed, noise=noise, **given)
        np.testing.assert_array_equal(kept, fresh)
        assert float(np.abs(fresh).max()) > 0.0
    # Two tails of seed 7 drew once; seed 9 took its place.
    assert (noise.drawn, noise.read) == (2, 1) and len(noise._held) == 1


def test_the_bank_s_kept_transforms_filter_as_the_bank_does() -> None:
    rows = np.random.default_rng(3).standard_normal((8, 4800))
    bands = np.arange(8)
    np.testing.assert_array_equal(
        band_rows(rows, 48000.0, bands), octave_filter_rows(rows, 48000, bands)
    )
    some = np.array([5, 5, 0])
    np.testing.assert_array_equal(
        band_rows(rows[:3], 48000.0, some), octave_filter_rows(rows[:3], 48000, some)
    )


def a_frame_at_a_time(signals: np.ndarray, rate: float, frame: int, start_s: float) -> np.ndarray:
    """:func:`reverberate.audio.apply_air_absorption` as it was: one transform a frame."""
    block = np.atleast_2d(np.asarray(signals, dtype=float))
    hop = frame // 4
    window = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame) / frame))
    length = block.shape[1]
    padded = np.zeros((block.shape[0], length + 2 * frame), dtype=float)
    padded[:, frame : frame + length] = block
    out = np.zeros_like(padded)
    overlap = np.zeros(padded.shape[1])
    attenuation = audio.Atmosphere().attenuation_np_per_m(np.fft.rfftfreq(frame, 1.0 / rate))
    for start in range(0, padded.shape[1] - frame + 1, hop):
        centre = (start + frame / 2.0 - frame) / rate + start_s
        gain = np.exp(-attenuation * C * max(centre, 0.0))
        spectrum = np.fft.rfft(padded[:, start : start + frame] * window, axis=-1)
        out[:, start : start + frame] += np.fft.irfft(spectrum * gain, n=frame, axis=-1) * window
        overlap[start : start + frame] += window * window
    return np.asarray(out[:, frame : frame + length] / overlap[frame : frame + length])


@pytest.mark.parametrize(("rate", "frame", "channels"), [(48000.0, 1024, 1), (4000.0, 84, 3)])
def test_the_air_s_frames_taken_at_once_are_the_frames_taken_one_by_one(
    rate: float, frame: int, channels: int
) -> None:
    signals = np.random.default_rng(5).standard_normal((channels, int(0.3 * rate)))
    for start_s in (0.0, 0.01):
        np.testing.assert_array_equal(
            audio.apply_air_absorption(
                signals, rate, sound_speed_m_s=C, frame=frame, start_time_s=start_s
            ),
            a_frame_at_a_time(signals, rate, frame, start_s),
        )
