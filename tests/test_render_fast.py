"""The fast parts of the signal engine against the reference's, and against themselves.

``test_render_engine.py`` holds the reference parts against closed forms, in
double precision. Here the fast parts (``render/fast.py``), which render in
single precision and order the same mathematics otherwise, are held to
them: the early part and the tail to the rounding of single precision, the
low band to that where the head is on its cell and to the quadrature's own
error where it is moved; then to themselves, as the reference is: blocks,
ranges, runs, the C text against its twin, several processes against one.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.signal import fftconvolve

from reverberate.render import delay, native
from reverberate.render.benchmark import density_pack
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.mix import write_mix
from reverberate.render.output import open_signal, write_signal
from reverberate.render.pack import ScenePack, synthetic_free_field
from reverberate.spatial.sh import real_sh, scene_to_ambisonic
from reverberate.spatial.translate import (
    fusion_inverse,
    pair_inverse,
    translation_matrices,
    translation_operator,
)

C = 343.2
FS = 48000
#: What single precision leaves of a part's peak: a few sums of values rounded at 6e-8.
SINGLE = 2e-6
REFERENCE = RenderSettings(engine="reference")


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).standard_normal(int(round(seconds * FS))))


def small_dense(moving: bool = True, **changes: int) -> ScenePack:
    return density_pack(duration_s=0.3, moving=moving, bins=30, **changes)


def off(got: Any, want: Any) -> float:
    """The largest difference over the peak of what was wanted."""
    return float(np.abs(got - want).max() / np.abs(want).max())


def test_the_default_engine_is_the_fast_one_and_renders_in_single_precision() -> None:
    assert RenderSettings().engine == "fast"
    out = Engine(small_dense(), {"s1": noise(0.3)}).stem("s1")
    assert out.dtype == np.float32 and np.abs(out).max() > 0.0
    with pytest.raises(ValueError, match="an engine is one of"):
        RenderSettings(engine="quick")
    with pytest.raises(ValueError, match="multiple of a run"):
        RenderSettings(tail_steps=15)


def test_a_source_at_rest_gives_the_delay_the_gain_and_the_direction() -> None:
    distance = 480 * C / FS  # ten milliseconds, a whole number of samples
    direction = np.array([0.6, 0.0, -0.8])
    pack = synthetic_free_field(
        source=distance * direction, listener_start=(0, 0, 0), duration_s=0.5
    )
    dry = noise(0.4)
    out = Engine(pack, {"s1": dry}).render()
    want = Engine(pack, {"s1": dry}, settings=REFERENCE).render()
    assert out.shape == (64, 24000)
    assert off(out, want) < SINGLE
    lag = np.argmax(fftconvolve(out[0].astype(float), dry[::-1])) - (dry.size - 1)
    assert lag == 480
    # The level and the direction, read off the channels themselves: 1 / d on each harmonic.
    harmonics = real_sh(7, scene_to_ambisonic(direction[None]))[0]
    middle = slice(2000, 18000)
    ratio = out[:, middle] @ out[0, middle] / (out[0, middle] @ out[0, middle])
    np.testing.assert_allclose(ratio, harmonics / harmonics[0], atol=1e-5)
    assert np.sqrt(np.mean(out[0, middle] ** 2)) == pytest.approx(
        np.sqrt(np.mean(want[0, middle] ** 2)), rel=1e-6
    )


def test_a_walk_is_the_reference_s_walk_delay_line_and_doppler() -> None:
    """A listener crossing a source at 1.5 m/s: every sample of the moving delay line."""
    pack = synthetic_free_field(
        source=(0.6, 1.5, 0.4), listener_start=(-0.6, 1.5, 0.0), listener_end=(0.9, 1.5, 0.0)
    )
    tone = np.sin(2.0 * np.pi * 3000.0 * np.arange(FS) / FS)
    out = Engine(pack, {"s1": tone}).render()
    want = Engine(pack, {"s1": tone}, settings=REFERENCE).render()
    assert off(out, want) < SINGLE


@pytest.mark.parametrize("part", ["early", "tail"])
@pytest.mark.parametrize("moving", [False, True])
def test_the_early_part_and_the_tail_are_the_reference_s(part: str, moving: bool) -> None:
    pack = small_dense(moving)
    dry = {"s1": noise(0.3)}
    got = Engine(pack, dry).stem("s1", parts=(part,))
    want = Engine(pack, dry, settings=REFERENCE).stem("s1", parts=(part,))
    assert np.abs(want).max() > 0.0
    assert off(got, want) < SINGLE


def test_a_level_a_band_above_the_crossover_is_the_reference_s() -> None:
    """The tapered join: three sets of bands, each at its own level a step, in every part."""
    pack = small_dense()
    source = pack.sources["s1"]
    rng = np.random.default_rng(7)
    scalar = (-3.3 + rng.uniform(-2.0, 2.0, pack.header.steps)).astype(np.float32)
    taper = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
    table = (-3.3 + (scalar[:, None] + 3.3) * taper[None, :]).astype(np.float32)
    level = replace(source.level, high_gain_db=scalar, band_gain_db=table)
    pack = replace(pack, sources={"s1": replace(source, level=level)})
    dry = {"s1": noise(0.3)}
    got = Engine(pack, dry).render()
    assert off(got, Engine(pack, dry, settings=REFERENCE).render()) < 2e-2  # the low band's
    for part in ("early", "tail"):
        made = Engine(pack, dry).render(parts=(part,))
        want = Engine(pack, dry, settings=REFERENCE).stem("s1", parts=(part,))
        assert off(made, want) < SINGLE
        flat = Engine(pack, dry, settings=RenderSettings(seam="broadband")).render(parts=(part,))
        assert off(flat, want) > 1e-3  # the table is read, and not the scalar


def test_the_low_band_is_the_reference_s_to_the_quadrature_s_error() -> None:
    """On its cell and one cell moved a little; then far, where the quadrature itself errs."""
    dry = {"s1": noise(0.5)}
    near = synthetic_free_field(
        level="B", listener_start=(0.0, 1.5, 0.0), listener_end=(0.05, 1.5, 0.0), duration_s=0.5
    )
    got = Engine(near, dry).stem("s1", parts=("low",))
    want = Engine(near, dry, settings=REFERENCE).stem("s1", parts=("low",))
    assert np.abs(want).max() > 0.0
    assert off(got, want) < 1e-5
    # Two cells fused, a head up to 0.3 m from each, a response that is noise to 2 kHz:
    # the reference's quadrature is 6e-4 of its operator at k d = 7.3, the closed form exact.
    far = small_dense()
    got = Engine(far, {"s1": noise(0.3)}).stem("s1", parts=("low",))
    want = Engine(far, {"s1": noise(0.3)}, settings=REFERENCE).stem("s1", parts=("low",))
    assert off(got, want) < 2e-2


def test_the_translation_in_closed_form_is_the_quadrature_s_where_it_is_exact() -> None:
    freqs = np.arange(75) * 20.0
    offsets = np.array([[0.05, 0.02, -0.03], [0.0, 0.0, 0.0], [0.1, 0.0, 0.05]])
    made = translation_matrices(offsets, freqs, 7)
    assert made.shape == (3, 75, 64, 64) and made.dtype == np.complex64
    for index, offset in enumerate(offsets):
        want = translation_operator(offset, freqs, 7, xp=np)
        # k d is 3.0 at most here: the quadrature's error is under 1e-7 of the operator.
        assert np.abs(made[index] - want).max() < 2e-6
        np.testing.assert_array_equal(translation_matrices(offset, freqs, 7), made[index])
    np.testing.assert_allclose(made[1], np.broadcast_to(np.eye(64), (75, 64, 64)), atol=1e-6)


def test_two_cells_inverse_by_blocks_is_the_library_s() -> None:
    freqs = np.arange(0.0, 1500.0, 100.0)
    for between in ([0.4, 0.0, 0.0], [0.17, 0.0, -0.33]):
        want = fusion_inverse(np.array([[0.0, 0.0, 0.0], between]), freqs, 7, xp=np)
        got = pair_inverse(np.array(between), freqs, 7)
        assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()


def test_the_c_text_and_its_twin_give_the_same_bits() -> None:
    if not native.available():
        pytest.skip(f"no build of the engine's loops here: {native.why_not()}")
    pack = small_dense()
    dry = {"s1": noise(0.3)}
    with native.disabled():
        twin = Engine(pack, dry).render()
    np.testing.assert_array_equal(Engine(pack, dry).render(), twin)
    streams = np.arange(40, 45)
    with native.disabled():
        drawn = native.carrier(0xDEADBEEF12345678, streams, 7, 500)
    np.testing.assert_array_equal(native.carrier(0xDEADBEEF12345678, streams, 7, 500), drawn)


def test_the_threads_of_a_first_render_are_given_one_library_and_one_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process's first run asks from every thread at once: one of each is made, for all.

    Each used to make a table of its own, and the one stored last freed the
    one a loop in C was reading; and a thread that asked for the library
    while another built it was told there was none. Both are made slowly
    here, so that every thread asks before the first has finished.
    """
    made = {"library": 0, "table": 0}
    build, kernel = native._build, delay.kernel_table

    def slow_build() -> Any:
        made["library"] += 1
        time.sleep(0.05)
        return build()

    def slow_kernel() -> np.ndarray:
        made["table"] += 1
        time.sleep(0.05)
        return np.asarray(kernel())

    monkeypatch.delenv("REVERBERATE_NO_NATIVE", raising=False)
    monkeypatch.setattr(native, "_build", slow_build)
    monkeypatch.setattr(delay, "kernel_table", slow_kernel)
    for name, fresh in (("tried", False), ("library", None), ("why", None), ("table", None)):
        monkeypatch.setitem(native._state, name, fresh)
    threads = 8
    together = threading.Barrier(threads)

    def first_use(_: int) -> tuple[Any, np.ndarray]:
        together.wait()
        return native._library(), native._table()

    with ThreadPoolExecutor(max_workers=threads) as pool:
        got = list(pool.map(first_use, range(threads)))
    assert made == {"library": 1, "table": 1}
    assert all(library is got[0][0] and table is got[0][1] for library, table in got)
    assert got[0][1].dtype == np.float32 and not got[0][1].flags.writeable
    np.testing.assert_array_equal(got[0][1], kernel().astype(np.float32))


def test_the_loop_in_c_reads_a_signal_that_is_not_one_block_as_its_twin_does() -> None:
    """What the loop is given the address of is what it is to read, kept until it is back."""
    if not native.available():
        pytest.skip(f"no build of the engine's loops here: {native.why_not()}")
    rng = np.random.default_rng(3)
    step, nodes, channels = 240, 4, 4
    # Every other sample of an array, and one in double precision: neither is the text's.
    strided = rng.standard_normal(4000).astype(np.float32)[::2]
    double = rng.standard_normal(2000)
    given: dict[str, Any] = {
        "start": 600,
        "nodes": nodes,
        "rate": 48000.0,
        "lead": 0.0,
        "speed": C,
        "q0": np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "q1": np.array([[1.2, 0.0, 0.0], [0.0, 2.0, 0.1]]),
        "l0": np.zeros(3),
        "l1": np.array([0.05, 0.0, 0.0]),
        "first": [(strided, 100), None],
        "second": [(strided, 100), (double, 0)],
        "harmonics": rng.standard_normal((2, nodes + 1, channels)).astype(np.float32),
    }
    out = np.zeros((channels, step), dtype=np.float32)
    native.early_interval(out, **given)
    twin = np.zeros((channels, step), dtype=np.float32)
    with native.disabled():
        native.early_interval(twin, **given)
    assert np.abs(twin).max() > 0.0
    np.testing.assert_array_equal(out, twin)
    with pytest.raises(ValueError, match="float32 in one block"):
        native.early_interval(np.zeros((channels, step)), **given)


def test_blocks_ranges_and_runs_give_the_same_samples() -> None:
    pack = small_dense()
    dry = {"s1": noise(0.3)}
    runs = RenderSettings(chunk_steps=2)  # three runs in the scene: blocks cross their joins
    whole = Engine(pack, dry, settings=runs).render()
    assert whole.dtype == np.float32
    for size in (1000, 7777):
        blocks = list(Engine(pack, dry, settings=runs).blocks(size))
        np.testing.assert_array_equal(np.concatenate(blocks, axis=1), whole)
    # A seek is a slice: no state is carried from what precedes it.
    np.testing.assert_array_equal(
        Engine(pack, dry, settings=runs).render(5000, 9000), whole[:, 5000:9000]
    )
    # A run's length, and the tail's, are settings: a transform's rounding apart.
    for other in (RenderSettings(), RenderSettings(chunk_steps=2, tail_steps=4)):
        assert off(Engine(pack, dry, settings=other).render(), whole) < SINGLE


def test_a_mix_is_its_stems_sum_and_shares_what_they_have_in_common() -> None:
    pack = small_dense(moving=False, sources=3)
    dry = {name: noise(0.3, seed) for seed, name in enumerate(pack.sources)}
    engine = Engine(pack, dry)
    mix = engine.render()
    stems = [engine.stem(name).astype(np.float64) for name in pack.sources]
    assert off(mix, np.sum(stems, axis=0)) < SINGLE
    assert off(mix, Engine(pack, dry, settings=REFERENCE).render()) < 2e-2
    for part in ("early", "low", "tail"):
        alone = engine.render(parts=(part,))
        apart = [engine.stem(name, parts=(part,)) for name in pack.sources]
        assert off(alone, np.sum(apart, axis=0, dtype=np.float64)) < SINGLE
    np.testing.assert_array_equal(engine.render(sources=[]), np.zeros_like(mix))


def _engine(sources: int) -> Engine:
    pack = small_dense(sources=sources)
    return Engine(pack, {name: noise(0.3, seed) for seed, name in enumerate(pack.sources)})


PARTS = ("early", "low", "tail")


def _parts(sources: int) -> dict[str, np.ndarray]:
    engine = _engine(sources)
    return {part: engine.render(parts=(part,)) for part in PARTS}


def _what_differs(made: np.ndarray, wanted: np.ndarray, sources: int) -> str:
    """For a failure's message: where two files differ, and in which part of the engine.

    ``made`` and ``wanted`` are ``[frame, channel]``. The parts are rendered
    again, here and by a process started afresh, each held against the other.
    """
    apart = np.abs(made - wanted)
    frames = np.flatnonzero(apart.max(axis=1) > 0)
    channels = np.flatnonzero(apart.max(axis=0) > 0)
    frame, channel = np.unravel_index(int(apart.argmax()), apart.shape)
    here = _parts(sources)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        there = pool.apply(_parts, (sources,))
    parts = ", ".join(f"{part} {off(there[part], here[part]):.3e}" for part in PARTS)
    return (
        f"{frames.size} of {made.shape[0]} frames differ, {frames[0]} to {frames[-1]}, on "
        f"{channels.size} of {made.shape[1]} channels, the most at frame {frame} of channel "
        f"{channel}; a fresh process's parts from this one's: {parts}; "
        f"the C text {'loaded' if native.available() else 'not loaded'} here"
    )


def test_a_mix_rendered_here_is_the_file_one_engine_writes_and_a_window_its_samples(
    tmp_path: Path,
) -> None:
    engine = _engine(2)
    h = engine.pack.header
    one = write_signal(
        tmp_path / "one", engine.blocks(4800), sample_rate_hz=h.sample_rate_hz, order=h.order
    )
    here = write_mix(tmp_path / "here", partial(_engine, 2), processes=1, scratch=tmp_path)
    assert here["sha256"] == one["sha256"] and here["peak"] == pytest.approx(one["peak"])
    assert here["frames"] == h.samples and here["render"]["processes"] == 1
    # A window of the scene is the scene's samples there.
    wanted = np.array(open_signal(tmp_path / "one").frames)
    window = write_mix(tmp_path / "w", partial(_engine, 2), processes=1, start=2400, stop=9600)
    np.testing.assert_array_equal(open_signal(tmp_path / "w").frames, wanted[2400:9600])
    assert window["frames"] == 7200
    assert not list(tmp_path.glob("carriers-*"))


def test_several_processes_write_the_file_one_engine_writes(tmp_path: Path) -> None:
    engine = _engine(2)
    h = engine.pack.header
    one = write_signal(
        tmp_path / "one", engine.blocks(4800), sample_rate_hz=h.sample_rate_hz, order=h.order
    )
    # Processes started afresh, three times, held to the very bytes this process writes.
    # For a while they were held to 1e-12 of its peak, because on the CI's Linux the two
    # did not write one digest one run in two: that was the delay table freed under the
    # loop reading it, in whichever process rendered for the first time on several
    # threads (docs/open-questions/engine-speed.md, "the same bytes"), and no difference
    # of a rounding's size was ever measured. If the digests part again the message
    # measures it, says where in the file it is and which part of the engine a fresh
    # process renders otherwise.
    wanted = np.array(open_signal(tmp_path / "one").frames)
    for _ in range(3):
        two = write_mix(tmp_path / "two", partial(_engine, 2), processes=2, scratch=tmp_path)
        assert two["frames"] == one["frames"] and two["render"]["processes"] == 2
        made = np.array(open_signal(tmp_path / "two").frames)
        assert hashlib.sha256(made.tobytes()).hexdigest() == two["sha256"]
        assert two["sha256"] == one["sha256"], (
            f"a fresh process is {off(made, wanted):.3e} of the peak from this one: "
            f"{_what_differs(made, wanted, 2)}"
        )
    assert not list(tmp_path.glob("carriers-*"))
