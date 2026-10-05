"""The low band in fewer bytes: what each lever keeps, and that no lever is no change.

A response here is what a room gives a cell: 64 channels of noise that
decays, under the crossover's low mask. The codec is checked on it lever by
lever, then through the pack's file and through the engine on the
synthetic free field, whose head is read from one cell and from two.
"""

from __future__ import annotations

import filecmp
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.special import spherical_jn

from reverberate.mirror.hybrid import Crossover
from reverberate.render.__main__ import main
from reverberate.render.compact import (
    BLOCK_HZ,
    RAMP_HZ,
    CompactIr,
    Levers,
    compact_pack,
    cut_hz,
    decode,
    degree_weights,
    encode,
    first_hz,
    pass_hz,
)
from reverberate.render.engine import Engine
from reverberate.render.pack import (
    PackWriter,
    ScenePack,
    read_pack,
    synthetic_free_field,
    write_pack,
)
from reverberate.settings import data_root
from reverberate.spatial.sh import degrees_of

C = 343.2
RATE = 4000.0
TOP = Crossover().band_hz()[1]
ALL = "bins,int16,degree=40,decay=60"


def room_response(seed: int = 0, t60_s: float = 0.35, samples: int = 4800) -> np.ndarray:
    """64 channels that arrive at 15 ms and fall 60 dB in ``t60_s``, under the low mask."""
    rng = np.random.default_rng(seed)
    start = 60
    decay = np.zeros(samples)
    decay[start:] = 10.0 ** (-3.0 * np.arange(samples - start) / (t60_s * RATE))
    noise = rng.standard_normal((64, samples)) * decay
    mask, _ = Crossover().masks(samples, RATE, power=True)
    return np.asarray(np.fft.irfft(np.fft.rfft(noise, axis=-1) * mask, n=samples, axis=-1))


def kept_and_back(x: np.ndarray, text: str) -> tuple[np.ndarray, int, np.ndarray]:
    levers = Levers.parse(text)
    kept = encode(x, levers, rate_hz=RATE, top_hz=TOP, sound_speed_m_s=C)
    back = decode(
        kept,
        first_hz=first_hz(7, levers.degree_db, C),
        top_hz=TOP,
        rate_hz=RATE,
        samples=x.shape[-1],
    )
    scales = kept.scale.nbytes if levers.sample == "int16" else 0
    return back.astype(float), int(kept.data.nbytes) + scales, kept.samples


def level_db(x: np.ndarray, reference: np.ndarray) -> float:
    return float(10.0 * np.log10(np.sum(x**2) / np.sum(reference**2) + 1e-300))


def on_head(signal: np.ndarray, rate: float, radius_m: float = 0.10) -> np.ndarray:
    """An order 7 signal as a sphere of the head's radius holds it: ``j_n(k a)`` a channel."""
    freqs = np.fft.rfftfreq(signal.shape[-1], 1.0 / rate)
    weight = spherical_jn(degrees_of(7)[:, None], (2 * np.pi * freqs / C * radius_m)[None, :])
    return np.asarray(
        np.fft.irfft(np.fft.rfft(signal, axis=-1) * weight, signal.shape[-1], axis=-1)
    )


# --------------------------------------------------------------------------
# the levers, one at a time
# --------------------------------------------------------------------------


def test_levers_are_read_from_a_line_and_refuse_what_they_cannot_be() -> None:
    assert Levers.parse("") == Levers.parse("none") == Levers() and Levers().off
    levers = Levers.parse("bins, int16, degree=40, decay=60")
    assert levers == Levers(sample="int16", bins=True, degree_db=40.0, decay_db=60.0)
    assert not levers.off and Levers(**levers.record()) == levers
    for wrong in ("int16", "degree=40", "decay=60", "bins,float16", "bins,shorter"):
        with pytest.raises(ValueError):
            Levers.parse(wrong)


def test_the_bins_alone_are_the_response() -> None:
    """Nothing lies above the ramp's top, so the bins under it give every sample back."""
    x = room_response()
    back, size, lengths = kept_and_back(x, "bins")
    assert np.abs(back - x).max() < 1e-6 * np.abs(x).max()
    assert lengths.tolist() == [4800] * 8
    # 1699 complex bins of float32 for 4800 samples: 1.41 times fewer bytes.
    assert size == 64 * 1699 * 2 * 4 and 1.41 < 64 * 4800 * 4 / size < 1.42


def test_sixteen_bits_leave_their_noise_under_the_decay() -> None:
    """One scale a channel and a block: 90 dB under the response, and under its end."""
    x = room_response(t60_s=0.5)
    back, size, _ = kept_and_back(x, "bins,int16")
    assert level_db(back - x, x) < -90.0
    assert size == 64 * 1699 * 2 * 2 + 64 * (int(TOP // BLOCK_HZ) + 2) * 4
    # The response has fallen 60 to 72 dB between 0.5 and 0.6 s; what the 16 bits added
    # is 30 dB under it there (measured: -35 dB), and meets it only 110 dB down.
    span = slice(2060, 2460)
    assert level_db((back - x)[:, span], x[:, span]) < -30.0
    # A channel a thousand times louder than the rest does not raise the others' noise.
    loud = x.copy()
    loud[40] *= 1000.0
    back, _, _ = kept_and_back(loud, "bins,int16")
    assert level_db((back - loud)[:40], loud[:40]) < -90.0


def test_a_degree_enters_where_its_share_within_reach_does() -> None:
    whole = pass_hz(7, 40.0, C)
    assert np.allclose(whole, [0, 3.2, 47.2, 133.8, 242.6, 362.6, 488.9, 618.7], atol=0.1)
    assert not pass_hz(7, 0.0, C).any()
    # At its frequency a degree's share of a plane wave's energy is under the level asked
    # for, at the reach: the bound is that level, and the degree itself a little less.
    n = np.arange(1, 8)
    x = 2 * np.pi * whole[1:] / C * 0.30
    bound = (2 * n + 1) * (x**n / np.cumprod(2.0 * n + 1.0)) ** 2
    assert np.allclose(10 * np.log10(bound), -40.0, atol=1e-6)
    share = (2 * n + 1) * spherical_jn(n, x) ** 2
    assert np.all(share <= bound) and np.all(10 * np.log10(share) > -43.5)
    # The lever leaves whole a degree that would enter under 100 Hz, and gives the
    # others a ramp 100 Hz wide, kept from 15 Hz under its foot.
    cut = cut_hz(7, 40.0, C)
    assert not cut[:3].any() and np.array_equal(cut[3:], whole[3:])
    assert np.allclose(first_hz(7, 40.0, C), [0, 0, 0, 18.8, 127.6, 247.6, 373.9, 503.7], atol=0.1)
    assert not first_hz(7, 0.0, C).any()
    freqs = np.array([0.0, 400.0, 518.7, 568.7, 618.7, 1400.0])
    assert np.allclose(degree_weights(freqs, cut)[7], [0, 0, 0, 0.5, 1, 1], atol=0.01)
    assert np.all(degree_weights(freqs, cut)[:3] == 1.0)
    x = room_response()
    back, size, _ = kept_and_back(x, "bins,degree=40")
    spectrum, kept = np.fft.rfft(x, axis=-1), np.fft.rfft(back, axis=-1)
    at = np.fft.rfftfreq(4800, 1 / RATE)
    degree = degrees_of(7)
    for d in range(8):
        above = at >= cut[d]
        # Whole to the float32 for a degree with no ramp; with one, to what its ramp rang
        # before the window's start, which is not kept: 1e-3 of the spectrum's peak here.
        gap = np.abs(kept[degree == d][:, above] - spectrum[degree == d][:, above]).max()
        assert gap < (1e-6 if cut[d] == 0.0 else 3e-3) * np.abs(spectrum).max()
        foot = cut[d] - RAMP_HZ
        under = np.abs(kept[degree == d][:, at < foot - 1.0])
        assert under.size == 0 or under.max() < 3e-3 * np.abs(spectrum).max()
        gone = np.abs(kept[degree == d][:, at < first_hz(7, 40.0, C)[d] - 1.0])
        assert gone.size == 0 or gone.max() < 1e-6 * np.abs(spectrum).max()
    assert 1.65 < 64 * 4800 * 4 / size < 1.75


def test_a_degree_is_cut_where_it_has_decayed_and_stays_under_the_ramp() -> None:
    """Fallen 60 dB under the pressure's loudest moment, a degree ends; it is still band limited."""
    x = room_response(t60_s=0.35)
    back, size, lengths = kept_and_back(x, "bins,decay=60")
    # 60 dB in 0.35 s after an arrival at 15 ms, the fade, and the ladder of 50 ms: 0.40 s
    # for the pressure, and 0.35 s for the degrees a head within reach hears less of.
    assert lengths.tolist() == [1600] * 6 + [1400] * 2
    assert size < 0.34 * 64 * 1699 * 2 * 4
    assert level_db(back - x, x) < -55.0
    # After the cut there is only what keeps the response under the ramp: 2e-4 of its peak.
    assert np.abs(back[:, 2000:]).max() < 5e-4 * np.abs(x).max()
    spectrum = np.abs(np.fft.rfft(back, axis=-1))
    above = np.fft.rfftfreq(4800, 1 / RATE) > TOP * 1.001
    assert spectrum[:, above].max() <= 1e-6 * spectrum.max()
    # A degree that never comes within 60 dB of the pressure is not kept at all.
    faint = x.copy()
    faint[degrees_of(7) == 7] *= 1e-4
    _, _, lengths = kept_and_back(faint, "bins,decay=60")
    assert lengths[7] == 0 and lengths[6] > 0


# --------------------------------------------------------------------------
# the pack's file
# --------------------------------------------------------------------------


def moved_pack(fuse: bool = False) -> ScenePack:
    return synthetic_free_field(
        level="B",
        source=(0.5, 1.5, 3.0),
        listener_start=(0.2 if fuse else 0.1, 1.5, 0),
        cell_origin=(0, 1.5, 0),
        duration_s=0.5,
        fuse=fuse,
    )


def test_no_lever_is_the_pack_s_own_bytes(tmp_path: Path) -> None:
    pack = moved_pack()
    first = write_pack(tmp_path / "a.h5", pack)
    record = compact_pack(first, tmp_path / "b.h5", Levers())
    assert filecmp.cmp(first, tmp_path / "b.h5", shallow=False)
    assert record["low_bytes"]["before"] == record["low_bytes"]["after"] == 2 * 64 * 4800 * 4
    # The writer with no lever, or with none said, writes what it always wrote.
    with PackWriter(
        tmp_path / "c.h5",
        pack.header,
        pack.recipe,
        pack.listener,
        pack.cells,
        mirror=pack.mirror,
        crossover=pack.crossover,
        air=pack.air,
        directivity=pack.directivity,
        low_levers=Levers(),
    ) as writer:
        for source in pack.sources.values():
            writer.add_source(source)
    assert filecmp.cmp(first, tmp_path / "c.h5", shallow=False)
    with pytest.raises(ValueError):
        compact_pack(first, first, Levers.parse("bins"))


def test_a_compact_pack_reads_as_a_pack_does(tmp_path: Path) -> None:
    """``low/compact`` in place of ``low/ir``; the reader gives rows, the writer keeps the form."""
    first = write_pack(tmp_path / "a.h5", moved_pack(fuse=True))
    record = compact_pack(first, tmp_path / "b.h5", Levers.parse(ALL))
    assert record["low_bytes"]["after"] < record["low_bytes"]["before"] / 3.3
    assert record["pack_bytes"]["after"] < record["pack_bytes"]["before"]
    with h5py.File(tmp_path / "b.h5", "r") as f:
        low = f["sources/s1/low"]
        assert "ir" not in low and low["compact"].attrs["format"] == "bins/1"
        assert json.loads(low["compact"].attrs["levers_json"])["degree_db"] == 40.0
        assert low["compact/data"].dtype == np.int16
    with read_pack(first) as a, read_pack(tmp_path / "b.h5", deep=True) as b:
        plain, held = a.sources["s1"].low.ir, b.sources["s1"].low.ir  # type: ignore[union-attr]
        assert isinstance(held, CompactIr) and held.shape == plain.shape == (2, 64, 4800)
        assert held.dtype == np.float32 and len(held) == 2
        row = held[1]
        assert row.shape == (64, 4800) and row.dtype == np.float32
        assert np.array_equal(held[-1], row) and np.array_equal(held[1, 0], row[0])
        assert np.array_equal(held[0:2][1], row) and held[2:].shape == (0, 64, 4800)
        with pytest.raises(IndexError):
            held[2]
        # Channel 0 is whole at every frequency and is kept to the 16 bits.
        assert level_db(row[0] - plain[1][0], plain[1][0]) < -80.0
        # Written again as it is, the pack keeps its form and its bytes of low band.
        write_pack(tmp_path / "c.h5", b)
    with read_pack(tmp_path / "c.h5") as c:
        again = c.sources["s1"].low.ir  # type: ignore[union-attr]
        assert isinstance(again, CompactIr) and np.array_equal(again[1], row)
    # And with no lever it is given its ``low/ir`` back.
    compact_pack(tmp_path / "b.h5", tmp_path / "d.h5", Levers())
    with read_pack(tmp_path / "d.h5") as d:
        back = d.sources["s1"].low.ir  # type: ignore[union-attr]
        assert isinstance(back, h5py.Dataset) and np.array_equal(back[1], row)


def test_a_compact_group_another_reader_wrote_is_refused(tmp_path: Path) -> None:
    first = write_pack(tmp_path / "a.h5", moved_pack())
    compact_pack(first, tmp_path / "b.h5", Levers.parse("bins"))
    with h5py.File(tmp_path / "b.h5", "r+") as f:
        f["sources/s1/low/compact"].attrs["format"] = "bins/2"
    with pytest.raises(ValueError, match="bins/2"):
        read_pack(tmp_path / "b.h5")


# --------------------------------------------------------------------------
# through the engine
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fuse", [False, True])
def test_the_engine_s_output_moves_by_no_more_than_each_lever_says(
    tmp_path: Path, fuse: bool
) -> None:
    """One cell translated, two fused: the low stem with each lever against the stem without."""
    first = write_pack(tmp_path / "a.h5", moved_pack(fuse))
    dry = np.random.default_rng(0).standard_normal(24000)
    with read_pack(first) as a:
        want = Engine(a, {"s1": dry}).stem("s1", parts=("low",))
    peak = np.abs(want).max()
    stems = {}
    for name, text in (("bins", "bins"), ("int16", "bins,int16"), ("all", ALL)):
        compact_pack(first, tmp_path / f"{name}.h5", Levers.parse(text))
        with read_pack(tmp_path / f"{name}.h5") as b:
            stems[name] = Engine(b, {"s1": dry}).stem("s1", parts=("low",))
    # The bins are the response: the engine's own rounding. Sixteen bits: 1e-5 of the peak,
    # under the 3.2e-5 two devices are allowed on this part.
    assert np.abs(stems["bins"] - want).max() < 1e-7 * peak
    assert np.abs(stems["int16"] - want).max() < 3.2e-5 * peak
    # A degree let go under its frequency is gone from the output's own channels too, and
    # they are not small: the bound is on what a head hears. On a sphere of 0.10 m round a
    # head 0.10 or 0.20 m from its cell, the lever's own level, 40 dB at 0.30 m.
    # Measured: -73 and -55 dB on the pressure, -47 and -46 dB on the head's sphere.
    gone = stems["all"] - want
    assert level_db(gone[0], want[0]) < -50.0
    assert level_db(on_head(gone, 48000.0), on_head(want, 48000.0)) < -40.0
    assert level_db(gone, want) > -20.0


def test_the_first_real_pack_is_heard_the_same(tmp_path: Path) -> None:
    """``v1_near``, a voice 2 m from the cell its listener is read from, with every lever."""
    real = data_root() / "runs" / "w45_clarify_scene" / "v1_near" / "pulled" / "pack.h5"
    if not real.is_file():
        pytest.skip("the first pack is not on this machine")
    record = compact_pack(real, tmp_path / "all.h5", Levers.parse(ALL))
    assert record["low_bytes"]["before"] / record["low_bytes"]["after"] > 4.0
    dry = np.random.default_rng(0).standard_normal(48000)
    with read_pack(real) as a, read_pack(tmp_path / "all.h5", deep=True) as b:
        want = Engine(a, {"s1": dry}).stem("s1", 0, 96000, parts=("low",))
        got = Engine(b, {"s1": dry}).stem("s1", 0, 96000, parts=("low",))
    assert level_db(got[0] - want[0], want[0]) < -40.0
    assert level_db(on_head(got - want, 48000.0), on_head(want, 48000.0)) < -40.0


def test_a_pair_is_decoded_in_a_few_milliseconds() -> None:
    """Measured on the laptop: 2 ms a pair with every lever; the bound is for a busy machine."""
    x = room_response()
    levers = Levers.parse(ALL)
    kept = encode(x, levers, rate_hz=RATE, top_hz=TOP, sound_speed_m_s=C)
    starts = first_hz(7, levers.degree_db, C)
    started = time.perf_counter()
    for _ in range(5):
        decode(kept, first_hz=starts, top_hz=TOP, rate_hz=RATE, samples=4800)
    assert (time.perf_counter() - started) / 5 < 0.1


def test_the_command_rewrites_a_pack(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = write_pack(tmp_path / "a.h5", moved_pack())
    assert main(["compact", str(first), str(tmp_path / "b.h5"), "--levers", "bins,int16"]) == 0
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["levers"]["sample"] == "int16"
    assert 2.7 < record["low_bytes"]["before"] / record["low_bytes"]["after"] < 2.9
    assert main(["validate", str(tmp_path / "b.h5"), "--deep"]) == 0
