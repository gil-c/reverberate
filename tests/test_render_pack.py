"""The scene pack: what is written is read back, and what breaks the format is refused."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.render.__main__ import main
from reverberate.render.benchmark import density_pack
from reverberate.render.noise import carrier, threefry2x32, uniform
from reverberate.render.pack import (
    PackError,
    PackWriter,
    ScenePack,
    Source,
    path_id,
    read_pack,
    synthetic_free_field,
    tail_seed,
    validate,
    write_pack,
)


def same(a: Any, b: Any) -> None:
    """Two packs' tables agree, dataclass by dataclass, array by array."""
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            same(x, y)
        elif hasattr(x, "shape"):
            rows = [np.asarray(x[i]) for i in range(x.shape[0])]
            other = [np.asarray(y[i]) for i in range(y.shape[0])]
            assert len(rows) == len(other), field.name
            for r, o in zip(rows, other, strict=True):
                np.testing.assert_array_equal(r, o, err_msg=field.name)
        else:
            assert x == y, field.name


def with_source(pack: ScenePack, **changes: Any) -> ScenePack:
    name = next(iter(pack.sources))
    return dataclasses.replace(
        pack, sources={name: dataclasses.replace(pack.sources[name], **changes)}
    )


@pytest.mark.parametrize("level", ["A", "B"])
def test_a_synthetic_pack_is_read_back_as_it_was_written(tmp_path: Path, level: str) -> None:
    pack = synthetic_free_field(level=level, duration_s=0.3)
    with read_pack(write_pack(tmp_path / "pack.h5", pack), deep=True) as back:
        assert back.header == pack.header
        assert back.recipe == pack.recipe
        assert back.mirror.lead_s == 0.0 and back.mirror.lowcut_hz == 0.0
        np.testing.assert_array_equal(back.mirror.signature, [1.0])
        assert back.crossover == pack.crossover
        assert back.air == pack.air
        same(back.listener, pack.listener)
        same(back.cells, pack.cells)
        for name, source in pack.sources.items():
            same(back.sources[name], source)
        if level == "B":
            low = back.sources["s1"].low
            assert low is not None and not isinstance(low.ir, np.ndarray)  # left in the file


def test_the_synthetic_profile_is_the_one_the_format_describes() -> None:
    pack = synthetic_free_field(level="A", source=(3.0, 1.5, 4.0), duration_s=0.2)
    source = pack.sources["s1"]
    assert pack.header.profile == "synthetic-free-field"
    assert not pack.header.has_low and not pack.header.has_tail
    assert pack.header.steps == 5 and pack.header.samples == 4 * 2400
    np.testing.assert_array_equal(np.diff(source.early.offsets), 1)
    np.testing.assert_allclose(source.early.delay_s, 5.0 / 343.2)
    np.testing.assert_allclose(source.early.gain, 1.0 / 5.0, rtol=1e-6)
    np.testing.assert_allclose(source.early.arrival[0], [0.6, 0.0, 0.8], atol=1e-6)
    np.testing.assert_allclose(source.early.departure, -source.early.arrival)
    assert int(source.early.path_id[0]) == path_id(0)
    assert path_id(0) == int.from_bytes(hashlib.sha256(b"\x00").digest()[:8], "little")
    assert path_id(1, [3, 7]) != path_id(1, [7, 3])
    assert source.tail_seed == tail_seed(0, "s1")
    assert tail_seed(0, "s1") == int.from_bytes(hashlib.sha256(b"0:tail:s1").digest()[:8], "little")


def test_level_b_holds_two_cells_on_the_listeners_line_and_names_the_mode() -> None:
    walk = synthetic_free_field(
        level="B", listener_start=(0, 1.5, 0), listener_end=(0.4, 1.5, 0), duration_s=0.4, fuse=True
    )
    low = walk.sources["s1"].low
    assert low is not None
    np.testing.assert_allclose(walk.cells.position, [[0, 1.5, 0], [0.4, 1.5, 0]])
    assert list(low.mode) == [1, 3, 3, 3, 3, 3, 3, 3, 1]  # on a cell at both ends
    assert list(low.cell[4]) in ([0, 1], [1, 0]) and list(low.cell[6]) == [1, 0]
    assert list(low.cell[0]) == [0, -1]
    one = synthetic_free_field(level="B", listener_start=(0.1, 1.5, 0), cell_origin=(0, 1.5, 0))
    assert one.sources["s1"].low is not None
    assert set(one.sources["s1"].low.mode) == {2}


def test_a_dense_pack_goes_through_the_writer_a_source_at_a_time(tmp_path: Path) -> None:
    pack = density_pack(duration_s=0.3, sources=2, bins=20)
    first = write_pack(tmp_path / "first.h5", pack)
    # The trace's way: tables still in a file, copied by the writer row by row.
    with read_pack(first) as held:
        with PackWriter(
            tmp_path / "second.h5",
            held.header,
            held.recipe,
            held.listener,
            held.cells,
            mirror=held.mirror,
            crossover=held.crossover,
            air=held.air,
            directivity=held.directivity,
        ) as writer:
            for source in held.sources.values():
                writer.add_source(source)
        assert main(["validate", str(tmp_path / "second.h5"), "--deep"]) == 0
        with read_pack(tmp_path / "second.h5", deep=True) as again:
            assert list(again.sources) == ["s1", "s2"]
            for name in held.sources:
                same(again.sources[name], held.sources[name])
            assert again.directivity["voice"].digest == pack.directivity["voice"].digest
            assert again.mirror.signature.size == 128 and again.air.enabled


def broken(pack: ScenePack) -> list[tuple[str, ScenePack]]:
    source = pack.sources["s1"]
    early = source.early
    assert source.low is not None and source.tail is not None
    swapped = np.array(early.path_id)
    swapped[[0, 1]] = swapped[[1, 0]]
    off = np.array(early.offsets)
    off[-1] += 1
    silent = np.array(source.audible)
    silent[2] = False
    far = np.array(pack.listener.position)
    far[0, 0] += 0.5
    loud = np.array(np.asarray(source.low.ir[0]), dtype=np.float32)[None].repeat(
        source.low.pair_cell.shape[0], 0
    )
    loud[0, 0, ::2] += 1.0  # a comb: energy at the low rate's Nyquist
    weights = np.array(source.tail.cell_weight)
    weights[1] = 1.5
    part = dataclasses.replace
    return [
        ("sorted by path_id", with_source(pack, early=part(early, path_id=swapped))),
        ("not unit", with_source(pack, early=part(early, arrival=early.arrival * 1.01))),
        ("offsets", with_source(pack, early=part(early, offsets=off))),
        ("not audible has rows", with_source(pack, audible=silent)),
        ("delay_s", with_source(pack, early=part(early, delay_s=-early.delay_s))),
        ("recipe_sha256", part(pack, recipe=pack.recipe + b" ")),
        ("mode 1 with the head", part(pack, listener=part(pack.listener, position=far))),
        ("is not zero above", with_source(pack, low=part(source.low, ir=loud))),
        ("is not in [0, 1]", with_source(pack, tail=part(source.tail, cell_weight=weights))),
        ("has_tail", with_source(pack, tail=None)),
        ("profile", part(pack, header=part(pack.header, profile="guess"))),
        ("expected", with_source(pack, yaw_deg=source.yaw_deg[:-1])),
    ]


def test_a_pack_that_breaks_an_invariant_is_refused_and_told_which(tmp_path: Path) -> None:
    pack = density_pack(duration_s=0.3, moving=False, bins=20)
    validate(pack, deep=True)
    for message, bad in broken(pack):
        with pytest.raises(PackError, match=message.replace("[", r"\[")):
            validate(bad, deep=True)
    with pytest.raises(PackError):
        write_pack(tmp_path / "bad.h5", broken(pack)[0][1])
    assert not (tmp_path / "bad.h5").exists()


def test_a_source_must_be_held_under_its_own_name() -> None:
    pack = synthetic_free_field(duration_s=0.1)
    source: Source = dataclasses.replace(pack.sources["s1"], id="other")
    with pytest.raises(PackError, match="holds the source"):
        validate(dataclasses.replace(pack, sources={"s1": source}))


def test_the_generator_is_threefry_and_the_same_wherever_it_is_cut() -> None:
    # Random123's known answers for Threefry 2x32, twenty rounds.
    cases = [
        ((0, 0), (0, 0), (0x6B200159, 0x99BA4EFE)),
        ((0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF), (0x1CB996FC, 0xBB002BE7)),
        ((0x13198A2E, 0x03707344), (0x243F6A88, 0x85A308D3), (0xC4923A9C, 0x483DF7A0)),
    ]
    for key, counter, answer in cases:
        x0, x1 = threefry2x32(
            key, np.array([counter[0]], dtype=np.uint32), np.array([counter[1]], dtype=np.uint32)
        )
        assert (int(x0[0]), int(x1[0])) == answer
    whole = carrier(99, np.arange(4), 0, 5000)
    np.testing.assert_array_equal(carrier(99, [2], 1234, 100)[0], whole[2, 1234:1334])
    assert abs(float(whole.mean())) < 0.02 and float(whole.std()) == pytest.approx(1.0, abs=0.02)
    assert float(np.abs(np.corrcoef(whole)[0, 1])) < 0.05
    assert not np.array_equal(carrier(98, [0], 0, 100), carrier(99, [0], 0, 100))
    draws = uniform(99, [0, 1], 0, 4000)
    assert draws.min() > 0.0 and draws.max() < 1.0
    assert float(draws.mean()) == pytest.approx(0.5, abs=0.02)
