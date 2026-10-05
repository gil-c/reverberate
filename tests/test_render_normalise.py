"""A traced pack's two bands on one scale, once for all: the tool, its arithmetic, and back.

``docs/open-questions/chain-audit.md`` found what stood between a pack's
mirror and its wave band: an alignment gain 2.4 dB low with its signature
(D1), that signature's treble (D2), a seam that read a band limit as a
level (D7), and a directivity over its clip on the axis (D3). These hold
the tool that undoes them in a pack already traced
(``python -m reverberate.render normalise``): every number it writes, that
the engine renders what the arithmetic says, that asking twice is asking
once, and that ``--undo`` leaves the file as the trace wrote it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from reverberate.mirror.direct import signature_level_db, unit_signature
from reverberate.mirror.directivity import omni, radiated_db, voice_v1
from reverberate.render.__main__ import main
from reverberate.render.benchmark import density_pack
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.normalise import (
    KEPT,
    SEAM_READING_DB,
    normalise_pack,
    radiating,
    seam_weight,
)
from reverberate.render.pack import ScenePack, read_pack, write_pack
from reverberate.render.seam import FIXED, SEAM_CONSTANT_DB, taper_pack

FS = 48000
#: The first scene's own: the alignment's gain over the field's unit, -5.22 dB.
GAIN = 0.5483522979302524
OCTAVE = (1000.0 / np.sqrt(2.0), 1000.0 * np.sqrt(2.0))


def traced_pack(signature: np.ndarray | None = None, seed: int = 0) -> ScenePack:
    """A pack as one was traced before the normalisation: a gain, a signature, seams round 1.9 dB.

    The scalar of a step is the gain plus the seam of the pair it reads,
    as ``trace.level.step_levels`` writes it for a step that reads one.
    """
    pack = density_pack(duration_s=0.3, moving=True, bins=30, seed=seed)
    source = pack.sources["s1"]
    low = source.low
    assert low is not None
    rng = np.random.default_rng(5)
    seams = (1.9 + rng.uniform(-1.5, 1.5, low.seam_db.shape[0])).astype(np.float32)
    weight = seam_weight(
        np.asarray(source.audible, dtype=bool),
        np.asarray(low.pair),
        np.asarray(low.position_weight, dtype=float),
        np.asarray(low.cell),
        np.asarray(low.mode),
        np.asarray(pack.listener.position, dtype=float),
        np.asarray(pack.cells.position, dtype=float),
    )
    assert np.all(weight[np.asarray(source.audible, dtype=bool)] == pytest.approx(1.0))
    first = np.asarray(low.pair)[:, 0, 0]
    scalar = (20.0 * np.log10(GAIN) + seams[np.maximum(first, 0)]).astype(np.float32)
    taps = pack.mirror.signature if signature is None else signature
    return replace(
        pack,
        mirror=replace(pack.mirror, alignment_gain=GAIN, signature=np.asarray(taps, dtype=float)),
        sources={
            "s1": replace(
                source,
                low=replace(low, seam_db=seams),
                level=replace(source.level, high_gain_db=scalar),
            )
        },
    )


def held(path: Path) -> dict[str, str]:
    """Every dataset and attribute of the file, by name, as a digest: what a tool may change."""
    found: dict[str, str] = {}

    def digest(value: Any) -> str:
        data = np.asarray(value)
        # Text of any length is held by reference: its bytes are read, not its addresses.
        text = repr(data.tolist()).encode() if data.dtype == object else data.tobytes()
        return hashlib.sha256(text).hexdigest()

    def visit(name: str, node: Any) -> None:
        for key, value in node.attrs.items():
            found[f"{name}@{key}"] = digest(value)
        if isinstance(node, h5py.Dataset):
            found[name] = digest(node[...])

    with h5py.File(path, "r") as f:
        visit("", f)
        f.visititems(visit)
    return found


def test_a_traced_pack_is_put_on_the_physical_scale_in_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = traced_pack()
    target = write_pack(tmp_path / "pack.h5", pack)
    before = held(target)
    source = pack.sources["s1"]
    assert source.low is not None
    seams, scalar = np.asarray(source.low.seam_db), np.asarray(source.level.high_gain_db)
    a_db = 20.0 * np.log10(GAIN)
    s_db = signature_level_db(pack.mirror.signature, OCTAVE, FS)
    # A dry run says every number and writes none.
    assert main(["normalise", str(target), "--dry-run"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["written"] is False and said["changed"] is True
    assert held(target) == before
    assert main(["normalise", str(target)]) == 0
    said = json.loads(capsys.readouterr().out)
    record = said["normalisation"]
    assert (record["alignment"], record["signature"], record["seam"]) == (
        "physical",
        "unit",
        "unbiased",
    )
    assert record["born"] is False and record["kept"] == f"/{KEPT}"
    changed = record["changed"]
    assert changed["mirror/alignment_gain"]["was_db"] == pytest.approx(a_db, abs=1e-4)
    assert changed["mirror/signature"]["was_octave_db"] == pytest.approx(s_db, abs=1e-4)
    parts = changed["low/seam_db"]["of_which"]
    assert parts == {
        "alignment_gain_db": pytest.approx(a_db, abs=1e-4),
        "signature_octave_db": pytest.approx(s_db, abs=1e-4),
        "seam_reading_db": SEAM_READING_DB,
    }
    moved = a_db + s_db + SEAM_READING_DB
    assert changed["low/seam_db"]["added_db"] == pytest.approx(moved, abs=1e-4)
    assert changed["low/seam_db"]["median_is_db"] == pytest.approx(
        changed["low/seam_db"]["median_was_db"] + moved, abs=1e-3
    )
    assert changed["level/high_gain_db"]["added_db"] == pytest.approx(
        s_db + SEAM_READING_DB, abs=1e-4
    )
    assert changed["level/high_gain_db"]["steps_that_read_a_part_of_their_pairs"] == 0
    # The level a band is made again, on the code's one number: 0 dB over a gain of one.
    assert said["seam"]["constant"] == FIXED and said["seam"]["constant_db"] == SEAM_CONSTANT_DB
    assert said["seam"]["level_db"] == 0.0
    with read_pack(target, deep=True) as read:  # still a pack, and the engine's numbers
        assert read.mirror.alignment_gain == 1.0
        np.testing.assert_array_equal(read.mirror.signature, [1.0])
        assert read.mirror.lead_s == pack.mirror.lead_s
        made = read.sources["s1"]
        assert made.low is not None
        np.testing.assert_allclose(made.low.seam_db, seams + moved, atol=1e-5)
        heard = np.asarray(made.audible, dtype=bool)
        np.testing.assert_allclose(
            made.level.high_gain_db[heard], scalar[heard] + s_db + SEAM_READING_DB, atol=1e-5
        )
        np.testing.assert_array_equal(made.level.high_gain_db[~heard], scalar[~heard])
        # K is 0 dB absolute: from 4 kHz the mirror's direct sound is 1 / d. At 1 kHz a
        # step keeps how far its pairs stand from the scene's median, and no more.
        table = np.asarray(made.level.band_gain_db)
        assert np.all(table[heard, 5:] == 0.0)
        median = float(np.median(np.asarray(made.low.seam_db)))
        np.testing.assert_allclose(
            table[heard, 3], made.level.high_gain_db[heard] - median, atol=1e-4
        )
        assert read.header.provenance["normalisation"] == record
        voice = read.directivity["voice"]
        assert voice.normalised == "axis"
        np.testing.assert_array_equal(voice.gain_db[:, 0], 0.0)
        # The recipe's key names the pattern, whichever way its table is level.
        assert voice.digest == voice_v1().digest == voice_v1("mean").digest
    assert changed["directivity"]["tables"]["voice"]["radiates_db"] == pytest.approx(
        [-0.9617, -1.4141, -2.263, -3.0402, -4.0809, -5.2619, -6.2474], abs=1e-3
    )
    assert "omni" not in changed["directivity"]["tables"]
    # Asking twice is asking once: nothing is written, nothing is added twice.
    after = held(target)
    assert main(["normalise", str(target)]) == 0
    assert json.loads(capsys.readouterr().out)["changed"] is False
    assert held(target) == after
    # And back: every dataset and every attribute as the trace wrote it.
    assert main(["normalise", str(target), "--undo"]) == 0
    capsys.readouterr()
    assert held(target) == before
    assert normalise_pack(target, undo=True)["changed"] is False


def test_the_old_colour_and_the_old_directivity_stay_selectable(tmp_path: Path) -> None:
    pack = traced_pack()
    target = write_pack(tmp_path / "pack.h5", pack)
    before = held(target)
    normalise_pack(target)
    # Asked again another way, it is made from what the trace wrote, not from the last.
    said = normalise_pack(target, keep_signature=True, directivity="mean", seam_reading_db=0.0)
    assert said["changed"] is True and said["normalisation"]["signature"] == "colour"
    with read_pack(target, deep=True) as read:
        kept = unit_signature(pack.mirror.signature, OCTAVE, FS)
        np.testing.assert_allclose(read.mirror.signature, kept, atol=1e-12)
        assert signature_level_db(read.mirror.signature, OCTAVE, FS) == pytest.approx(0.0, abs=1e-9)
        assert read.mirror.alignment_gain == 1.0
        voice = read.directivity["voice"]
        assert voice.normalised == "mean"
        np.testing.assert_allclose(radiated_db(voice.gain_db), 0.0, atol=1e-5)
    assert "directivity" not in said["normalisation"]["changed"]
    assert said["normalisation"]["changed"]["low/seam_db"]["of_which"]["seam_reading_db"] == 0.0
    # A level a band the pack held before is made again as it was, on its own terms.
    normalise_pack(target, undo=True)
    assert held(target) == before
    taper_pack(target, taper=(1.0, 0.25), constant_db=1.5)
    tapered = held(target)
    normalise_pack(target)
    with h5py.File(target, "r") as f:
        seam = json.loads(f.attrs["provenance_json"])["seam"]
        assert seam["shares"] == [1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25]
        assert seam["constant"] == FIXED
    normalise_pack(target, undo=True)
    assert held(target) == tapered


def test_the_engine_renders_what_the_arithmetic_says(tmp_path: Path) -> None:
    """Under the scalar, a signature that is a level and a seam read right: not a sample moves."""
    level = 10.0 ** (2.77 / 20.0)
    pack = traced_pack(signature=np.array([level]))
    target = write_pack(tmp_path / "pack.h5", pack)
    dry = {"s1": np.random.default_rng(0).standard_normal(int(0.3 * FS))}
    scalar = RenderSettings(seam="broadband")
    with read_pack(target) as read:
        was = {part: Engine(read, dry, settings=scalar).stem("s1", parts=(part,)) for part in PARTS}
    normalise_pack(target, directivity="mean", seam_reading_db=0.0)
    with read_pack(target) as read:
        now = {part: Engine(read, dry, settings=scalar).stem("s1", parts=(part,)) for part in PARTS}
    # The mirror lost a signature of 2.77 dB and its scalar took 2.77 dB: the same samples.
    for part in ("early", "tail"):
        peak = float(np.abs(was[part]).max())
        assert peak > 0.0 and np.abs(now[part] - was[part]).max() < 1e-5 * peak
    np.testing.assert_array_equal(now["low"], was["low"])  # the wave band is not touched
    # With the seam's reading given back, the mirror is that much louder and no more.
    normalise_pack(target, directivity="mean")
    with read_pack(target) as read:
        louder = Engine(read, dry, settings=scalar).stem("s1", parts=("early",))
    gain = 10.0 ** (SEAM_READING_DB / 20.0)
    assert np.abs(louder - gain * was["early"]).max() < 1e-5 * float(np.abs(was["early"]).max())


PARTS = ("early", "low", "tail")


def band_noise(low_hz: float, high_hz: float, seconds: float = 0.3) -> np.ndarray:
    n = int(round(seconds * FS))
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    shape = np.clip((freqs - low_hz) / 50.0, 0, 1) * np.clip((high_hz - freqs) / 50.0, 0, 1)
    white = np.random.default_rng(3).standard_normal(n)
    return np.asarray(np.fft.irfft(np.fft.rfft(white) * shape, n) * np.hanning(n))


def test_the_late_part_of_a_voice_level_with_its_axis_is_what_it_radiates() -> None:
    """D3: the tail is omnidirectional; a table that is 0 dB ahead radiates its index less."""
    pack = density_pack(duration_s=0.3, moving=True, bins=30)
    source = pack.sources["s1"]
    assert source.directivity_enabled and pack.directivity["voice"].normalised == "mean"
    axis = replace(pack, directivity={"voice": voice_v1(), "omni": omni()})
    # A table of unit mean power radiates what the omni does: the pack itself, to the bit.
    assert radiating(pack, source, True) is pack
    assert radiating(axis, source, False) is axis
    lowered = radiating(axis, source, True)
    down = radiated_db(voice_v1().gain_db)
    np.testing.assert_allclose(
        np.asarray(lowered.mirror.tail_gain_db) - np.asarray(axis.mirror.tail_gain_db), down
    )
    dry = {"s1": band_noise(3600.0, 4400.0)}
    tail = {
        name: Engine(held_pack, dry).stem("s1", parts=("tail",))
        for name, held_pack in (("mean", pack), ("axis", axis))
    }
    fell = 10.0 * np.log10(np.sum(tail["axis"] ** 2) / np.sum(tail["mean"] ** 2))
    # The 4 kHz band: -5.26 dB. Measured -5.32: the late part's bands lean on their
    # neighbours through the bank's reading, which the next bands' -4.1 and -6.2 dB enter.
    assert fell == pytest.approx(down[5], abs=0.1)
    # With the directivity switched off the source is omnidirectional in both: one tail.
    off = RenderSettings(directivity=False)
    np.testing.assert_array_equal(
        Engine(axis, dry, settings=off).stem("s1", parts=("tail",)),
        Engine(pack, dry, settings=off).stem("s1", parts=("tail",)),
    )
    # And every arrival of the axis's table is the mean's less the band's index.
    early = {
        name: Engine(held_pack, dry).stem("s1", parts=("early",))
        for name, held_pack in (("mean", pack), ("axis", axis))
    }
    fell = 10.0 * np.log10(np.sum(early["axis"] ** 2) / np.sum(early["mean"] ** 2))
    assert fell == pytest.approx(down[5], abs=0.05)


def test_a_pack_it_does_not_understand_is_refused(tmp_path: Path) -> None:
    pack = traced_pack()
    target = write_pack(tmp_path / "pack.h5", pack)
    with h5py.File(tmp_path / "other.h5", "w") as f:
        f.attrs["schema"] = "something else"
    with pytest.raises(ValueError, match="not one the normalisation understands"):
        normalise_pack(tmp_path / "other.h5")
    with pytest.raises(ValueError, match="normalised on one of"):
        normalise_pack(target, directivity="front")
    # A pack born on the physical scale holds nothing to change, and nothing to put back.
    with h5py.File(target, "r+") as f:
        provenance = json.loads(f.attrs["provenance_json"])
        provenance["normalisation"] = {"alignment": "physical", "born": True}
        f.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
    before = held(target)
    assert normalise_pack(target)["changed"] is False and held(target) == before
    with pytest.raises(ValueError, match="traced on the physical scale"):
        normalise_pack(target, undo=True)
    # One that says it was normalised and no longer holds what the trace wrote.
    with h5py.File(target, "r+") as f:
        provenance["normalisation"] = {"alignment": "physical", "born": False}
        f.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
    with pytest.raises(ValueError, match="rewritten since"):
        normalise_pack(target)
