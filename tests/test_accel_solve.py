"""The local engine driver's bookkeeping: the job directory and the slices."""

from __future__ import annotations

from pathlib import Path

import pytest

from reverberate.accel.solve import prepare_job, slices_for


def an_entry(tmp_path: Path) -> Path:
    entry = tmp_path / "entry"
    entry.mkdir()
    for name in ("sim_consts.h5", "vox_out.h5", "sim_mats.h5", "cart_grid.h5"):
        (entry / name).write_bytes(name.encode())
    return entry


class TestPrepareJob:
    def test_the_entry_s_files_are_linked_and_the_comms_kept(self, tmp_path: Path) -> None:
        entry = an_entry(tmp_path)
        job = tmp_path / "job"
        job.mkdir()
        (job / "comms_out.h5").write_bytes(b"comms of this run")
        prepare_job(job, entry, job / "comms_out.h5")
        assert (job / "comms_out.h5").read_bytes() == b"comms of this run"
        for name in ("sim_consts.h5", "vox_out.h5", "sim_mats.h5"):
            assert (job / name).read_bytes() == name.encode()
        assert not (job / "cart_grid.h5").exists()

    def test_a_comms_file_from_elsewhere_is_linked_in(self, tmp_path: Path) -> None:
        entry = an_entry(tmp_path)
        elsewhere = tmp_path / "comms_out.h5"
        elsewhere.write_bytes(b"elsewhere")
        job = prepare_job(tmp_path / "job", entry, elsewhere)
        assert (job / "comms_out.h5").read_bytes() == b"elsewhere"

    def test_a_missing_comms_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prepare_job(tmp_path / "job", an_entry(tmp_path), tmp_path / "nope.h5")


class TestSlices:
    def test_the_mid_band_of_hssd_0076_on_a_183_gb_host_is_one_slice(self) -> None:
        assert slices_for(103_969_949_248, 183.0) == 2
        assert slices_for(74_635_723_008, 183.0) == 1
        assert slices_for(103_969_949_248, 400.0) == 1


class TestCards:
    def test_the_grid_must_fit_the_cards_and_the_receivers_do_not_count(self) -> None:
        from reverberate.accel.solve import grid_fits_cards

        # The 8 kHz storey over four 32 GB cards: 20.3 GB a card plus the overhead.
        fits, why = grid_fits_cards(8.98e9, [34_073_000_000] * 4)
        assert fits and "22.4 GB a card" in why
        # The same grid on one 24 GB card does not.
        assert not grid_fits_cards(8.98e9, [25_760_000_000])[0]
        assert grid_fits_cards(8.98e9, [])[0]


class TestResumption:
    def test_a_solved_slice_is_consumed_and_an_encoded_one_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that failed downstream must not solve again what is on disk."""
        import numpy as np

        from reverberate.accel import solve as module

        entry = an_entry(tmp_path)
        import h5py

        with h5py.File(entry / "sim_consts.h5", "w") as handle:
            handle.create_dataset("SR", data=np.float64(1000.0))
        positions = np.zeros((40, 3))
        rows: list[list[int] | None] = [[i * 10, (i + 1) * 10] for i in range(4)]
        job_root = tmp_path / "jobs"
        # Slice 1 of 2 was solved: its comms, its log at 100 %, its output of the right size.
        part1 = job_root / "part1"
        part1.mkdir(parents=True)
        (part1 / "comms_out.h5").write_bytes(b"c")
        # As the engine writes it: the progress line, then a dump of every
        # receiver's last samples, then its last words.
        (part1 / "engine.log").write_text(
            "Running [ 99.0%]\rRunning [100.0%]\n"
            + "".join(f"receiver {r}\nsample 1: 0.0\n" for r in range(3000))
            + "sim data freed\n--Date and time: Mon Sep 14 11:36:47 2026\n"
        )
        (part1 / "sim_outs.h5").write_bytes(b"x" * 100)
        engines: list[int] = []
        consumed: list[tuple[int, str]] = []
        monkeypatch.setattr(module, "write_comms", lambda *a, **k: None)

        def fake_engine(job_dir: Path, **kwargs: object) -> module.EngineResult:
            engines.append(1)
            return module.EngineResult(job_dir / "sim_outs.h5", 1.0, job_dir / "engine.log", 100)

        monkeypatch.setattr(module, "run_engine", fake_engine)
        monkeypatch.setattr(module, "prepare_job", lambda job_dir, entry_path, comms: job_dir)

        def fake_consume(k: int, a: int, b: int, out: Path, comms: Path) -> dict[str, int]:
            consumed.append((k, out.name))
            return {"k": k}

        real_check = module.pressure_complete
        monkeypatch.setattr(
            module, "pressure_complete", lambda output, log, expected: real_check(output, log, 50.0)
        )
        records = module.solve_slices(
            job_root=job_root,
            entry_path=entry,
            source_position=np.zeros(3),
            positions=positions,
            rows=rows,
            duration_s=0.01,
            output_bytes=20e9,
            ram_gb=30.0,  # 2.2 x 20 GB over 22 GB: two slices
            pffdtd_dir=tmp_path,
            devices=None,
            consume=fake_consume,
            already_done=lambda k, a, b: k == 0,
            card_bytes=[],
        )
        assert [r["slice"] for r in records] == [0, 1]
        assert records[0].get("skipped") is True
        assert records[1].get("reused") is True
        assert engines == []
        assert consumed == [(1, "sim_outs.h5")]


class TestOutputSampleBytes:
    def test_the_patched_engine_writes_four_bytes_and_upstream_eight(self, tmp_path: Path) -> None:
        from reverberate.accel.solve import output_sample_bytes

        header = tmp_path / "c_cuda" / "fdtd_data.h"
        header.parent.mkdir()
        header.write_text("   double *u_out; //for output signals\n")
        assert output_sample_bytes(tmp_path) == 8
        header.write_text("   Real *u_out; //REVERBERATE PATCH 8\n")
        assert output_sample_bytes(tmp_path) == 4
        assert output_sample_bytes(tmp_path, double_precision=True) == 8
        assert output_sample_bytes(tmp_path / "nowhere") == 8

    def test_the_patch_file_carries_its_mark_for_every_engine(self) -> None:
        patch = (
            Path(__file__).parents[1]
            / "scripts"
            / "pffdtd"
            / "0008-receiver-records-in-the-engines-own-precision.patch"
        )
        text = patch.read_text()
        for name in ("fdtd_data.h", "gpu_engine.h", "cpu_engine.h"):
            assert f"+++ b/c_cuda/{name}" in text
        assert text.count("REVERBERATE PATCH 8") >= 10
        assert "H5T_REAL" in text and "-   double *u_out; //for output signals" in text


class TestRunEngine:
    """The engine as a shell script: a run that finishes, and one that stalls and is killed."""

    def a_job(self, tmp_path: Path) -> Path:
        job = tmp_path / "job"
        job.mkdir()
        for name in ("sim_consts.h5", "vox_out.h5", "sim_mats.h5", "comms_out.h5"):
            (job / name).write_bytes(b"x")
        return job

    def a_binary(self, tmp_path: Path, script: str) -> Path:
        binary = tmp_path / "c_cuda" / "fdtd_main_gpu_single.x"
        binary.parent.mkdir(exist_ok=True)
        binary.write_text("#!/bin/sh\n" + script)
        binary.chmod(0o755)
        return tmp_path

    def test_a_finished_run_reports_its_output_and_its_progress(self, tmp_path: Path) -> None:
        from reverberate.accel.solve import run_engine

        pffdtd = self.a_binary(
            tmp_path,
            '/usr/bin/printf "Running [ 50.0%%]\\r"; sleep 0.2;'
            ' /usr/bin/printf "Running [100.0%%]\\n";'
            ' /usr/bin/printf "pressure" > sim_outs.h5; echo "sim data freed"\n',
        )
        seen: list[float | None] = []
        result = run_engine(
            self.a_job(tmp_path),
            pffdtd_dir=pffdtd,
            poll_s=0.05,
            on_progress=lambda p, t: seen.append(p),
        )
        assert result.output_bytes == 8 and result.log.read_text().endswith("sim data freed\n")
        assert 50.0 in seen or 100.0 in seen

    def test_a_run_whose_progress_and_output_stop_moving_is_killed(self, tmp_path: Path) -> None:
        from reverberate.accel.solve import run_engine

        pffdtd = self.a_binary(tmp_path, '/usr/bin/printf "Running [ 10.0%%]\\r"; sleep 30\n')
        with pytest.raises(RuntimeError, match="stalled at 10.0%"):
            run_engine(self.a_job(tmp_path), pffdtd_dir=pffdtd, poll_s=0.05, stall_polls=20)

    def test_a_failing_engine_shows_the_end_of_its_log(self, tmp_path: Path) -> None:
        from reverberate.accel.solve import run_engine

        pffdtd = self.a_binary(tmp_path, 'echo "GPUassert: out of memory"; exit 3\n')
        with pytest.raises(RuntimeError, match=r"(?s)exited with 3 and no output.*out of memory"):
            run_engine(self.a_job(tmp_path), pffdtd_dir=pffdtd, poll_s=0.05)

    def test_a_missing_binary_says_how_to_build_it(self, tmp_path: Path) -> None:
        from reverberate.accel.solve import run_engine

        with pytest.raises(FileNotFoundError, match="build_pffdtd"):
            run_engine(self.a_job(tmp_path), pffdtd_dir=tmp_path)
