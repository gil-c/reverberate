"""The app's configuration file: found from a worktree, paths made absolute."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reverberate.viz.walk_config import CONFIG_NAME, find_config, load_config, main_checkout


def test_paths_are_resolved_against_the_file(tmp_path: Path) -> None:
    (tmp_path / CONFIG_NAME).write_text(
        'hssd_root = "raw/hssd"\ndata_root = "data"\nrun = "mock"\n'
        "port = 9000\nopen_browser = false\n"
    )
    config = load_config(tmp_path / CONFIG_NAME)

    assert config.hssd_root == (tmp_path / "raw" / "hssd").resolve()
    assert config.data_root == (tmp_path / "data").resolve()
    assert config.run == "mock" and config.port == 9000 and config.open_browser is False
    assert config.scene is None and config.measured_head is None


def test_a_file_without_hssd_root_is_refused(tmp_path: Path) -> None:
    (tmp_path / CONFIG_NAME).write_text("port = 1\n")
    with pytest.raises(ValueError, match="hssd_root"):
        load_config(tmp_path / CONFIG_NAME)


def test_a_worktree_finds_the_main_checkouts_file(tmp_path: Path) -> None:
    """Every agent works in a worktree; the machine's paths live once, at the
    root of the main checkout, unversioned."""
    main = tmp_path / "main"
    main.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main"], cwd=main, check=True)
    (main / "a").write_text("a")
    subprocess.run([*git, "add", "a"], cwd=main, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "a"], cwd=main, check=True)
    worktree = tmp_path / "wt"
    subprocess.run([*git, "worktree", "add", "-q", str(worktree), "-b", "wt"], cwd=main, check=True)
    (main / CONFIG_NAME).write_text('hssd_root = "raw"\n')

    assert main_checkout(worktree) == main.resolve()
    assert find_config(start=worktree) == main / CONFIG_NAME
    assert find_config(start=main) == main / CONFIG_NAME
    (worktree / CONFIG_NAME).write_text('hssd_root = "raw"\n')
    assert find_config(start=worktree) == worktree / CONFIG_NAME
