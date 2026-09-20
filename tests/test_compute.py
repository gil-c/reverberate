"""The device layer: shares cover the work once, and come back in order whatever ran them."""

from __future__ import annotations

import os
import sys

import numpy as np

from reverberate.compute import Devices


def test_the_shares_cover_every_item_once_in_order() -> None:
    for count in (1, 3, 8):
        shares = Devices.host(count).split(10)
        np.testing.assert_array_equal(np.concatenate(shares), np.arange(10))
    assert len(Devices.host(4).split(2)) == 2


def test_the_host_runs_every_share_and_keeps_their_order() -> None:
    devices = Devices.host(3)
    parts = devices.map(lambda card, share: (card, os.getpid(), share.tolist()), devices.split(7))
    assert [p[0] for p in parts] == [-1, -1, -1]
    assert [x for p in parts for x in p[2]] == list(range(7))
    if sys.platform.startswith("linux"):
        # A pool starts its processes as work arrives, so a quick one may take two shares.
        assert os.getpid() not in {p[1] for p in parts}


def test_one_share_runs_here() -> None:
    parts = Devices.host(1).map(lambda card, share: os.getpid(), [np.arange(3)])
    assert parts == [os.getpid()]


def test_forcing_the_host_hides_the_cards(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("REVERBERATE_NO_GPU", "1")
    assert not Devices.detect().gpu
