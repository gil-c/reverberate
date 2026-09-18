"""Where a stage computes: the host's cores, one card, or every card.

``cupy`` when a CUDA device answers, ``numpy`` otherwise; ``REVERBERATE_NO_GPU=1``
forces numpy on a machine that has a card. :class:`Devices` names what a stage
may use and runs its shares: one thread per card; on the host one forked
process per core on Linux, one thread per core elsewhere (macOS's
Accelerate does not survive a fork). Results come back in share order, so the
answer does not depend on how many devices computed it.

**Kernels are compiled without fused multiply-add** (``-fmad=false``): numpy
rounds ``a*b+c`` twice and a kernel that must match it bit for bit cannot
fuse. :func:`raw_kernel` is the only way a kernel is built.
"""

from __future__ import annotations

import functools
import multiprocessing
import os
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

__all__ = [
    "Devices",
    "KERNEL_OPTIONS",
    "NO_GPU_ENV",
    "cuda_available",
    "device_report",
    "raw_kernel",
    "to_numpy",
    "xp_for",
]

#: nvcc options for every custom kernel. See the module docstring.
KERNEL_OPTIONS = ("-fmad=false", "--std=c++14")

#: Set to ``1`` to run the numpy twin on a machine that has a card.
NO_GPU_ENV = "REVERBERATE_NO_GPU"


def cuda_available() -> bool:
    """Whether ``cupy`` imports and sees at least one device."""
    if os.environ.get(NO_GPU_ENV) == "1":
        return False
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - no cupy, no driver, no card: all mean numpy
        return False


def xp_for(gpu: bool | None = None) -> Any:
    """The array module a stage computes with: ``cupy`` or ``numpy``.

    ``gpu=None`` asks the machine; ``True`` insists and raises when there is
    no card, so a campaign that was promised a card does not quietly run on
    the host's cores for hours.
    """
    if gpu is None:
        gpu = cuda_available()
    if not gpu:
        return np
    import cupy

    if int(cupy.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("a GPU was asked for and cupy sees no CUDA device")
    return cupy


def to_numpy(array: Any) -> np.ndarray:
    """A host copy of ``array`` whichever module made it."""
    if hasattr(array, "get"):
        return np.asarray(array.get())
    return np.asarray(array)


def device_report() -> dict[str, Any]:
    """What card this process would compute on, for provenance and for the log."""
    if not cuda_available():
        return {"gpu": None, "backend": "numpy", "numpy": np.__version__}
    import cupy

    device = cupy.cuda.Device()
    properties = cupy.cuda.runtime.getDeviceProperties(device.id)
    free, total = device.mem_info
    return {
        "gpu": properties["name"].decode()
        if isinstance(properties["name"], bytes)
        else str(properties["name"]),
        "backend": "cupy",
        "cupy": cupy.__version__,
        "numpy": np.__version__,
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "vram_gb": round(total / 1e9, 2),
        "vram_free_gb": round(free / 1e9, 2),
        "cuda_runtime": int(cupy.cuda.runtime.runtimeGetVersion()),
        "driver": int(cupy.cuda.runtime.driverGetVersion()),
        "devices": int(cupy.cuda.runtime.getDeviceCount()),
    }


@functools.cache
def raw_kernel(source: str, name: str) -> Any:
    """Compile ``name`` from ``source`` once per process, under :data:`KERNEL_OPTIONS`."""
    import cupy

    return cupy.RawKernel(source, name, options=KERNEL_OPTIONS)


T = TypeVar("T")

#: The work of the share being run in a forked core process; see :meth:`Devices.map`.
_FORKED: list[Any] = []


def _run_forked(index: int) -> Any:
    work, shares = _FORKED
    return work(-1, shares[index])


@dataclass(frozen=True)
class Devices:
    """The cards a stage computes on, or, with none, the host's cores.

    ``cards`` are CUDA device ordinals; empty means the host, where ``cores``
    processes share the work.
    """

    cards: tuple[int, ...] = ()
    cores: int = 1

    @classmethod
    def detect(cls, cores: int | None = None) -> Devices:
        """Every card cupy sees; the host's cores when there is none."""
        cores = cores or max(1, (os.cpu_count() or 2) - 1)
        if not cuda_available():
            return cls((), cores)
        import cupy

        return cls(tuple(range(int(cupy.cuda.runtime.getDeviceCount()))), cores)

    @classmethod
    def host(cls, cores: int | None = None) -> Devices:
        """The host's cores, even on a machine with a card."""
        return cls((), cores or max(1, (os.cpu_count() or 2) - 1))

    @property
    def gpu(self) -> bool:
        return bool(self.cards)

    @property
    def count(self) -> int:
        """How many shares run at once: the cards, or the cores."""
        return len(self.cards) if self.cards else self.cores

    def split(self, n: int) -> list[np.ndarray]:
        """``range(n)`` in :attr:`count` contiguous shares, some possibly empty."""
        return np.array_split(np.arange(n), max(1, min(self.count, max(n, 1))))

    def map(self, work: Callable[[int, np.ndarray], T], shares: Sequence[np.ndarray]) -> list[T]:
        """``work(card, share)`` for every share at once, results in share order.

        On cards, one thread per card and ``card`` is its ordinal. On the host
        ``card`` is -1: on Linux one forked process per share, which inherits
        ``work`` and what it closes over so only the result is pickled; elsewhere
        one thread per share.
        """
        shares = [s for s in shares if s.size]
        if not shares:
            return []
        if self.cards:
            with ThreadPoolExecutor(max_workers=len(shares)) as pool:
                futures = [
                    pool.submit(work, card, share)
                    for card, share in zip(self.cards, shares, strict=False)
                ]
                return [f.result() for f in futures]
        if len(shares) == 1:
            return [work(-1, shares[0])]
        if not sys.platform.startswith("linux"):
            with ThreadPoolExecutor(max_workers=len(shares)) as pool:
                return list(pool.map(lambda share: work(-1, share), shares))
        _FORKED[:] = [work, shares]
        try:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=len(shares), mp_context=context) as pool:
                return list(pool.map(_run_forked, range(len(shares))))
        finally:
            _FORKED.clear()

    def record(self) -> dict[str, Any]:
        return {"cards": list(self.cards), "cores": self.cores}
