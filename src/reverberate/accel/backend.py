"""Which array library runs a stage, and the rules every custom kernel is built under.

``cupy`` when a CUDA device answers, ``numpy`` otherwise, and the caller never
knows which beyond the module it was handed. The tests run on numpy; a rented
card runs cupy; ``REVERBERATE_NO_GPU=1`` forces numpy on a machine that has a
card, which is how the twin is measured on the same host.

**Kernels are compiled without fused multiply-add.** Numpy computes ``a*b+c``
as two roundings; nvcc fuses them into one by default, which is more accurate
and different in the last bit. A voxeliser that must reproduce a CPU grid byte
for byte cannot afford "more accurate", so :data:`KERNEL_OPTIONS` carries
``-fmad=false`` and :func:`raw_kernel` is the only way a kernel is built.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np

__all__ = [
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
