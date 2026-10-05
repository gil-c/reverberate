"""A card kernel's own text, compiled for the host: the proof a machine without a card can give.

The solver's rule is that a kernel equals the ``numpy`` step to the bit, and
``python -m reverberate.wave.lowband verify`` holds it on a card. A change of
a kernel made on a laptop would otherwise wait for a rental to be read at
all. Here the CUDA text is compiled as C++ by the host's compiler, without
contraction of a product and a sum into one rounding (``-ffp-contract=off``,
what ``-fmad=false`` is to ``nvcc``), and each launch is run as its grid of
blocks and threads, one after another: a thread of these kernels writes only
its own node and its own rows, so the order of the threads changes nothing.

:func:`host_kernel` gives what :func:`reverberate.compute.raw_kernel` gives,
called the same way, on ``numpy`` arrays. It proves the arithmetic and the
indexing of the text. It does not prove what only a card has: its compiler's
own choices, and two threads at once. ``verify`` stays the last word.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["compiler", "host_kernel"]

_PREAMBLE = r"""
struct lowband_dim { int x, y, z; };
static lowband_dim blockIdx, threadIdx, blockDim;
#define __global__
#define __restrict__ __restrict
static inline int __popc(unsigned int v) { return __builtin_popcount(v); }
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
"""

_SCALARS: dict[str, Any] = {
    "int": ctypes.c_int,
    "long long": ctypes.c_longlong,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
}

_ELEMENTS: dict[str, Any] = {
    "float *": np.float32,
    "double *": np.float64,
    "int *": np.int32,
    "unsigned int *": np.uint32,
    "unsigned short *": np.uint16,
    "signed char *": np.int8,
    "long long *": np.int64,
}


def compiler() -> str | None:
    """The host's C++ compiler, or ``None`` where there is none."""
    for name in ("c++", "g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _signature(source: str, name: str) -> list[tuple[str, str]]:
    """The parameters of kernel ``name`` in ``source``: ``(type, name)`` in order."""
    found = re.search(rf"void\s+{re.escape(name)}\s*\(([^)]*)\)", source)
    if found is None:
        raise ValueError(f"no kernel {name} in the source")
    out = []
    for parameter in found.group(1).split(","):
        words = parameter.replace("*", " * ").split()
        kind = " ".join(w for w in words[:-1] if w not in ("const", "__restrict__"))
        out.append((kind, words[-1]))
    return out


@functools.cache
def _library(source: str, name: str) -> Any:
    parameters = _signature(source, name)
    declared = ", ".join(f"{kind} {held}" for kind, held in parameters)
    passed = ", ".join(held for _, held in parameters)
    driver = f"""
extern "C" void run_{name}(int grid_x, int grid_y, int threads, {declared})
{{
    blockDim.x = threads; blockDim.y = 1; blockDim.z = 1;
    threadIdx.y = 0; threadIdx.z = 0; blockIdx.z = 0;
    for (int by = 0; by < grid_y; ++by)
        for (int bx = 0; bx < grid_x; ++bx)
            for (int t = 0; t < threads; ++t) {{
                blockIdx.x = bx; blockIdx.y = by; threadIdx.x = t;
                {name}({passed});
            }}
}}
"""
    found = compiler()
    if found is None:
        raise RuntimeError("no C++ compiler on this host")
    digest = hashlib.sha256((source + name).encode()).hexdigest()[:16]
    work = Path(tempfile.mkdtemp(prefix="lowband-hostkernel-"))
    text = work / f"{name}-{digest}.cpp"
    built = work / f"{name}-{digest}.so"
    text.write_text(_PREAMBLE + source + driver)
    subprocess.run(
        [
            found,
            "-O1",
            "-std=c++14",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(text),
            "-o",
            built,
        ],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(built))
    run = getattr(library, f"run_{name}")
    run.restype = None
    run.argtypes = [ctypes.c_int] * 3 + [
        ctypes.c_void_p if "*" in kind else _SCALARS[kind] for kind, _ in parameters
    ]
    return run, parameters


def host_kernel(source: str, name: str) -> Any:
    """``name`` of ``source`` as a callable ``(grid, block, arguments)``, run on the host."""
    run, parameters = _library(source, name)

    def launch(grid: tuple[int, ...], block: tuple[int, ...], arguments: tuple[Any, ...]) -> None:
        if len(arguments) != len(parameters):
            raise TypeError(f"{name} takes {len(parameters)} arguments, not {len(arguments)}")
        passed = []
        for (kind, held), value in zip(parameters, arguments, strict=True):
            if "*" in kind:
                if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
                    raise TypeError(f"{name}: {held} is not a contiguous array")
                if value.dtype != _ELEMENTS[kind]:
                    raise TypeError(f"{name}: {held} is {value.dtype}, the kernel reads {kind}")
                passed.append(value.ctypes.data)
            else:
                passed.append(value.item() if isinstance(value, np.generic) else value)
        grid_y = int(grid[1]) if len(grid) > 1 else 1
        run(int(grid[0]), grid_y, int(block[0]), *passed)

    return launch
