"""That two processes handed the same scene build the same geometry.

The existing tests were all within one process, which is exactly the case the
defect did not affect. The same scene exported twice gave 608 098 triangles
against 614 330, because ``trimesh.sample.sample_surface`` drew from OS entropy
and the 95th percentile it fed decided which face budget ``decimate_within``
accepted. Within a process the two calls also differ, but the acceptance
happens once, so nothing downstream disagrees with anything.

The check therefore has to be made in a child interpreter, and with
``PYTHONHASHSEED`` varied, since iteration order over a set or a dict is the
other classic way a result travels between processes without being asked to.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import trimesh

from reverberate.geometry.decimation import decimate_within, deviation

SRC = str(Path(__file__).resolve().parents[1] / "src")

#: Built in the child as well as here, so the two never share a mesh object.
#: Bumpy on purpose: a smooth primitive decimates the same way whatever the
#: sample says, and would pass this test while the defect was still present.
BUILD_MESH = """
import numpy as np
import trimesh

sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
rng = np.random.default_rng(7)
offsets = 1.0 + 0.25 * rng.random(len(sphere.vertices))
mesh = trimesh.Trimesh(
    vertices=sphere.vertices * offsets[:, None], faces=sphere.faces, process=False
)
"""


def bumpy_mesh() -> trimesh.Trimesh:
    scope: dict[str, object] = {}
    exec(BUILD_MESH, scope)  # noqa: S102 - the child runs the identical source
    mesh = scope["mesh"]
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


def run_in_child(body: str, hash_seed: str) -> dict[str, object]:
    """Run ``body`` in a fresh interpreter and read back its JSON report."""
    script = textwrap.dedent(BUILD_MESH) + textwrap.dedent(body)
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": SRC}
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return dict(json.loads(completed.stdout.strip().splitlines()[-1]))


REPORT = """
import hashlib, json
import numpy as np
from reverberate.geometry.envelope import acoustic_envelope

envelope = acoustic_envelope(mesh)
digest = hashlib.sha256()
digest.update(np.ascontiguousarray(envelope.mesh.vertices, dtype=np.float64).tobytes())
digest.update(np.ascontiguousarray(envelope.mesh.faces, dtype=np.int64).tobytes())
print(json.dumps({
    "faces": len(envelope.mesh.faces),
    "parts": envelope.parts,
    "deviation": envelope.deviation,
    "digest": digest.hexdigest(),
}))
"""


def test_two_processes_build_the_same_envelope() -> None:
    """The W7 defect, as a test: same input, two interpreters, one answer.

    Vertices and faces are hashed rather than counted. A face count agreeing by
    luck is precisely what made the original defect survive review, since the
    two exports differed by one per cent of triangles and each looked plausible.
    """
    first = run_in_child(REPORT, "0")
    second = run_in_child(REPORT, "12345")
    assert first == second


def test_the_sample_that_decides_the_budget_is_pinned() -> None:
    """``deviation`` is the accept test, so it must not move on its own."""
    mesh = bumpy_mesh()
    hull = mesh.convex_hull
    assert deviation(hull, mesh) == deviation(hull, mesh)


def test_a_different_seed_is_allowed_to_disagree() -> None:
    """Pinning is not the same as the sample being irrelevant.

    If two seeds gave identical numbers, the sample would not be measuring
    anything and the fix would be hiding the defect rather than removing it.
    """
    mesh = bumpy_mesh()
    hull = mesh.convex_hull
    assert deviation(hull, mesh, seed=0) != deviation(hull, mesh, seed=1)


def test_reduction_is_reproducible_within_a_process() -> None:
    mesh = bumpy_mesh()
    first, first_error = decimate_within(mesh)
    second, second_error = decimate_within(mesh)
    assert len(first.faces) == len(second.faces)
    assert first_error == second_error
    assert np.array_equal(first.faces, second.faces)
