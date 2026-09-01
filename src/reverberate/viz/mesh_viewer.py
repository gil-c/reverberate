"""Standalone mesh viewer for raw 3D-FRONT-MIDI scenes.

This tool exists to answer one question before any decimation, material
assignment or simulation code is written: do the per object semantic labels
that the MIDI-3D repackaging claims to preserve actually survive, and is the
room shell usable at all.

A "room" directory in the MIDI-3D layout (for example
``<scene-uuid>/SecondBedroom-1656/``) holds one GLB file per furniture
object, named ``<Category>_<object-uuid>_<index>.glb``, plus a single
``ceil.glb`` that has no semantic label of its own. There is no separate
wall or floor mesh: whatever is not a piece of furniture ends up in
``ceil.glb``, despite the name.

Usage::

    python -m reverberate.viz.mesh_viewer <room_dir> -o report.html

The output is a single self contained HTML file (no external assets, no
network access needed to view it) with the scene coloured by semantic
category, so it can be opened and shared as easily as a screenshot.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pyvista as pv
import trimesh

__all__ = [
    "SEMANTIC_UNLABELLED",
    "LabelledMesh",
    "category_from_filename",
    "load_room",
    "render_room_html",
]

#: Category assigned to geometry that carries no semantic label in its
#: filename (in practice, always the room shell in ``ceil.glb``).
SEMANTIC_UNLABELLED = "shell (unlabelled)"

#: A v4 UUID, hyphenated, lower or upper case. MIDI-3D filenames look like
#: ``Cabinet_Shelf_Desk_e314cd3c-e309-4614-98d2-13f99208ced8_4.glb``: the
#: category is whatever precedes this pattern.
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

#: Fixed, deterministic colour palette (Tableau 20, hex), so the same
#: category always gets the same colour across renders and reports.
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]


@dataclass(frozen=True)
class LabelledMesh:
    """One GLB file's geometry, tagged with the semantic category read from
    its filename."""

    path: Path
    category: str
    mesh: trimesh.Trimesh


def category_from_filename(filename: str) -> str:
    """Return the semantic category encoded in a MIDI-3D GLB filename.

    ``ceil.glb`` and any file without an embedded object UUID are reported
    as :data:`SEMANTIC_UNLABELLED`.
    """
    stem = Path(filename).stem
    match = _UUID_RE.search(stem)
    if match is None:
        return SEMANTIC_UNLABELLED
    category = stem[: match.start()].rstrip("_")
    return category or SEMANTIC_UNLABELLED


def _load_single_mesh(path: Path) -> trimesh.Trimesh:
    """Load one GLB file and flatten it to a single triangle mesh in world
    coordinates (units as stored in the file, no rescaling)."""
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"{path}: expected a triangle mesh, got {type(loaded)!r}")
    return loaded


def load_room(room_dir: Path) -> list[LabelledMesh]:
    """Load every ``*.glb`` directly inside ``room_dir`` (non recursive, so
    the top level ``<Room>.glb`` / ``<Room>_full.glb`` aggregates that sit
    next to the room directory are not picked up and double counted)."""
    glb_paths = sorted(room_dir.glob("*.glb"))
    if not glb_paths:
        raise FileNotFoundError(f"no .glb files found directly in {room_dir}")
    return [
        LabelledMesh(
            path=path,
            category=category_from_filename(path.name),
            mesh=_load_single_mesh(path),
        )
        for path in glb_paths
    ]


def _color_for_category(category: str, known_categories: list[str]) -> str:
    index = known_categories.index(category)
    return _PALETTE[index % len(_PALETTE)]


def render_room_html(room_dir: Path, output_path: Path) -> Path:
    """Render every mesh in ``room_dir``, coloured by semantic category, to
    a single self contained interactive HTML file.

    Returns ``output_path`` for convenience.
    """
    labelled_meshes = load_room(room_dir)
    categories = sorted({lm.category for lm in labelled_meshes})

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")  # type: ignore[arg-type]
    legend_entries: list[tuple[str, str]] = []
    for category in categories:
        colour = _color_for_category(category, categories)
        legend_entries.append((category, colour))
        for labelled in labelled_meshes:
            if labelled.category != category:
                continue
            poly = pv.wrap(labelled.mesh)
            plotter.add_mesh(poly, color=colour, show_edges=False, opacity=0.9, label=category)

    plotter.add_legend(legend_entries, bcolor="white", face="rectangle")  # type: ignore[arg-type]
    plotter.add_axes()  # type: ignore[call-arg]
    plotter.camera_position = "iso"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.export_html(str(output_path))
    plotter.close()
    return output_path


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "room_dir", type=Path, help="a MIDI-3D room directory, e.g. .../SecondBedroom-1656"
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("mesh_viewer_report.html"))
    args = parser.parse_args(argv)

    output = render_room_html(args.room_dir, args.output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
