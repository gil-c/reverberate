"""Interactive first person "walkthrough" viewer for a reconstructed HSSD
room, reproducing the interior exploration view shown in the teaser on
https://3dlg-hcvc.github.io/hssd (a person walking through a room and
turning their head to look around), rendered with our own stack.

The HSSD authors' own interactive viewer is Habitat-sim (see
``3dlg-hcvc/hssd`` on GitHub and its ``teaser_small.mp4``): a C++/conda
simulator built for training navigation agents, not a lightweight
visualisation tool, and it is not part of this project's stack (section 7 of
ROADMAP.md is `pip`/`venv` only, `PyVista` for 3D visualisation). This module
reproduces the *effect*, walking a fixed-height viewpoint through the room
and turning it to look around, using ``PyVista`` and ``trame``, which are
already project dependencies. It does not use, vendor or depend on
Habitat-sim.

Two render modes, toggled live in the browser:

- **Colour**: a muted, plausible interior colour per surface and per
  furniture category (see :mod:`reverberate.viz.palette` for why this is an
  approximation, not a photorealistic render: HSSD's furniture textures are
  frequently small palette atlases that do not reduce to one representative
  colour).
- **Labels**: flat, vivid colour per semantic category, with a legend, i.e.
  a semantic segmentation view, the same idea as the semantic map shown
  alongside RGB in the HSSD teaser.

Usage::

    python -m reverberate.viz.interior_walkthrough <hssd_root> <scene_id> <region_name>

Then open the printed URL. Controls: ``W``/``S`` or up/down arrows to walk
forward/back, ``A``/``D`` to strafe, ``Q``/``E`` or left/right arrows to turn,
``R``/``F`` to look up/down.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from shapely.geometry import Polygon
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

from reverberate.geometry.hssd_room import RoomReconstruction, build_room
from reverberate.viz.palette import color_for_category, muted_color_for_category
from reverberate.viz.walk_navigation import clamp_into_polygon, forward_vector, walk_step

if TYPE_CHECKING:
    import trimesh

__all__ = [
    "EYE_HEIGHT_ABOVE_FLOOR",
    "classify_shell_faces",
    "split_shell_by_surface",
    "InteriorWalkthroughViewer",
    "build_trame_app",
]

#: Adult standing eye height above the floor, metres.
EYE_HEIGHT_ABOVE_FLOOR = 1.6
#: Horizontal distance covered by one forward/strafe key press, metres: a
#: single walking pace.
_MOVE_STEP = 0.35
#: Degrees turned per left/right key press.
_TURN_STEP = 12.0
#: Degrees the head tilts per look up/down key press.
_LOOK_STEP = 8.0
#: How close a walker may get to a wall before being nudged back, metres.
_WALL_CLEARANCE = 0.3
_PITCH_LIMIT_DEGREES = 60.0

_REALISTIC_SHELL_COLOR = {
    "floor": "#c9a876",
    "wall": "#e8e2d5",
    "ceiling": "#f5f5f0",
}

#: Keyboard key to (forward, strafe, turn, look) unit deltas.
_KEY_ACTIONS: dict[str, dict[str, float]] = {
    "w": {"forward": 1.0},
    "ArrowUp": {"forward": 1.0},
    "s": {"forward": -1.0},
    "ArrowDown": {"forward": -1.0},
    "a": {"strafe": -1.0},
    "d": {"strafe": 1.0},
    "q": {"turn": -1.0},
    "ArrowLeft": {"turn": -1.0},
    "e": {"turn": 1.0},
    "ArrowRight": {"turn": 1.0},
    "r": {"look": 1.0},
    "f": {"look": -1.0},
}


def classify_shell_faces(face_normals: npt.NDArray[np.float64]) -> npt.NDArray[np.str_]:
    """Classify each shell face as floor, ceiling or wall from its outward
    normal's vertical (Y) component.

    A simple, deterministic heuristic: the extruded room shell only ever has
    these three surface types (see
    :func:`reverberate.geometry.hssd_room.extrude_region_shell`), so a
    steeply up or down facing normal is the horizontal cap, anything else is
    a vertical wall.
    """
    labels = np.full(len(face_normals), "wall", dtype=object)
    labels[face_normals[:, 1] > 0.5] = "ceiling"
    labels[face_normals[:, 1] < -0.5] = "floor"
    return labels


def split_shell_by_surface(shell: trimesh.Trimesh) -> dict[str, pv.PolyData]:
    """Split an extruded room shell into floor/wall/ceiling sub-meshes so
    each can be coloured independently."""
    labels = classify_shell_faces(shell.face_normals)
    result: dict[str, pv.PolyData] = {}
    for surface in ("floor", "wall", "ceiling"):
        face_indices = np.where(labels == surface)[0]
        if len(face_indices) == 0:
            continue
        sub = shell.submesh([face_indices], append=True)
        result[surface] = pv.wrap(sub)
    return result


@dataclass
class _ActorEntry:
    actor: Any
    label_color: str
    muted_color: str


class InteriorWalkthroughViewer:
    """Owns the PyVista scene and the walker's pose for one reconstructed
    room, independent of any particular UI framework."""

    def __init__(self, room: RoomReconstruction, mode: str = "labels") -> None:
        if mode not in ("labels", "color"):
            raise ValueError(f"mode must be 'labels' or 'color', got {mode!r}")
        self.room = room
        self.polygon = Polygon(room.region.poly_loop_xz)
        self.eye_height = room.region.floor_height + EYE_HEIGHT_ABOVE_FLOOR
        start = self.polygon.representative_point()
        self.position_xz = (start.x, start.y)
        self.yaw = 0.0
        self.pitch = 0.0
        self.mode = mode

        self.plotter = pv.Plotter()
        self.plotter.set_background("black")  # type: ignore[arg-type]
        self.plotter.camera.view_angle = 90  # a wide, human-like field of view
        self._actors: list[_ActorEntry] = []
        self._build_scene()
        self._apply_camera()

    @property
    def categories(self) -> list[str]:
        """Every category present, shell surfaces plus furniture, sorted so
        colour assignment is stable across a session."""
        categories = {"floor", "wall", "ceiling"}
        categories.update(fm.category for fm in self.room.furniture)
        return sorted(categories)

    def _build_scene(self) -> None:
        categories = self.categories
        for surface, mesh in split_shell_by_surface(self.room.shell).items():
            label_color = color_for_category(surface, categories)
            muted_color = _REALISTIC_SHELL_COLOR[surface]
            initial = label_color if self.mode == "labels" else muted_color
            actor = self.plotter.add_mesh(
                mesh, color=initial, show_edges=False, name=f"shell_{surface}"
            )
            self._actors.append(_ActorEntry(actor, label_color, muted_color))

        for index, labelled in enumerate(self.room.furniture):
            poly = pv.wrap(labelled.mesh)
            label_color = color_for_category(labelled.category, categories)
            muted_color = muted_color_for_category(labelled.category, categories)
            initial = label_color if self.mode == "labels" else muted_color
            actor = self.plotter.add_mesh(
                poly, color=initial, show_edges=False, name=f"furniture_{index}"
            )
            self._actors.append(_ActorEntry(actor, label_color, muted_color))

    def set_mode(self, mode: str) -> None:
        if mode not in ("labels", "color"):
            raise ValueError(f"mode must be 'labels' or 'color', got {mode!r}")
        self.mode = mode
        for entry in self._actors:
            entry.actor.prop.color = entry.label_color if mode == "labels" else entry.muted_color
        self.plotter.render()

    def _apply_camera(self) -> None:
        eye = (self.position_xz[0], self.eye_height, self.position_xz[1])
        fx, fy, fz = forward_vector(self.yaw, self.pitch)
        focal = (eye[0] + fx, eye[1] + fy, eye[2] + fz)
        self.plotter.camera.position = eye
        self.plotter.camera.focal_point = focal
        self.plotter.camera.up = (0.0, 1.0, 0.0)
        self.plotter.render()

    def move(
        self, forward: float = 0.0, strafe: float = 0.0, turn: float = 0.0, look: float = 0.0
    ) -> None:
        """Advance the walker by one step: a discrete pace, turn or head
        tilt, matching a person walking and looking around a room rather
        than a free-flying camera."""
        self.yaw = (self.yaw + turn * _TURN_STEP) % 360.0
        self.pitch = max(
            -_PITCH_LIMIT_DEGREES, min(_PITCH_LIMIT_DEGREES, self.pitch + look * _LOOK_STEP)
        )
        proposed = walk_step(self.position_xz, self.yaw, forward * _MOVE_STEP, strafe * _MOVE_STEP)
        self.position_xz = clamp_into_polygon(proposed, self.polygon, _WALL_CLEARANCE)
        self._apply_camera()


def build_trame_app(viewer: InteriorWalkthroughViewer, title: str) -> Any:
    """Wire ``viewer`` into a trame web app: mode toggle, legend, and
    keyboard-driven first person navigation."""
    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller
    state.mode = viewer.mode

    def on_key(key: str) -> None:
        action = _KEY_ACTIONS.get(key)
        if action is not None:
            viewer.move(**action)

    ctrl.on_key = on_key

    @state.change("mode")  # type: ignore[untyped-decorator]
    def _on_mode_change(mode: str, **_kwargs: object) -> None:
        viewer.set_mode(mode)

    with SinglePageLayout(server) as layout:
        layout.title.set_text(title)
        with layout.toolbar:
            vuetify.VSpacer()
            with vuetify.VBtnToggle(v_model=("mode", viewer.mode), mandatory=True, dense=True):
                vuetify.VBtn("Colour", value="color")
                vuetify.VBtn("Labels", value="labels")

        content_style = "position:relative;width:100%;height:100%;outline:none;"
        with (
            layout.content,
            html.Div(
                style=content_style,
                tabindex="0",
                **{"@keydown.window": (ctrl.on_key, "[$event.key]")},
            ),
        ):
            with html.Div(style="width:100%;height:100%;"):
                plotter_ui(viewer.plotter)  # type: ignore[no-untyped-call]
            with html.Div(
                style=(
                    "position:absolute;top:8px;left:8px;background:rgba(0,0,0,0.6);"
                    "color:white;padding:10px 12px;border-radius:6px;"
                    "font-size:12px;line-height:1.5;max-width:260px;"
                )
            ):
                html.Div(
                    "W/S or \u2191/\u2193 walk, A/D strafe, "
                    "Q/E or \u2190/\u2192 turn, R/F look up/down"
                )
                for category in viewer.categories:
                    swatch_color = color_for_category(category, viewer.categories)
                    with html.Div(style="display:flex;align-items:center;gap:6px;"):
                        html.Div(
                            style=(
                                f"width:10px;height:10px;background:{swatch_color};"
                                "border-radius:2px;flex-shrink:0;"
                            )
                        )
                        html.Span(category)

    return server


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hssd_root", type=Path, help="path to a cloned hssd/hssd-hab checkout")
    parser.add_argument("scene_id", help="HSSD scene id, e.g. 102343992")
    parser.add_argument("region_name", help="a room name from that scene's region_annotations")
    parser.add_argument("--mode", choices=["color", "labels"], default="labels")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    room = build_room(args.hssd_root, args.scene_id, args.region_name)
    if room.skipped_instances:
        print(
            f"warning: skipped {len(room.skipped_instances)} furniture instance(s) "
            "with no collider file on disk",
            file=sys.stderr,
        )
    viewer = InteriorWalkthroughViewer(room, mode=args.mode)
    server = build_trame_app(viewer, title=f"{args.scene_id} / {args.region_name}")
    server.start(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
