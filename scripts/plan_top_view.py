"""Top view of an HSSD scene, twice: the rooms a person would name, and the
regions the dataset cuts the same air into.

The four rules that turn regions into rooms live in
:mod:`reverberate.geometry.rooms`, which is what the simulation uses too, so
the picture cannot drift from the volume that gets solved.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import box
from shapely.ops import unary_union

from reverberate.geometry.apartment import (
    OUTDOOR_LABELS,
    WALL_SECTION_HEIGHT,
    load_stage,
    wall_footprint,
)
from reverberate.geometry.hssd_room import load_regions
from reverberate.geometry.rooms import (
    DOOR_MAX_M,
    MIN_ROOM_M2,
    WALL_MAX_M,
    everyday_rooms,
    shared_boundaries,
)

ROOT = Path("/Users/gilles/Developer/reverberate/data/raw/hssd-hab")


#: Never part of a room's name: a walk-in absorbed by its bedroom would only
#: lengthen the label.
CLOSET = {"closet"}

#: Everyday name per region. A merged room is named by joining these with a
#: hyphen, largest part first, so an open kitchen and lounge read "salon-cuisine".
FR = {
    "102344403": {
        "lounge": "petit salon",
        "rec/game": "salle de jeux",
        "garage": "garage",
        "living room": "salon",
        "kitchen": "cuisine",
        "laundryroom": "buanderie",
        "hallway": "degagement",
        "gym": "salle de sport",
        "office": "bureau",
        "bedroom": "chambre 1",
        "bathroom": "salle d'eau 1",
        "hallway.001": "couloir",
        "bathroom.001": "salle d'eau 2",
        "bedroom.001": "chambre 2",
        "bedroom.002": "suite parentale",
        "bathroom.002": "salle de bains",
        "toilet": "WC",
        "closet.001": "dressing 2",
        "closet.002": "dressing de la suite",
        "outdoor": "jardin",
        "outdoor.001": "terrasse",
        "driveway": "allee",
    },
    "102344022": {
        "bedroom": "studio",
        "bathroom": "salle d'eau du studio",
        "other room": "arriere-cuisine",
        "bathroom.001": "salle de bains",
        "living room": "salon",
        "garage": "garage",
        "toilet": "WC",
        "bedroom.001": "chambre 1",
        "other room.001": "chambre 2",
        "outdoor": "cour",
        "hallway": "entree et escalier",
        "closet.002": "sous l'escalier",
        "dining room": "coin repas",
        "kitchen": "cuisine",
    },
    "102344094": {
        "bedroom": "chambre",
        "bathroom": "salle de bains",
        "living room": "salon",
        "balcony": "balcon",
        "kitchen": "cuisine",
        "dining room": "coin repas",
        "closet.003": "cagibi du balcon",
        "closet.001": "dressing",
        "closet": "coin machine a laver",
    },
}

#: Region highlighted on the dataset panel, because the project solved or budgeted it.
FLAG = {
    "102344403": ("living room", "solve par W39 : 296 m3 scelles"),
    "102344022": ("living room", "budget salon de la roadmap : 151,7 m3"),
    "102344094": (None, "aucune simulation sur cette scene"),
}


def _runs(geometry):
    return list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]


def metres(area):
    """A square metre is coarse for a 0.8 m2 toilet, so keep a decimal there."""
    text = f"{area:.1f}" if area < 10 else f"{area:.0f}"
    return text.replace(".", ",") + " m2"


LABEL_BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9)


def place(ax, polygon, text, fontsize, centre, pending):
    """Draw a region's label, and note it for the fit check in ``settle``."""
    point = polygon.representative_point()
    artist = ax.text(
        point.x,
        point.y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=6,
        bbox=LABEL_BOX,
    )
    pending.append((ax, artist, polygon, point, fontsize, centre))


def settle(fig, pending):
    """Give a leader to every label its own region cannot hold, and keep labels
    off each other.

    The trigger is the label as rendered, not a guess at the region's size. A
    guess is what let `buanderie` -- 2.70 by 1.77 m -- keep a label wider than
    itself, spilling onto the 7.5 m2 hallway beside it and making a 4.8 m2 room
    look like a large one.
    """
    fig.canvas.draw()
    sized = []
    for ax, artist, polygon, point, fontsize, centre in pending:
        window = artist.get_window_extent()
        (x0, z0), (x1, z1) = ax.transData.inverted().transform(
            [[window.x0, window.y0], [window.x1, window.y1]]
        )
        sized.append((ax, artist, polygon, point, fontsize, centre, abs(x1 - x0), abs(z1 - z0)))

    taken = {}
    # Largest room first: it keeps the obvious spot, and the small ones move.
    for ax, artist, polygon, point, fontsize, centre, width, height in sorted(
        sized, key=lambda e: -e[2].area
    ):
        away = np.array([point.x - centre[0], point.y - centre[1]])
        norm = np.linalg.norm(away)
        step = away / norm if norm > 1e-6 else np.array([0.0, 1.0])
        placed, chosen = None, None
        for distance in np.arange(0.0, 6.0, 0.3):
            x = point.x + step[0] * distance
            z = point.y + step[1] * distance
            here = box(x - width / 2, z - height / 2, x + width / 2, z + height / 2)
            if distance == 0.0 and not polygon.contains(here):
                continue
            if any(here.intersects(other) for other in taken.get(ax, [])):
                continue
            placed, chosen = (x, z), here
            break
        if placed is None:
            placed, chosen = (
                (point.x, point.y),
                box(
                    point.x - width / 2,
                    point.y - height / 2,
                    point.x + width / 2,
                    point.y + height / 2,
                ),
            )
        taken.setdefault(ax, []).append(chosen)
        if placed == (point.x, point.y):
            continue
        text = artist.get_text()
        artist.remove()
        ax.annotate(
            text,
            xy=(point.x, point.y),
            xytext=placed,
            ha="center",
            va="center",
            fontsize=fontsize - 0.5,
            zorder=7,
            bbox=LABEL_BOX,
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.9, shrinkA=2, shrinkB=1),
        )


def room_name(scene, room):
    """The room's own name, region by region, said in everyday French."""
    fr = FR.get(scene, {})
    named = [n for n in room.regions if n.split(".")[0] not in CLOSET] or list(room.regions)
    return "-".join(dict.fromkeys(fr.get(n, n) for n in named))


def draw(scene, out, root=ROOT):
    regions = load_regions(root / "semantics" / "scenes" / f"{scene}.semantic_config.json")
    stage = load_stage(root, scene)
    floor = float(np.mean([r.floor_height for r in regions]))
    height = floor + WALL_SECTION_HEIGHT
    section = stage.section(plane_origin=[0, height, 0], plane_normal=[0, 1, 0])
    segs = [np.array([[v[0], v[2]] for v in section.vertices[e]]) for e in section.vertex_nodes]
    # The same wall band the rules are measured against, never a second one.
    wall = wall_footprint(stage, height)
    polys = [r.polygon_xz.buffer(0) for r in regions]
    rooms = everyday_rooms(regions, wall)

    # What each rule did, drawn so it can be checked: a boundary inside a room
    # was dissolved, a boundary between two rooms still holds a passage.
    where = {r.name: i for i, r in enumerate(regions)}
    owner = {name: room for room in rooms for name in room.regions}
    dissolved, doorways = [], []
    for (i, j), boundary in shared_boundaries(polys, wall).items():
        if regions[i].label in OUTDOOR_LABELS and regions[j].label in OUTDOOR_LABELS:
            continue
        line = polys[i].boundary.intersection(polys[j].buffer(0.16))
        same = owner[regions[i].name] is owner[regions[j].name]
        if same:
            dissolved.append(line)
        elif boundary.widest_m > 0:
            doorways.extend(g for g in _runs(line.difference(wall)) if g.length > 0.25)

    fig, axes = plt.subplots(1, 2, figsize=(26, 13))
    # tab20 holds two greys, and grey already means "outdoor" here, so drop them.
    palette = [c for k, c in enumerate(plt.get_cmap("tab20").colors) if k not in (14, 15)]
    cmap = lambda k: palette[k % len(palette)]  # noqa: E731
    pending = []
    whole = unary_union(polys)
    centre = (whole.centroid.x, whole.centroid.y)

    ax = axes[0]
    for k, room in enumerate(rooms):
        colour = "0.85" if room.outdoor else cmap(k % 20)
        for part in _runs(room.polygon):
            xs, zs = part.exterior.xy
            ax.fill(xs, zs, color=colour, alpha=0.35, zorder=0)
            ax.plot(xs, zs, color="0.2" if room.outdoor else colour, lw=2.8, zorder=3)
        name = room_name(scene, room)
        wrapped = name if len(name) < 26 else name.replace("-", "-\n")
        biggest = polys[where[room.regions[0]]]
        place(ax, biggest, f"{wrapped}\n{metres(room.area_m2)}", 11, centre, pending)
    for line in dissolved:
        for g in line.geoms if hasattr(line, "geoms") else [line]:
            ax.plot(*g.xy, color="crimson", lw=1.3, ls=(0, (4, 3)), zorder=4)
    for run in doorways:
        ax.plot(*run.xy, color="#0b7285", lw=5.5, solid_capstyle="butt", zorder=5)
    ax.set_title(
        "Les pieces au sens courant\n"
        f"fondues si le passage depasse {DOOR_MAX_M:.2f} m et que la frontiere "
        f"porte moins de {WALL_MAX_M:.2f} m de mur ;\n"
        "un couloir separe puis rejoint la plus grande piece qu'il dessert ; "
        f"rien sous {MIN_ROOM_M2:.0f} m2 n'est une piece",
        fontsize=12.5,
    )

    ax = axes[1]
    flag_name, flag_txt = FLAG.get(scene, (None, "-"))
    for i, r in enumerate(regions):
        outdoor = r.label in OUTDOOR_LABELS
        # A region whose authored loop self-intersects cleans to a MultiPolygon;
        # two of the 168 scenes have one, so never assume a single ring here.
        for part in _runs(polys[i]):
            xs, zs = part.exterior.xy
            ax.fill(xs, zs, color="0.85" if outdoor else cmap(i % 20), alpha=0.35, zorder=0)
            ax.plot(xs, zs, color="0.2" if outdoor else cmap(i % 20), lw=2.2, zorder=3)
            if r.name == flag_name:
                ax.fill(xs, zs, facecolor="none", hatch="///", edgecolor="crimson", lw=0, zorder=1)
                ax.plot(xs, zs, color="crimson", lw=3.2, zorder=4)
        place(ax, polys[i], f"{r.name}\n{metres(polys[i].area)}", 9.5, centre, pending)
    ax.set_title(
        f"Le decoupage du dataset : region_annotations de {scene}\nhachure rouge = {flag_txt}",
        fontsize=13,
    )

    for ax in axes:
        for g in segs:
            ax.plot(g[:, 0], g[:, 1], color="0.15", lw=1.4, zorder=2)
        ax.set_aspect("equal")
        ax.margins(0.06)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)

    cut = f"{WALL_SECTION_HEIGHT:.2f}".replace(".", ",")
    handles = [
        Line2D([], [], color="0.15", lw=1.4, label=f"murs du stage, coupe a {cut} m"),
        Line2D([], [], color="#0b7285", lw=5, label="passage qui ne fond pas les deux regions"),
        Line2D([], [], color="crimson", lw=1.3, ls=(0, (4, 3)), label="limite du dataset dissoute"),
        Line2D([], [], color="0.6", lw=8, alpha=0.5, label="exterieur, hors volume simule"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.suptitle(f"HSSD {scene} - vue du dessus", fontsize=16)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    settle(fig, pending)
    fig.savefig(out, dpi=120)

    inside = [room for room in rooms if not room.outdoor]
    print(
        f"{scene}: {len(inside)} pieces "
        f"pour {len([r for r in regions if r.label not in OUTDOOR_LABELS])} regions -> {out}"
    )
    for room in inside:
        print(f"  {room.area_m2:6.1f} m2  {room_name(scene, room)}   <- {', '.join(room.regions)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--hssd-root", type=Path, default=ROOT)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        draw(scene, out_dir / f"plan_{scene}.png", args.hssd_root)
