# 0010: a room is not a region

Status: accepted.

## Context

HSSD's `region_annotations` are floor polygons with a label. **Nothing in them
says whether a wall stands between two neighbours**, so the dataset routinely
cuts one continuous body of air into several "rooms". On `102344403` the
`living room` it names is 81 m2 of a space that runs on into a 26.7 m2 `kitchen`
through an opening **6.97 m wide**.

Every part of this project that means "room" was taking that word from the
dataset. `w39_living_8k` therefore solved a sealed **296.1 m3** volume whose
walls include a rigid partition where the plan has a 6.97 m opening. The room is
126.3 m2 and **461.8 m3**: the run was 36 per cent short of the volume it named,
and a comparison of its RT60 against the roadmap's Sabine bounds was comparing
two different rooms. Volume, absorbing area and the whole modal picture are
wrong together, and none of it shows up as an error.

## Decision

**A room is the volume a person standing in it would call one room**, built from
regions by four rules, in `reverberate.geometry.rooms`. `isolated_storey` -- the
only place the pipeline picks a room -- goes through them, so a run that asks
for `living room` gets the room and not the slice. The manifest records
`room_regions` beside `room`: what was simulated, against what was asked for.

1. **The passage must be wider than a door**, `DOOR_MAX_M = 1.10 m`. A dressing
   room shut by a door keeps its own volume, whatever the dataset labels it.
2. **A wall separates, whatever the hole in it.** The shared boundary must carry
   less than `WALL_MAX_M = 1.05 m` of wall.
3. **A hallway counts as a door.** It never joins the rooms at its two ends; it
   is itself attached to the largest room it opens onto. Applies to `hallway`
   and `entryway/foyer/lobby`.
4. **Nothing under `MIN_ROOM_M2 = 2 m2` is a room.** Such a region joins the
   neighbour it opens onto widest. The `closet` label gets no other treatment.

A merged room is named by its parts, largest first: `living room-kitchen`.

## Why these thresholds and not others

**Both are read off the measurements.** HSSD models no room door anywhere --
not in the stage, not as an object; the 68 articulated objects of `102344403`
are all furniture -- so a doorway can only be read as a gap the walls leave in a
section at `WALL_SECTION_HEIGHT`. Over the three scenes in play:

- the gaps stop at **0.96 m** and start again at **1.33 m**, and a single leaf
  cannot be wider than about a metre;
- among the boundaries that pass that test, the ones that ought to dissolve
  carry at most **0.79 m** of wall and the ones that ought to hold carry at
  least **1.35 m**.

Each threshold sits in the middle of an empty band. `MIN_ROOM_M2` is a decision
rather than a measurement, and is labelled as one.

Both are measured through `wall_footprint`, and that is the only wall band
anything may use: widening it by 5 cm moves every number here.

## Consequences

- **A room must be unioned with its own doorways.** HSSD's regions do not tile
  the plan; the wall band holds them 0.15 m apart. A plain union of the parts of
  the 126.3 m2 room comes back as three disjoint prisms, which is not a volume
  anything can be simulated in. Rooms are joined the way `build_storey` already
  joined a whole storey.
- `merge_closets` and `MIN_ROOM_AREA` are gone. `Storey` carries `everyday`.
- `OUTDOOR_LABELS` gains `porch/terrace/deck/driveway`, which 71 regions carry.
  On `102344403` the drive opens onto the garage through 8.64 m of doorway, so
  without it a 110 m2 outdoor slab joined the interior.
- `viz.room_surfaces.shell_mesh` and `select_region` are deleted. They extruded
  a single region as if it were a room; nothing in production called them, and
  leaving them was leaving the old idea loaded.
- The exported room model is `room_only.json`, not `bedroom_only.json`: in W39
  that file held a living room. The name was also load-bearing -- `grid_page`
  picked the room's sealed-volume census by `scene_name.startswith("bedroom")`
  -- so producer and consumer now share a `ROOM_SCENE` constant.
- On the three scenes in play, **49 interior regions become 29 rooms**: 20 to
  15 on `102344403`, 19 to 10 on `102344022`, 10 to 4 on `102344094`. The counts
  are pinned in `tests/test_rooms_on_hssd.py`, so they are checkable rather than
  asserted. The dataset over-cuts by about a third.
- **Every result produced before this is against the old definition** and cannot
  be compared to one produced after it without saying so. `w39_living_8k` is the
  case in point.

## What holds it in place

- `tests/test_rooms.py` guards each rule on synthetic geometry, including the
  disconnection trap.
- `tests/test_rooms_on_hssd.py` pins the counts, the 461.8 m3 and one refusal
  against the real scenes. Marked `slow` because it needs the download; run it
  with `make test-slow`.
- `scripts/room_openings.py` prints every shared boundary with the rule that
  decided it. It is where the two thresholds above were read off.
