# Scenes that are not dwellings: the street, the garden, the train, the station

Date: 2026-10-07

Status: **an analysis, written for a decision. Nothing was built, solved or
rendered, and no card was rented.** Lot L28 of ADR 0016. The owner's request:
in the long run the simulator renders scenes far more diverse than
dwellings, and its engines compute impulse responses in semi-outdoor
settings (a street, a train, a station, a garden); what does that imply for
the solver, and what audio has to be found.

Every figure carries its kind. **Measured**: read in this repository's own
records, with the document that holds it. **Computed**: arithmetic on
measured figures or on a closed formula, reproducible from what is written
here. **Estimated**: a judgement or a value from the literature, not checked
by this lot. What was read in the code is cited by file and line, on the
branch `clarify-simulator` at `bbfd2ce4`; what is inferred from it says so.

## 1. The answer

1. **The wave band is the wrong tool for most of these settings, and is not
   needed in them.** A 60 m piece of street to 1 kHz is 2.1 thousand million
   nodes, 177 GB and 43 minutes of an RTX 3090 a source position (computed),
   and a car's pass is 750 positions. The same street has a mode every few
   hertz from 20 Hz up: nothing in it is modal, its surfaces are tens of
   wavelengths across, and a geometric band is right there down to 100 or
   200 Hz. The wave band is dear where it adds least. It stays cheap and
   useful in exactly one family: **vehicle interiors**, where a carriage to
   1 kHz is 10 s a position and a car's cabin could be solved to 4 kHz in
   14 s (computed).
2. **What replaces it outdoors is small: the mirror's own image paths
   rendered through the low band.** The pack already has a low band engine
   that is not a solve, the free field stand-in
   (`FreeFieldPairs`, `trace/engines.py:306`). A stand-in that sums the direct path, the
   ground's image and the facades' images is the same idea one step on, costs
   no card, and is exact where the surfaces are large. It is the first lot
   proposed (section 7).
3. **A fast source breaks the engine in three places that are constants,
   not physics.** The delay line gives the Doppler shift exactly, at any
   speed (`render/early.py:301-306`). But a recipe refuses any speed over
   1.5 m/s (`scenes/validate.py:60`); a path whose apparent source moves
   more than 0.30 m in a step is cut in two and cross-faded
   (`render/pack.py:87`), which is every path of a source faster than 6 m/s;
   and under the crossover a moving source is the weighted sum of two fixed
   responses (`scene-pack.md`, "Source side"), which has **no Doppler shift
   at all**. A car at 50 km/h would be shifted by 4 % above 1 kHz and by
   nothing below, where an engine's sound is.
4. **Most of what makes these places sound as they do is not a point source
   and needs no impulse response.** Traffic hum, rain, wind, a crowd, a
   carriage's rumble are extended and diffuse. They are ambient beds:
   ambisonic recordings laid on the output. Neither the recipe, the pack nor
   the clip library can hold one today (a clip is mono,
   `clip-library.md`, "The files"). One lot adds them, and that lot alone
   changes what every scene sounds like, dwellings with an open window
   included.
5. **The audio is the harder half, and licences decide it.** Section 5.
   One dataset gives beds at order 4 under a licence that allows commercial
   use (EigenScape, 10.7 h); the three closest matches to this request
   forbid it; no bed of a train's interior and no dry recording of a
   vehicle passing was found at all. And **HSSD itself, on which every
   present scene stands, is CC BY-NC 4.0** (section 4).
6. **Neither engine transmits sound through a pane or a shell, and it
   should not be taught to.** A pane as a few secondary sources behind a
   transmission loss filter is one lot and covers the open and the closed
   window. A train's shell is not a transmission problem: what is heard
   inside is structure-borne, and is a bed.

**The first step recommended**: an *open-air profile*, a procedural terrace
or garden (a ground, zero to three facades, a few boxes), with the low band
by images, the mirror above as it is, slow sources, and ambient beds. Three
lots and one of validation, under 1 USD of card a scene (estimated from the
mirror's share of the first scene, `performance-audit.md` section 1).

## 2. The settings, by what matters acoustically

Dimensions and reverberation times are estimated, from the literature's
usual ranges; section 6 names the measurements that would replace them. The
Schroeder frequency `2000 sqrt(T / V)` is computed from the two columns
before it and says from where the field is dense enough in modes for a
geometric band to be right; a dwelling's room of 50 m3 and 0.5 s has
200 Hz.

| setting | size | volume of air to model | reverberation | Schroeder | what a listener hears that a dwelling never has |
| --- | --- | --- | --- | --- | --- |
| **car** | 2.5 x 1.5 x 1.2 m | 3 m3 | 0.05 to 0.1 s | 330 Hz | a talker at 0.5 to 0.9 m who does not face the listener; glass 0.3 m from an ear; road and wind noise at 65 to 75 dB(A), most of it under 500 Hz; no late field at all |
| **bus** | 12 x 2.5 x 2.2 m | 66 m3 | 0.3 to 0.5 s | 150 Hz | rows of talkers behind seat backs; an engine at one end; doors, announcements from ceiling loudspeakers |
| **train carriage** | 25 x 2.8 x 2.3 m | 160 m3 | 0.2 to 0.5 s | 100 Hz | a long narrow cavity: a voice four rows away arrives by the ceiling; rolling noise from the floor, 65 to 75 dB(A); tunnels and passing trains change the level by 10 dB in a second |
| **cafe terrace** | 12 x 8 x 4 m, a facade on one side, an awning or nothing above | 380 m3 | none to 0.4 s | not a room | many near talkers (babble at 1 to 4 m), cutlery, a street behind; one strong reflection from the facade and one from the table |
| **garden, park** | 20 x 15 m, open above | 1500 m3 for 5 m of height | none: discrete reflections | not a room | a ground that is not hard (the ground dip, 200 to 600 Hz); birds, wind in leaves, a mower; sources at 10 to 50 m with no reverberation to hide in |
| **covered platform** | 60 x 8 x 6 m, open at the sides and ends | 2900 m3 | 1 to 2.5 s | not a room | a canopy that returns everything; a train arriving (a source 100 m long at 10 m/s); announcements from a line of loudspeakers, each delayed by its distance |
| **street** | facades 12 to 20 m high, 15 to 25 m apart, the sky open | 21 600 m3 for 60 m of it | 1 to 3 s between facades, less in a wide street | not a room; cross modes every 10 Hz | vehicles passing at 8 to 14 m/s with their Doppler shift; a flutter between facades 44 to 73 ms apart; sirens and horns 20 dB over everything; a noise floor of 65 to 80 dB(A) that is a line and not a point |
| **shopping centre, atrium** | 60 x 20 x 15 m | 18 000 m3 | 1.5 to 4 s | 24 Hz | a late field that is long and loud; music and announcements from a ceiling of loudspeakers; hundreds of voices, none intelligible |
| **station hall** | 80 x 30 x 20 m | 48 000 m3 | 3 to 10 s | 20 Hz | the same, longer; an announcement that is speech, loud, and not the voice to follow; a critical distance of 3 to 5 m, beyond which a companion's voice is mostly reverberation |

Three things run through the table. **The level**: every one of these
places is 10 to 25 dB louder than a living room, so the listener's
companion speaks louder and nearer (the Lombard effect changes the voice
itself, which a dry clip read in a booth does not have). **The late
field**: it is absent (car, garden), short and coloured (carriage), or far
longer than the 1.2 s this project's responses hold (station). **The
sources**: in a dwelling every source is a point; here the loudest ones are
lines, surfaces and volumes.

## 3. What it implies for the solver

### 3.1 The wave band

**What the boundary code is today** (read in the code).

- A wall is a **locally reacting admittance** on the nodes of air that touch
  it: a sum of series branches `1 / (D s + E + F / s)`
  (`wave/lowband/walls.py:109-114`), refitted to seven branches over 40 to
  1500 Hz (`walls.py:77`). Nothing propagates inside a wall: the nodes
  buried in walls and furniture are never updated (`low-band-solver.md`,
  "What a solve is made of").
- **A termination at the air's impedance exists already**: one branch
  `[0, 1, 0]`, an admittance of one at every frequency
  (`wave/lowband/outside.py:55`), put on an opening by
  `outside.without(closure="open")` (`outside.py:175`, `235-239`). It
  absorbs a wave that meets it squarely: 45 dB in a duct under its first
  cross mode (measured, `solver-boundary.md` section 5). It is off by
  default.
- **The box itself ends in a first-order absorbing layer** one node inside
  each face, with `Q` the count of faces a node lies on
  (`wave/lowband/problem.py:198-210`), updated as
  `(partial + l Q previous) / (1 + l Q)` (`wave/lowband/solver.py:285-289`).
  It is the engine's, kept bit for bit.
- **A wall may not reach that layer.** `problem.py:463-464` raises
  `NotImplementedError("a reached boundary node lies on the box's absorbing
  layer")`. A storey sits inside its shell and never meets it. A ground or
  a facade that runs out of the domain meets it by definition: **an open
  scene is refused today by that line**, before any question of quality.
- The card's node mask keeps `Q` in two bits (`solver.py:420-425`,
  `MASK_Q_SHIFT = 12` under `MASK_BOUNDARY = 0x4000`): values 0 to 3, the
  count of faces, and no room for a graded profile.

**What each open boundary costs in this scheme.** The scheme is a two step
update of the pressure alone (`solver.py:281-284`); there is no velocity
field to split.

| boundary | what it is here | reflection | nodes added | engineering |
| --- | --- | --- | --- | --- |
| admittance of one on the open faces | exists (`outside.py:55`); the guard of `problem.py:463` to lift, and the faces to mark | `(1 - cos a) / (1 + cos a)` at `a` from the normal (computed): none at 0, -23 dB at 30 degrees, -15 dB at 45, -9.5 dB at 60, -4.6 dB at 75 | none | half a lot, with a proof in a duct and over a plane |
| the same, with the box grown | the open face is put further off, so that what it returns is late, weak and near the normal | a source at 1 m and ears at 1.6 m under an open face 5 m up: -9 dB re the direct sound when they are 15 m apart, -26 dB at 5 m; under one 10 m up and 15 m apart, -21 dB (computed, reflection and spreading) | in proportion to the height added | none more |
| a graded sponge | the absorbing layer's own update over 20 to 40 nodes with `l Q` rising | -30 to -40 dB over most angles (estimated, not tried here) | 0.45 to 0.9 m a face: 30 % on a garden of 20 x 15 x 5 m, 5.5 % on a street whose sides are facades (computed) | one lot: the mask's two bits become a table, on both kernels and the `numpy` twin |
| a perfectly matched layer | two to four auxiliary fields a node of the layer and a kernel of its own; where it meets a lossy ground, both updates on one node | -60 dB and better with 8 to 12 nodes (estimated) | 10 % on the garden | two lots at least: a new kernel, its `numpy` twin to the bit, a stability proof beside the branches, slabs |

The card's time is not the price of any of them; the engineering is. **The
first row is enough for the cases where the wave band should be used at
all**, which are small (below): a terrace or a platform under a low
crossover, with the open faces put twice the largest source distance away.
A perfectly matched layer is not recommended.

**Size and cost of each setting.** Scaled from the measured solve of
hssd_0076: 47 371 003 updated nodes, 32 769 steps (1.2 s), 70.5 s on an RTX
3090 with walls of seven branches (measured, `solver-boundary.md` section
2), which is 4.54e-11 s a node and step, and 0.120 s of card for a cubic
metre of air and a second of response at 10.5 points a wavelength to
1500 Hz (computed). At 7.2 points the measured ratio is used, 36 s against
110 on an RTX 3080 (`low-band-solver.md`), which is 0.039 s for the same;
the count of nodes alone would say 0.027. Memory is given between two
bounds: 16 bytes a node, two fields and a mask, and 85 bytes a node, the
"about 4 GB for one source" of the dwelling's grid (`low-band-solver.md`,
"Memory"). An open scene has fewer lossy nodes than a dwelling, so its
truth is nearer the low bound and its time a little under what is written.
USD at 0.155 an hour.

| setting | air, m3 | response | nodes at 10.5 | memory, GB | s a position | nodes at 7.2 | s a position |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the dwelling, measured | 490 reached | 1.2 s | 47.4 M | 4 | **70.5** | 21.2 M of box | 36 on a 3080 |
| car | 3 | 0.15 s | 0.3 M | under 0.1 | 0.1 | 0.1 M | under 0.1 |
| bus | 66 | 0.4 s | 6.4 M | 0.1 to 0.5 | 3.2 | 2.1 M | 1.0 |
| train carriage | 160 | 0.5 s | 15.5 M | 0.2 to 1.3 | 9.6 | 5.0 M | 3.1 |
| terrace | 384 | 0.3 s | 37 M | 0.6 to 3.2 | 14 | 12 M | 4.5 |
| garden | 1500 | 0.3 s | 145 M | 2.3 to 12 | 54 | 47 M | 18 |
| covered platform | 2880 | 1.0 s | 278 M | 4.5 to 24 | 345 | 90 M | 113 |
| street, 60 m | 21 600 | 1.0 s | 2088 M | 33 to 177 | 2590 | 673 M | 850 |
| atrium | 18 000 | 2.5 s | 1740 M | 28 to 148 | 5390 | 561 M | 1760 |
| station hall | 48 000 | 5 s | 4640 M | 74 to 394 | 28 700 | 1496 M | 9400 |

All computed. A source position of the street is 0.11 USD and one of the
station 1.24 USD, where a dwelling's is 0.003; and a moving source is a
position every 8 cm (`scene-recipe.md`, "rails"): a car crossing 60 m is
750 positions, 22 days of one card and 84 USD for one pass in one lane.
The grid above 24 GB also needs the slabs of `wave/lowband/slabs.py` over
several cards, which exist and were proven on `numpy` only.

**Where it stops being affordable**: at the garden if sources move, at the
platform in any case. The line is not a matter of taste. Cost goes as the
volume times the response's length times the fourth power of the frequency
limit (`performance-audit.md` section 10).

**What replaces it**, in the order recommended.

1. **Geometric only, where nothing is modal.** Station, atrium, street,
   platform, garden. Their Schroeder frequency is under 30 Hz or they are
   not rooms. The low band is the image paths rendered whole (section 3.2).
   The loss is diffraction round small things under 500 Hz (a parked car, a
   pillar, a bench): they are transparent there in truth, and a geometric
   band that ignores obstacles smaller than a wavelength under the
   crossover is nearer the truth than one that mirrors on them.
2. **A lower crossover, where a cavity is modal but large.** The cost at a
   crossover of 250 Hz is 1/256 of that at 1 kHz: the street 10 s a
   position and 2.8 GB, the platform 1.3 s, the station 112 s (computed).
   And the rail's pitch grows with the wavelength, 8 cm at 1 kHz to 32 cm at
   250 Hz, so a pass is 190 positions and not 750. ADR 0016 rejected a
   crossover at 500 Hz *for dwellings*, because the mirror is not validated
   under 1 kHz there. That reason is the dwelling's: its rooms are a few
   wavelengths across. It does not carry to a street, and it would have to
   be shown, not assumed (section 6).
3. **The wave band whole, where the cavity is small**: car, bus, carriage.
   This is the one family where it is cheaper than in a dwelling, and where
   it earns more: a car's cabin is modal to 330 Hz and a seat's headrest is
   a wavelength at 1 kHz. To a crossover of 4 kHz a car is 18.6 M nodes and
   14 s a position (computed), and there are five positions in a car.
4. **An analytic ground**, for the garden: a point source over an impedance
   plane has a closed form (section 6), which is both the engine and its
   own validation.
5. **The wave band round the listener only**, the far field brought in on
   the box's faces as an incident wave. It would give the table, the bench
   and the listener's own neighbourhood at a dwelling's cost. The solver's
   drive is point sources on nodes; an incident field on a surface is new
   code and a new proof. Not recommended before the others are exhausted.

### 3.2 The mirror band

**What already works without walls** (read in the code; none of it was run
on an open scene).

- The direct path and the images need facets, not a room: a ground and a
  facade are facets. The tree is built to order 3 inside 80 ms, with flutter
  sequences between parallel facets to order 6 (`mirror/ism.py:61-78`).
- **A ray that leaves is counted and dropped**: `tally["escapes"]`
  (`mirror/tracer.py:412`), and the ray ends there. The tracer does not
  assume a closed shell.
- Diffraction round an obstacle is Maekawa's loss on the shortest way round
  an occupancy grid of 0.10 m (`mirror/occupancy.py:1-8`, `38`). It applies
  to a wall between gardens or a parked van as to a door frame.
- Air absorption is ISO 9613-1, with the recipe's humidity and pressure.

**What breaks, or is unproven.**

| what | why | where | remedy |
| --- | --- | --- | --- |
| the tree stops too early | facades 15 to 25 m apart return every 44 to 73 ms; order 3 inside 80 ms holds one or two crossings of a street whose decay is 1 to 3 s | `ism.py:61`, `72` | few facets make a deep tree cheap: the tree's size goes as the facets to the power of the order, and 164 facets to order 3 (hssd_0076, 1.1 s a source, measured) is 20 facets to order 5 (computed). Order and window become the setting's, not constants |
| late echoes become noise | the tail is a histogram of 2 ms bins rendered as shaped noise (`render/tail.py:3-16`). In a room a bin holds hundreds of arrivals. Outdoors it holds one, from one facade, and one echo rendered as a 2 ms burst of noise is a rasp, not an echo | `render/tail.py`, `mirror/render.py` `tail_from_histogram` | what the images do not hold must be little: the deeper tree above, and scattering as the only thing left to the rays |
| too few rays return | 100 000 rays and a receiver sphere of 0.20 m (`mirror/rays.py:60`, `67`). With the sky open most rays leave after one to three bounces; a bin's energy is then a handful of hits and its estimate is shot noise (estimated, to be measured: section 6) | `rays.py` | count the hits a bin; more rays where a bin is under a stated count (the tracer makes 1e8 segments a second on a card, `ray-tracer.md`), or no tail at all where the hits say there is none (a garden) |
| the response is too short | rays last 1.2 s, 600 bins (`rays.py:61`); a station decays for 3 to 10 s | `rays.py:61`, the pack's 600 bins | the duration becomes the setting's; a pack's tail grows in proportion and stays small beside its low band |
| the calibration is a dwelling's | absorption scales, a scattering for the shell, a tail gain a band (`mirror/parameters.py:37-45`), fitted against the wave field of one dwelling; D8 of the audit says it returns 117 % of a bounce on the shell | `parameters.py` | outdoors, the catalogue's values uncalibrated (`Parameters()`, "nothing calibrated"), and validation against analytic cases instead of a fit: there is no wave field to 8 kHz of a street to fit against |
| the diffuse assumptions of the audit | D4 (the tail's comb, 2.7 dB) is heard more on a sparse tail; D5 (the tail's first 10 ms dropped) drops the facade's scattered return at a terrace; D6 (a power join after 10 ms) assumes two bands that have lost their coherence, which a single echo has not; the tail's `scale` falls back on the median of the cells that see the source, a level that does not exist outdoors, where level falls 6 dB a doubling | `chain-audit.md` section 3; `scene-pack.md`, "tail" | D4 and D5 are open in any case; D6 has no object without a wave band; the fallback must become the spreading law's |
| tails interpolated in energy between sites 0.80 m apart | measured to hold in a dwelling, where the late level differs by 0.3 to 0.7 dB between neighbours (ADR 0016) | `mirror/tails.py` | holds better outdoors for far sources, worse within a metre of a facade; to measure |

**The ground.** A material is octave absorption and a scattering
coefficient (`materials/db.py:14`, `68-73`): an energy, without phase and
without angle. For asphalt, concrete and paving that is right: the
reflection is +1 to a few per cent and the interference of the direct path
and the ground's image, which the engine makes by adding two delayed paths,
is the true one. For grass, soil, gravel and snow it is not: the reflection
turns towards -1 at grazing incidence, the sum has a broad dip between 200
and 600 Hz for ears at 1.6 m and sources 5 to 30 m away, and a real band
gain cannot make it. The exact reflection of a spherical wave on an
impedance plane is known in closed form, with the ground's impedance from
one number, its flow resistivity (section 6). It is a short filter a path:
in the pack, one more factor on the ground's image, a band at a time if the
bands are made narrower under 1 kHz, or a filter of its own. **This matters
for the garden and the park only**; a street is hard.

**Vegetation.** As an obstacle it is nearly nothing at these distances:
ISO 9613-2 gives dense foliage 0.02 to 0.12 dB a metre by octave, for paths
of 20 m and more through it (section 6; read at second hand). Ten metres of hedge is a decibel at 4 kHz. It is
worth nothing as an obstacle and a great deal as a **source**: wind in
leaves is a bed.

**Distance.** The air's loss is filtered at distances 8 m apart, twelve of
them at most, and linear between two (`render/early.py:51-55`); beyond 96 m
the spacing widens. At 200 m the loss at 10 kHz is some 30 dB (estimated,
ISO 9613-1 at 20 C and 50 %) and a linear step between distant nodes is no
longer 3 %. The nodes should be geometric in distance. Refraction by wind
and temperature gradients, and the loss of coherence by turbulence, change
levels by several decibels beyond 100 to 200 m and by little under 50 m
(estimated). **Recommended: point sources stop at 100 m; beyond, a source
is in a bed.** A hearing aid's problem is within 10 m.

**Temperature.** A recipe's temperature must be 20 C, because the wave
solve fixes the speed of sound (`scene-recipe.md`, "atmosphere"). Outdoors
it runs from -10 to 35 C, 325 to 352 m/s. With no wave band the constraint
goes; with one, the grid's step would have to follow.

### 3.3 Sources that move fast

**Does the delay line give the Doppler shift? Yes, above the crossover, and
exactly.** Between two steps the apparent source and the head each move in
a straight line, the delay is `|q(t) - l(t)| / c` sample by sample
(`render/early.py:301-306`) and the dry signal is read at that moving delay
through a windowed sinc at twice the rate (`render/delay.py:59-82`; -94 dB
from 1 to 20 kHz, measured). Nothing in it assumes a slow source. The
native kernel does the same (`render/native.py:8-9`, `351-353`).

**What refuses or spoils a fast source** (read in the code; the
consequences are inferred, nothing was rendered).

| | where | effect at 50 km/h (13.9 m/s, 0.69 m a step) |
| --- | --- | --- |
| the recipe's speed limit | `scenes/validate.py:60`, `MAX_SPEED_M_S = 1.5` | the recipe is refused |
| the jump rule | `render/pack.py:87`, `JUMP_M = 0.30`; `scene-pack.md`: "a real image moves at the source's speed, 7.5 cm a step at most" | every path of a source faster than 6 m/s is two paths cross-faded over a step: the shift is replaced by a comb. The rule has to scale with the source's speed, or the step to shorten |
| the step of 50 ms | `render/pack.py:82` | a straight chord of 0.69 m is exact on a straight road; a path's birth and death (a car passing a gap between buildings) fade over 50 ms, which is 0.7 m of road: acceptable; the direction is evaluated 8 times a step, and a car passing at 5 m turns 0.14 rad a step, 0.2 % of the top order (computed from the format's own bound) |
| **the low band has no Doppler shift** | `scene-pack.md`, "Source side": the output of two solved positions cross-faded with a weight linear in arc length | at 1.5 m/s the shift is 0.4 %, 8 cents, and its absence is not heard. At 13.9 m/s it is 4 %, 70 cents: an engine's harmonics under 1 kHz stay where they are while its hiss above moves. This is a property of sampling the source, and no pitch of rail cures it |
| positions every 8 cm | `scene-recipe.md`, "rails" | 1250 positions for 100 m of one lane |
| tail sites every 0.80 m | `scene-pack.md`, "tail" | 125 sites for the same; 16 s each as ADR 0016 measured it, less with the tracer of `ray-tracer.md` |

So a vehicle cannot have a sampled low band. With the low band by images
(section 3.1, first remedy) there is nothing to sample: every image is a
moving delay line from the bottom of the spectrum up, and a road is a rail
with a speed. That is the second argument for the geometric low band
outdoors, independent of its cost.

A source inside a moving vehicle is at rest with the listener: no shift.
The vehicle's own motion is heard only in its noise, which is a bed.

### 3.4 Sources that are not points

| source | what it is | how to render it | what the formats need |
| --- | --- | --- | --- |
| one car, one bird, one door, one announcement loudspeaker | a point | the engine, as today | nothing (a speed, section 3.3) |
| a busy road at 20 to 100 m | a line of incoherent points | under 30 m, 5 to 20 points along it, each its own clip of a pass; beyond, a bed | nothing for the points; each costs what a source costs |
| traffic hum, rain, wind in trees, a crowd, a ventilation plant, a carriage's rumble | a volume or a surface, without position | **a bed**: an ambisonic recording added to the output, with no impulse response; turned into the scene's frame, levelled, faded | a source `kind` `"bed"` with no segments, no stations and no low band; a clip of 4, 16 or more channels, where a clip is mono today (`clip-library.md`, "The files"); in the pack, a group that names the clip, its gain and its rotation and holds no arrivals |
| rain on a canopy, a fountain, an escalator | a surface near the listener | 4 to 12 incoherent points on it, each with a different stretch of the same clip | nothing |
| a public address system | several loudspeakers playing one signal, each delayed by its cable and its distance | several sources with the same clip: they are coherent, and the listener hears the comb and the echo that make announcements unintelligible | to check that two sources may share a clip and its offset; a loudspeaker's directivity table beside `voice_v1` |

**A bed is not free of consequences.** A first order recording fills four
of the output's 64 channels. Beside point sources that are sharp to order 7
it is a blur, and a separation model may learn that what is blurred is
noise, a cue no real hearing aid has. Three answers, in order of cost: (a)
use beds of order 3 or 4 where they exist (section 5); (b) sharpen a first
order bed parametrically, direction and diffuseness a band, as spatial
audio coding does; (c) build the bed in the engine from 20 to 50 far point
sources of mono recordings, which is sharp, labelled, free of unknown
voices, and costs 20 to 50 sources. The owner's choice (section 8).

**A bed carries its own room.** A recording made in a street is that
street. Laid over a rendered garden it is a garden with a street's
reverberation. For noise that is acceptable; it is why the point sources,
which carry the scene's geometry, must stay dry.

### 3.5 Through a pane, through a shell

Neither engine transmits. In the wave band a wall is a surface admittance
and the nodes behind it are not updated; in the mirror a facet reflects or
occludes. A source outside a closed dwelling is silent inside it today.

**The cheapest sound approximation: the pane as secondary sources.**

1. Outside, the sound reaching the pane is computed as for any listener
   there: the direct path and the ground, on the facade (where the pressure
   doubles).
2. It passes a **transmission loss filter**, a level a band: a single pane
   of 4 mm some 29 dB weighted, a double glazing 30 to 35 dB with a dip of
   10 dB at its mass-air-mass resonance near 200 Hz and another at
   coincidence near 3 kHz, an open window nothing over its aperture
   (estimated, to be replaced by tabulated values, section 6).
3. Inside, the pane is 4 to 9 incoherent point sources on its face,
   radiating into the half space, each a **station of the dwelling**: the
   existing engines, wave band included, do the rest.

This is nearly what the first library already does with its `street` noise,
a point source in the room (`clip-library.md`, `clarify_v1`), with the two
things it lacks: the filter and the place. Cost: the stations on panes (the
export knows which facets are glazing: 11 sheets on hssd_0076,
`scene-pack.md`), a filter a source, no change of either engine.

**Its limits.** The direction the outside sound comes from is lost, save in
level. A pane radiates as a piston above a few hundred hertz and not as
points: its beam follows the angle of incidence, which a passing car
sweeps. Flanking paths (the frame, the wall, a ventilation slot) set the
true insulation of a good window and are not in it. An open window at low
frequency is a coupling of two volumes, which the wave band could do
exactly with the source outside the opening and the `"open"` closure of
`outside.py`; that is the one case worth more.

**A vehicle's shell is another matter.** What is heard in a carriage is not
the outside transmitted: it is the floor, the walls and the glazing
vibrating, and the ventilation. Modelling it is structural dynamics, which
is out of scope and should stay there. Inside a vehicle the noise is a bed
recorded in a vehicle, and the engine renders the voices.

## 4. Geometry: where the scenes come from

**How this section and the next were verified.** Each source's own page was
opened on 2026-10-07 through a tool that fetches a page and summarises it.
"Seen" means the fact was on the source's own page that day; "secondary"
means it was read in a search result or on another site's page; "not
verified" means the page failed or did not say. A summary is not a legal
reading: **every licence below is to be read again on its page, by a
person, before anything is downloaded.**

**First, the dwellings themselves.** HSSD is published under **CC BY-NC
4.0** (seen, https://huggingface.co/datasets/hssd/hssd-hab): non-commercial.
Every scene this project has rendered stands on it. That is not this lot's
question, and it is the first one a commercial use of any render meets; it
is put to the owner in section 8.

| setting | source | what it holds | licence, commercial use | status |
| --- | --- | --- | --- | --- |
| street, terrace, square | 3DBAG, https://docs.3dbag.nl/en/copyright | every building of the Netherlands, some 10 million, LoD1 and LoD2, CityJSON, OBJ, IFC | CC BY 4.0: yes | seen |
| | swissBUILDINGS3D 3.0, https://www.swisstopo.admin.ch/en/landscape-model-swissbuildings3d-3-0-beta | Switzerland; roofs, facades and footprints apart; 30 to 50 cm | free geodata terms, attribution: yes | seen |
| | Berlin, Hamburg, Vienna, Montreal LoD2 | CityGML | dl-de/zero-2-0, dl-de/by-2-0, CC BY 4.0, CC BY 4.0: yes | secondary |
| | PLATEAU, https://www.mlit.go.jp/plateau/open-data/ | 306 Japanese locations, CityGML; its standard defines buildings to LoD4, railways, underground structures | Public Data License 1.0, stated compatible with CC BY 4.0: yes | seen; **whether stations and underground malls at LoD3 or 4 can be downloaded was not verified** |
| | OpenStreetMap, Overture buildings | footprints, heights where tagged | ODbL: yes, with share-alike on a derived database; whether a render is one is a lawyer's question | seen |
| | Google Photorealistic 3D Tiles, https://developers.google.com/maps/documentation/tile/policies | | the policy forbids storing, extracting and machine interpretation: **no** | seen |
| | CARLA's towns, https://github.com/carla-simulator/carla | hand made towns: roads, pavements, buildings, vegetation, with semantic classes | code MIT, assets CC BY: yes | seen; the count of towns and the export to a mesh were not verified |
| | KITTI-360, Virtual KITTI 2 | | CC BY-NC-SA 3.0: **no** | seen |
| garden, park, trees | Infinigen, https://github.com/princeton-vl/infinigen | procedural terrain, trees, plants; rooms | BSD 3-Clause: yes | seen |
| halls, stations, malls | Matterport3D, HM3D, Gibson, Replica | scans of buildings, homes mostly | non-commercial, each: **no** | seen (HM3D, Replica), secondary (the two others) |
| | Stanford 2D-3D-S, ScanNet++, Structured3D | offices, rooms | own terms, the commercial clause not read | not verified |
| | an open, labelled, watertight model of a station, a mall or an airport | | **none was found** | |
| vehicle interiors | Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse | 800 000 objects; licence by object: CC BY 4.0 for 721 000, CC0 for 3500, non-commercial for 77 000 | yes for the CC BY and CC0 objects, object by object | seen; **how many are a carriage, a bus or a cabin with its interior was not counted** |
| | an open CAD of a carriage or a cabin made for acoustics | | **none was found** | |

What published acoustic simulation has used: SoundSpaces (Replica,
Matterport3D), SonicSim (Matterport3D), GWA (furnished houses). All indoor,
and the first two on scans that forbid commercial use (seen on each
project's page). No synthetic acoustic dataset of semi-outdoor scenes was
found.

**The reading.** Three settings have no usable geometry in public: the
station, the mall, the vehicle. Two have more than is needed: the street
and the garden. And the acoustics does not ask for what the datasets hold.
A LoD2 building is a prism with a roof: no window reveals, no balconies, no
shop fronts, no parked cars, no material. What the band above 1 kHz needs
of a street is the facades' planes, their share of glass and their
scattering, and those are parameters, not data.

**Recommended: procedural first, for every setting but the vehicle.**

- *Street*: a ground, two rows of facades drawn from ranges (height 9 to
  25 m, width 12 to 30 m, gaps and side streets, the share of glazing,
  balconies as a scattering coefficient), parked vehicles as boxes. A city
  model is then a source of **ranges** (the distribution of heights and
  widths in 3DBAG), and later of real blocks.
- *Terrace, garden, platform*: a ground of one or two materials, zero to
  three facades, a canopy, boxes for furniture, a hedge as a scattering
  occluder.
- *Station, atrium*: a shoebox or two joined, 10 000 to 100 000 m3, with a
  ceiling's height and a mean absorption drawn to give a decay of 2 to 9 s.
  At a Schroeder frequency of 20 Hz a hall's decay and its first
  reflections are what is heard, and a shoebox with the right ones is a
  station to the ear (estimated; to be judged by listening).
- *Vehicle*: here shape matters, and the wave band needs it to a
  centimetre. A parametric carriage is feasible (a tube, rows of seats,
  glazing strips, luggage racks); a car's cabin is not, and needs a model
  from Objaverse or a purchase, repaired until watertight by hand.

**Materials and labels needed.** The present catalogue (`materials/data`)
is a dwelling's and was not opened by this lot to see what it holds of
these: asphalt, concrete paving, gravel, soil, grass and snow, each with a
**flow resistivity** and not only an absorption; rendered masonry, brick,
curtain wall glazing, a shop front; a canopy of steel sheet or glass; a
seat of foam and cloth, a headliner, a carriage's floor; and for each
facade its share of glass, which sets both its reflection and what passes
through it.

**What of `reverberate.scenes` carries over** (read in the code and in
`scene-recipe.md`).

| | carries over | does not |
| --- | --- | --- |
| stations and seats | a bench, a terrace chair, a carriage's seat are seats; a seat as an exact listening position is what a vehicle needs, and nearly all it needs | a station names its `room` by ADR 0010; a street has none |
| rails | a pavement is a rail; so is a lane, with a speed | `MAX_SPEED_M_S`; a pitch of 8 cm; rails are routed on a storey's free floor (`scenes/layout.py:1-19`, its `Floor` and `Room`) |
| the free floor | a walkable outline is the same idea outdoors | it is computed from a storey's walls and rooms |
| the recipe | sources, segments, activity, listener, atmosphere, the seed | `dwelling` and `assets` are a storey's (`export_sha256`, `voxel_low_key`, `rooms_rule`); unknown keys are refused, so a `site` is a new version of the format |
| heights | 1.70 and 1.20 m | an engine at 0.5 m, a loudspeaker at 4 m, a bird at 8 m: a source needs a height of its own |
| the generator | the drawing of clips, levels and activity | everything it knows of where people stand in a home |

## 5. Audio: what has to be found

### 5.1 What each setting needs

| setting | dry or near-dry points | beds |
| --- | --- | --- |
| street | a car, a van, a bus, a motorcycle, a bicycle passing, each at several speeds, **recorded so that its Doppler shift was never there or can be undone**; an engine idling; a horn, a siren, a bell; footsteps on paving; a door, a shutter; roadworks | distant traffic; a wet road; wind between buildings; a crowd on a pavement |
| garden, park | birds one at a time; a dog; a mower, a hedge trimmer; children; a ball; a gate | wind in leaves; rain on grass; insects; a far road; a dawn chorus |
| car, bus, train | an indicator, a wiper, a seat belt, a door closing; a door's warning tone; an announcement, dry | **the cabin's noise at several speeds, on several roads, with the window open and shut; a carriage's rolling noise, in the open and in a tunnel**: the gap of this whole inventory |
| station, mall | an announcement, dry, to be played through a model of loudspeakers; a suitcase's wheels; a ticket gate; an escalator; a train arriving | the hall's own murmur; a crowd; ventilation |
| terrace | cutlery, a glass, a cup on a saucer; a chair on paving; a coffee machine | babble; the street behind |
| platform | a train arriving, braking, its doors; a whistle; an announcement | wind; a far train; the canopy in rain |

Voices are not in the table: they are the first library's (VCTK, CC BY
4.0), with one thing missing. People speak louder and differently in noise,
and a voice read in a booth at a normal effort, raised by a gain, is not a
voice in a station. A corpus of Lombard speech is to be found; none was
looked for by this lot.

### 5.2 The datasets, as read on their own pages

**Usable in a commercial product, by what was seen.**

| dataset | what, how much | form | licence | voices in it |
| --- | --- | --- | --- | --- |
| **EigenScape**, https://zenodo.org/records/1012809 | 64 recordings of 10 min, 10.7 h, 127 GB: beach, busy street, park, pedestrian zone, quiet street, shopping centre, train station, woodland, eight of each | **order 4**, 25 channels, 48 kHz, 24 bit, an Eigenmike | CC BY 4.0 | not labelled; passers-by to be expected in three of the eight |
| **Urban Soundscapes of the World**, https://zenodo.org/records/10106181 | some 130 recordings in nine cities, 6.2 GB | order 1 and binaural, 48 kHz, 24 bit | CC BY 4.0 on Zenodo; the project's own page says "free to use for research and educational purposes": **to be confirmed in writing** | not labelled |
| **The SoundField library by RØDE**, https://library.soundfield.com/ | "hundreds" of recordings: ambience, animals, people, effects, vehicles | order 1, an NT-SF1 | CC BY 4.0; the terms file by file not verified | not labelled |
| **Freesound, tagged `ambisonic`**, https://freesound.org/search/?q=ambisonic | 713 sounds; 233 of four channels, of which 113 CC0 and 94 Attribution | order 1 mostly; A and B format mixed, two channel orders mixed, said in free text | by sound: CC0 307, Attribution 276, **NonCommercial 130** | not labelled |
| **FSD50K**, https://zenodo.org/records/4060432 | 51 197 clips, 108 h, 200 classes | mono, 44.1 kHz; what each author made, seldom dry | by clip: CC0 19 873, CC BY 23 506, **CC BY-NC 6041, Sampling+ 1777**; and the page asks that its authors be contacted for a commercial use | labelled by clip, weakly |
| **DEMAND**, https://zenodo.org/records/1227121 | 7.4 GB; kitchen, living room, washing, field, park, river, hallway, meeting, office, cafe, restaurant, square, street traffic, bus, car, metro | 16 microphones 5 to 22 cm apart, 48 kHz: **not ambisonic** | CC BY 4.0 by Zenodo's tag, CC BY-SA 3.0 by its own text and by this project's library: **to be read in the archive** | not labelled |
| **SONYC-UST-V2**, https://zenodo.org/records/3966543 | 18 515 clips of 10 s, 51 h, New York's street sensors | far and reverberant | CC BY 4.0 | **labelled**: talking, shouting, amplified speech, a crowd |
| **MAVD-traffic**, https://zenodo.org/records/3338727 | Montevideo's roadsides, 1.1 GB, events labelled by vehicle and by time | 48 kHz, 24 bit, from the roadside | CC BY 4.0 | not labelled |
| **In-Car McVAMPIRE**, https://zenodo.org/records/12806684 | a minivan: responses from 8 seats and 11 head orientations to 14 microphones, 6 loudspeakers, **and driving noise at several speeds** | 14 channels | CC BY 4.0 | none said |
| **QUT-NOISE**, https://github.com/qutsaivt/QUT-NOISE | 7.7 GB: cafe, car, home, street | | CC BY-SA: yes, and share-alike on what is redistributed | not labelled |
| **MUSAN**, https://www.openslr.org/17/ | 11 GB: music, speech, noise | reported at 16 kHz, too narrow for this output (not verified: its README did not open) | CC BY 4.0 | a speech part of its own |
| **STARSS23**, https://zenodo.org/records/7880637 | 11 h in 16 rooms, events labelled in time and direction | order 1, 24 kHz | MIT | speech is most of it, labelled; indoor, and half the band: of little use here |

**Not usable in a commercial product.** Each licence forbids commercial
use. None was seen to forbid training as such; the BBC's is reported to.

| dataset | licence, as seen | what is lost with it |
| --- | --- | --- |
| TAU Urban Acoustic Scenes 2019, 2020 Mobile, Audio-Visual 2021 (https://zenodo.org/records/2589280, https://zenodo.org/records/3819968, https://zenodo.org/records/4477542) | "Other (Non-Commercial)"; the full text is in the archive and was not read | 40, 64 and 34 h of airport, mall, metro station, street, tram, bus, metro, park: the closest match to this request there is |
| **ARTE**, https://zenodo.org/records/3386569 | tagged CC BY 4.0, **and the record's text says "only to be used for non-commercial personal, educational or research purposes"** | 13 scenes made for hearing aid research, mixed order to 4 and to 7 in the plane: cafes, a food court, a train station, a street balcony. Worth one letter to its authors |
| WHAM!, WHAMR!, WHAM!48kHz, http://wham.whisper.ai/ | CC BY-NC 4.0 | 78 h of restaurants, cafes, bars and parks, binaural, from which intelligible speech was already cut |
| ESC-50, https://github.com/karolpiczak/ESC-50 | CC BY-NC (its subset ESC-10 is CC BY) | 2000 clips |
| UrbanSound8K, https://zenodo.org/records/1203745 | CC BY-NC 4.0 | 8732 clips: horns, sirens, idling engines |
| IDMT-Traffic, https://zenodo.org/records/7551553 | CC BY-NC-ND 4.0 | 17 506 vehicle passes |
| TAU-NIGENS (https://zenodo.org/records/4844825), TUT Sound Events 2017 (https://zenodo.org/records/814831); TAU-SRIR (secondary) | CC BY-NC 4.0 or non-commercial | task data of DCASE |
| AudioSet, https://research.google.com/audioset/download.html | the labels CC BY 4.0, the ontology CC BY-SA 4.0, **the audio under no licence**: it is YouTube's uploaders' | the audio; the ontology and the taggers trained on it stay useful |
| the DNS Challenge's noise, https://github.com/microsoft/DNS-Challenge | "the original terms": a part is AudioSet's audio | all but its Freesound CC0 and DEMAND parts |
| BBC Sound Effects, https://sound-effects.bbcrewind.co.uk/licensing | **not verified**: the page could not be read; reported as RemArc, personal, educational and research use, a paid licence for the rest | 33 000 effects |
| CHiME-3 and 4 backgrounds, https://www.chimechallenge.org/challenges/chime4/data; NOISEX-92 | no licence on the page; none found | bus, cafe, street; treat as closed |

L3DAS22 (CC BY 4.0 for its responses, seen), 3D-MARCo, ISOBEL and Clotho
were read and hold nothing these settings need.

### 5.3 What the inventory says

1. **Beds exist for the street, the park, the station and the mall, at
   order 4, under CC BY 4.0, in one dataset: EigenScape.** 10.7 h, 80
   minutes a class. It is the first thing to fetch, and it is not much: a
   model trained on thousands of hours will hear each of its 64 recordings
   many times.
2. **No bed of a train's interior was found at any order under an open
   licence**, and of a car's only a 14 microphone array (McVAMPIRE) and 16
   spaced microphones (DEMAND). Vehicles are where the solver is cheapest
   and the audio is missing. It has to be recorded, or bought.
3. **No dry recording of a vehicle passing was found.** Every pass in every
   dataset read was recorded from a roadside: it carries its own Doppler
   shift, its own ground reflection and its own street. Rendering it as a
   moving source applies all three twice. What a moving source needs is
   the source's signal in its own frame: an engine recorded on board, a
   synthesis (harmonics of the firing rate, and tyre noise), or a roadside
   recording whose shift is undone. This is a piece of work, not a
   download.
4. **Announcements have no cleared source.** A voice of the library, or a
   licensed synthesis, played through a loudspeaker model is the way; it is
   also the truer one, since it then has the hall's response.
5. **The three best matches are closed**: TAU Urban Acoustic Scenes, ARTE,
   WHAM!. Letters could open them.
6. **Share-alike** (DEMAND perhaps, QUT-NOISE) allows commercial use and
   binds whatever is redistributed. What it means for a model trained on a
   render is a lawyer's question, not answered here. The first library
   already holds three DEMAND recordings.
7. **Attribution is an obligation on every CC BY clip**: a manifest by clip
   (author, licence, address) from the first day. The clip library's
   manifest is the place.

### 5.4 Voices in the noise

The owner found voices dominating the present noise clips. They will be in
every recording of a public place.

| dataset | speech in it | what tells |
| --- | --- | --- |
| SONYC-UST | yes | its labels, by clip |
| FSD50K | in some clips of every class | its labels by clip, which are weak: a clip tagged as a car may hold a voice |
| STARSS23 | most of it | its labels, by frame and direction |
| WHAM! | intelligible speech was cut by its authors; babble stays | (closed in any case) |
| EigenScape, Urban Soundscapes, DEMAND, QUT-NOISE, Freesound, the SoundField library, MAVD | to be expected in the public places (estimated: no page says) | nothing |

**How to screen**, from practice, not tried by this lot on these
recordings:

1. The dataset's labels, to reject and never to accept.
2. A tagger trained on AudioSet, on windows of one to two seconds, with a
   low threshold on its speech classes (0.1 to 0.2, not 0.5), and its
   babble and crowd classes read apart. The licence of the tagger's weights
   is to be checked.
3. A voice activity detector as a second vote: alone it fires on birds and
   horns and misses babble.
4. **The test that answers the owner's complaint is intelligibility**: a
   speech recogniser on each window, rejected when it returns confident
   words, and only when the tagger agrees, since a recogniser invents words
   in noise.
5. On an ambisonic bed, **beam first**: a talker is clear in one of twelve
   beams and buried in channel 0.
6. Cut what is flagged, with a fade. Never remove a voice with a
   separator: it leaves traces a separation model would learn.
7. Report the worst clip of each dataset, heard, not a mean.

**And one decision before any of it**: whether babble is wanted. A crowd no
one can follow is what a station is. One intelligible stranger in a bed is
a voice the training labels do not know about.

## 6. Validation: how one would know

**Analytic cases**, exact, free, and the first thing to hold:

| case | truth | what it tests |
| --- | --- | --- |
| a point source over a rigid plane | two paths in closed form | the image low band; the output without a wave band |
| **a point source over an impedance plane** | the spherical wave reflection coefficient (Weyl and van der Pol's form, as computed by Chien and Soroka), the ground's impedance from its flow resistivity by Delany and Bazley or by Miki; resistivities in kPa s/m2: new snow 10 to 30, grass 150 to 300, worn asphalt over 20 000 (secondary; the papers were not opened by this lot) | the soft ground of section 3.2; and the wave band's own ground, should it be kept |
| a half plane, a wedge | Maekawa's curve, which the mirror uses, against the exact wedge solution | diffraction round a wall or a van |
| two parallel rigid planes, the sky open | the image series in closed form | the deep tree of a street, and when to stop it |
| a source in uniform motion past a listener | the shift and the level in closed form | the Doppler shift at 14 and 30 m/s, the jump rule, both bands |
| a duct, and a plane wave at an angle | `(1 - cos a) / (1 + cos a)` | the open face of a grid |
| a foliage belt | ISO 9613-2: 1 dB from 250 Hz to 2 kHz, 2 dB at 4 kHz, 3 dB at 8 kHz for 10 to 20 m; 0.02 to 0.12 dB a metre from 63 Hz to 8 kHz beyond (secondary: as reproduced in a noise mapping manual, https://doku.datakustik.com/CadnaA/en/BebaungBewuchs.html) | that it may be ignored |

**Measurements, as found.**

| what | source | status |
| --- | --- | --- |
| a car's cabin: responses from outside to a 64 capsule sphere inside, 8 directions | IR64-CAR, https://zenodo.org/records/15168578, CC BY 4.0 | seen; a Peugeot 208, **with no geometry** |
| a minivan's cabin, 8 seats | In-Car McVAMPIRE, https://zenodo.org/records/12806684, CC BY 4.0 | seen |
| four cars, talkers to microphones | CAVEMOVE, https://github.com/SPL-FORTH-ICS/CAVEMOVE, CC BY 4.0 | seen |
| 271 responses of daily places at 1.5 m, a forest, a restaurant and a supermarket among them | the MIT survey, https://mcdermottlab.mit.edu/Reverb/IR_Survey.html | seen; mono; its licence stated on a mirror only |
| a forest, in B-format, in summer and under snow | OpenAIR, Koli National Park, https://www.openairlib.net | secondary: the site's certificate had expired; the licence of each entry not verified |
| a street: decay of 1 to 3 s over 144 streets, longer with the facades' height | a survey of 2019 | secondary; the paper's page refused; authors and DOI not confirmed |
| a station's waiting hall: 9.2 s at mid frequencies | one paper of 2014 | secondary; one hall is not a range |
| malls: 1.7 to 3.2 s in four of them; 4 to 5 s elsewhere | conference papers | secondary |
| a carriage: 0.39 s | one paper; a study of five vehicles at Southampton not read | secondary |
| a car: 0.05 to 0.1 s | | **not verified: no page read states a figure** |
| glazing: a 4 mm pane 29 dB weighted, a 4-16-4 unit 30 to 33 dB | trade sources | secondary |
| an open window's insulation | | **not verified** |
| a carriage's shell: some 30 dB | a patent's text | secondary |

**No open measured impulse response of a street, a carriage, a bus, a
station or a covered platform was found.** So the ranges of section 2 rest
on a handful of papers read at second hand, and three things follow. The
analytic cases are the validation, and they are enough for the open
settings, which are a few surfaces. The cabin is the one setting that can
be held against measurements, and only once a geometry is matched to one
of the measured cars. And for the station and the street the bar is
statistical: a decay time and a level against distance inside the
published ranges, and the owner's ear. A day with a loudspeaker and an
ambisonic microphone in one street and on one platform would be worth more
than any dataset found here.

## 7. A staged proposal

A lot is a pull request of the size of those of ADR 0016. Card costs are
estimated from the first scene's ledger (`performance-audit.md` section 1):
a dwelling's scene is 8.6 USD, nine tenths of it the wave solve, and its
rays 0.13 USD.

**Step 1. The open-air profile: a terrace and a garden.** The smallest
thing that is a new setting.

| lot | what | card |
| --- | --- | --- |
| 1a | a `site` beside a `dwelling` in the recipe: a procedural ground, zero to three facades, a canopy, boxes; stations and rails on its walkable outline; outdoor materials | none |
| 1b | a low band engine by images, beside the free field stand-in: the direct path and the tree's images under the crossover, in the pack's own form; the tracer's hits counted a bin, and no tail where they say there is none | none |
| 1c | beds: the source `kind`, the multichannel clip, the pack's group, the engine's sum; EigenScape's park, woodland, quiet street and pedestrian zone fetched and screened for voices | none |
| 1d | validation: the rigid plane; the impedance plane, as a truth that says how wrong a soft ground is before the engine has one; two planes; the open tracer's hit counts | none |

Four lots, and **under 1 USD of card a scene** (estimated: no wave solve;
the mirror's paths on cores and its rays on a card). What it gives: a
conversation at a terrace table or in a paved garden, slow sources, a real
ambience at order 4. What it does not: a soft ground's dip, vehicles, a
tail worth the name.

**Then, in the order of value over cost.**

| step | what | lots | card a scene (estimated) | why here |
| --- | --- | --- | --- | --- |
| 2 | the window: stations on a dwelling's panes, a transmission loss filter, a bed outside | 1 | as a dwelling | it makes every existing dwelling a new scene, with what lot 1c built |
| 3 | fast sources: the speed limit, the jump rule, a road as a rail, the shift checked in closed form; a source signal for a vehicle (section 5.3, point 3) | 2, and the audio | under 1 USD | the street's signature; needs 1b, since a sampled low band cannot do it |
| 4 | the street: procedural canyons, the tree to the depth a canyon needs, traffic as points near and a bed far | 2 | 1 to 2 USD | after 3 |
| 5 | the carriage and the bus: a parametric cavity, the wave band as it is, seats as exact positions, interior beds | 2 to 3, **and the recordings** | 0.2 to 0.5 USD: 10 s a position, a few hundred positions | cheap to solve; blocked by audio, not by the engine |
| 6 | the soft ground: the spherical reflection as a filter on the ground's image | 1 | none | gardens and parks become right under 1 kHz |
| 7 | the hall: a tail of 3 to 10 s, the public address as coherent sources with a loudspeaker's directivity | 2 | 1 USD | a long tail is the engine's memory and time: to measure first |
| 8 | the car: a cabin's model made watertight, the wave band to 2 or 4 kHz, held against IR64-CAR or CAVEMOVE | 3 | under 0.1 USD | the only setting with measurements to meet; its geometry is handwork |
| 9 | a wave band at a low crossover for the platform and the street, with open faces | 2 | 0.5 to 2 USD | only if step 4, heard, lacks something under 250 Hz |

**Recommended not to do.**

- **A perfectly matched layer.** Two lots for a boundary whose only use is
  step 9, which may never be needed.
- **The wave band to 1 kHz in a street, on a platform or in a hall.** Six
  minutes to eight hours a position (section 3.1), for a field that is not
  modal.
- **Transmission through structures**, a vehicle's shell above all.
- **Refraction, turbulence, wind.** Under 100 m they are below everything
  else in this list.
- **Vegetation as an obstacle.** A decibel.
- **City models as scenes**, before procedural streets have been heard.
  They bring a pipeline (formats, repair, materials by guess) and no
  acoustic detail the generator lacks.
- **First order beds alone as the noise of a scene.** A separation model
  would learn the blur.
- **Any audio whose licence is non-commercial, "to try".** What is trained
  on it cannot be untrained.
- **A calibration fitted outdoors** on the dwelling's pattern. There is no
  truth to fit against; use the catalogue's values and the analytic cases.
- **Scanned public buildings.** Those read are all non-commercial.

## 8. For the owner

| | question | recommended |
| --- | --- | --- |
| 1 | HSSD is CC BY-NC 4.0. Is a model trained on its renders within that licence for Clarify's use? | a lawyer's reading before a dataset is generated, whatever is decided here; procedural scenes carry no such question, which is one more reason for them |
| 2 | Is the wave band under 1 kHz a requirement of every scene, or of dwellings? ADR 0016 states it without a scope | of rooms that are modal: dwellings and vehicles. Outdoors and in halls, the images from the bottom of the spectrum, shown against the analytic cases |
| 3 | Which setting first? | the terrace and the garden (step 1), then the window (step 2): they need no audio that does not exist |
| 4 | Vehicles are the cheapest to solve and have no beds under an open licence. Record, buy, or wait? | record: a day in a car, a bus and a train with an ambisonic microphone of order 3 or more gives what no dataset holds, under the project's own terms |
| 5 | Is babble wanted in a bed, or is a bed free of all speech? | babble where it is the place's (terrace, station), unintelligible by the recogniser's test; no single intelligible voice anywhere |
| 6 | A bed at order 1 beside sources at order 7: accept, sharpen, or build beds from points? | order 4 beds (EigenScape) as they are; build from points what has no such recording; no order 1 bed alone |
| 7 | An announcement is speech, loud, and not the target. How is it labelled for training? | a `kind` of its own, so that training may choose; not `far_voice` |
| 8 | Is a source beyond 100 m ever a point? | no: it is in a bed |
| 9 | Do scenes keep 20 C outdoors? | yes while a scene has a wave band; free where a setting has none |
| 10 | Write to the authors of ARTE and of the TAU scenes for a commercial grant, and to FSD50K's as its page asks? | yes: ARTE was made for hearing aids and is the best material read here |
| 11 | A day of measurements in a street and on a platform, for validation? | yes, before step 4: nothing public replaces it |

## 9. What this lot did not do

- Nothing was run: no open scene was traced, no grid built, no bed laid on
  an output. Every cost of section 3 is a scaling of the dwelling's
  measured solve, and every statement about the mirror outdoors is a
  reading of its code.
- The share of rays that return to a receiver outdoors, on which the
  tail's remedy rests, is not known.
- No licence was read by a person. The pages were read through a
  summarising fetch; several could not be opened at all (BBC Sound Effects,
  OpenAIR, IGN, Helsinki, nuScenes), and the reverberation figures of
  streets, stations, malls and carriages are at second hand.
- Whether PLATEAU's stations can be had at LoD3 or 4, how many vehicle
  interiors Objaverse holds, and the true licence of DEMAND.
- Lombard speech, loudspeaker directivities and a vehicle's source signal
  are named as needs and were not searched.
- The materials catalogue was not opened to see which outdoor materials it
  already holds, and whether seven series branches fit a porous ground's
  impedance from 40 to 1500 Hz was not tried.
