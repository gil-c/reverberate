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
   (`trace/engines.py:325-349`). A stand-in that sums the direct path, the
   ground's image and the facades' images is the same idea one step on, costs
   no card, and is exact where the surfaces are large. It is the first lot
   proposed (section 7).
3. **A fast source breaks the engine in three places that are constants,
   not physics.** The delay line gives the Doppler shift exactly, at any
   speed (`render/early.py:300-306`). But a recipe refuses any speed over
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
   Dry recordings of vehicles passing, of doors, of announcements barely
   exist in public; beds exist, mostly first order, mostly with voices in
   them, and several of the best known sets forbid commercial use.
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
  `outside.without(closure="open")` (`outside.py:176`, `235-239`). It
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
| the same, with the box grown | the open face is put further off, so that what it returns is late, weak and near the normal | a ceiling 5 m up and a source 15 m away: -12 dB re the direct sound; 5 m away: -30 dB; a ceiling 10 m up and 15 m away: -24 dB (computed, reflection and spreading) | in proportion to the height added | none more |
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
of 20 m and more through it (estimated from the standard's table, not
verified by this lot). Ten metres of hedge is a decibel at 4 kHz. It is
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
(`render/early.py:300-306`) and the dry signal is read at that moving delay
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
| tail sites every 0.80 m | `scene-pack.md`, "tail" | 125 sites for the same; 16 s each before lot L24's tracer, a second or less after (`ray-tracer.md`) |

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

*Being verified against each source's own page; this section is completed
in the next commit of this lot.*

## 5. Audio: what has to be found

*Being verified against each dataset's own page; this section is completed
in the next commit of this lot.*

## 6. Validation: how one would know

*Completed in the next commit of this lot.*

## 7. A staged proposal

## 8. For the owner
