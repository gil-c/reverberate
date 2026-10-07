# The second generation of scene recipes

Date: 2026-10-07

Status: **the recipe side is written and tested; no engine was changed and
nothing was rented.** Written for lot L30 of ADR 0016, after the owner
listened to the first scene. The format is
[`scene-recipe.md`](../formats/scene-recipe.md), "Version 2"; the generator is
`reverberate.scenes.social`, the rules `reverberate.scenes.validate` (12 to
17), the levels `reverberate.scenes.levels`, the cost
`reverberate.scenes.cost`. Version 1 recipes are read, written and identified
as before, and the first generator is untouched. Every cost below is a
prediction from counts of the recipe and the constants of
`reverberate.trace.machines`; none was measured on a card. Section 6 lists
what the trace and the render must learn before a version 2 recipe sounds as
it is written.

## 1. The answer

The first scene asked for 1529 source positions under the crossover, 1774 of
its 1919 on rails: its price was people talking while they walked. A scene of
version 2 keeps people at stations and writes everything else that moves as
a quantity the low band does not read.

| scene | source positions | cells | pairs | 1 x RTX 3090, 0.173 USD/h | 8 x RTX 3090, 1.382 USD/h |
| --- | --- | --- | --- | --- | --- |
| the first scene, 20 min (its own plan, today's constants) | 1529 | 831 | 16 887 | 33.5 h, **5.79 USD** | 5.7 h, 7.92 USD |
| version 2, 3 min, `clarify_v1` | 3 to 14 | 1 to 119 | 3 to 559 | 0.19 to 0.41 h, **0.03 to 0.07 USD** | 0.23 to 0.37 USD |
| version 2, 3 min, every kind of sound | 12 to 33 | 1 | 12 to 33 | 0.06 to 0.14 USD | 0.33 to 0.40 USD |
| version 2, 20 min, `clarify_v1` | 4 to 15 | 1 to 145 | 4 to 952 | 0.21 to 0.59 h, **0.04 to 0.10 USD** | 0.24 to 0.46 USD |
| version 2, 20 min, every kind of sound | 48 to 73 | 1 to 145 | 48 to 1656 | 0.19 to 0.33 USD | 0.46 to 0.78 USD |

Each range is the quiet, medium and lively presets on hssd_0076, one seed
each (section 5). "Every kind of sound" is the same draw on a placeholder
library, which holds the footsteps, breaths, acknowledgements and laughter
that `clarify_v1` does not. Read with these cautions:

- **A short scene is priced by the machine's start, not by its solves.**
  Three solves are 3.5 minutes of a card and 0.01 USD; the rental's start and
  the grid's preparation are 400 s whatever the scene (`START_S`,
  `PREPARE_S`). On eight cards those 400 s are billed eight times, which is
  why the right machine for a scene of version 2 is one card, and why the
  campaign of a dwelling (`performance-audit.md`, section 6) matters more
  than before: twenty recipes on one rental share one start.
- **The brief quotes 1035 positions for the first scene.** Its recipe as
  stored (`data/runs/w45_clarify_scene/scene1/recipe.json`) counts 1529,
  which is also its plan's and ADR 0016's figure. At 1035 the solves alone
  are 20.3 card hours, 3.51 USD. The ratio to version 2 is of the same order
  either way: 15 to 500 times fewer solves.
- **Where the footsteps are in the low band, they are most of the
  positions**: 41 to 58 of 48 to 73 in twenty minutes with four walks.
  Question 3 below.
- The cells and pairs are the rule's first stage counted without the
  dwelling's surfaces (`scenes.cost`); on the first scene that count gives
  1529 positions and 831 cells, as its plan, and 15 731 pairs where the plan
  has 16 887.

## 2. What each decision became

Numbered as the owner's.

**1. Roles.** `near_voice` and `far_voice` are gone. A `voice` is a person;
its `roles` say by interval whether it is of the listener's `conversation`
or `outside` it, and in which `group` it talks. A group is placed as an
F-formation (Kendon 1990, *Conducting Interaction*): two to four people in
one room, 0.75 to 2.2 m from one another, each turned to the others; seats
keep their furniture's front and the head does the turning. Other groups
talk among themselves, with turns drawn from a stream of their own, at
twice the listener's reach to its own partners or further, which is 6 dB of
distance between a partner's voice and a stranger's. Membership changes
`membership_changes` times a scene (once in the three minute preset): a
stranger joins, a partner leaves, or the listener walks to another group.
During a walk the walker is of no group.

**2. The wearer's own voice** is a source of kind `own_voice`, carried at
the listener's mouth, `[0.09, 0, -0.05]` m from the centre of the head in
the head's frame, with the turns the listener takes and a stem of its own.
How it can be rendered is section 4.

**3. Turns.** A turn is whole utterances of the talker's clips, aimed at a
length drawn from a log-normal law of median 2.5 s. The next talker starts
a floor transfer offset after the last stops, drawn from a normal law of
mean 0.2 s and standard deviation 0.45 s, cut at -1 and +2 s: a gap two
times in three and an overlap one time in three. One turn in four is the
same talker going on. A lapse of 2 to 8 s follows 4 to 14 per cent of
turns, more in a calm scene. Acknowledgements fall inside somebody else's
turn, one about every 8 s of it, 3 dB under the talker's level; laughter
follows 6 per cent of turns or so, by several people at once. The figures
they are set from are in section 3. **With `clarify_v1` neither is
scheduled**: the library holds read sentences, and a sentence is not an
acknowledgement. What the generator would use: per speaker, short tokens
tagged `event: backchannel` ("mm", "yeah", "right", 0.2 to 0.7 s), laughs
tagged `laughter` (0.5 to 2 s), turns of spontaneous speech cut at their
own pauses, and the same speaker at several efforts (`effort`), a whisper
and a raised voice above all. The manifest takes the two tags today
(`clip-library.md`).

**4. Movement.** Three kinds, by what they cost.

- *The head* (free: applied at render). The listener's `gaze` is drawn in
  spells: attentive ones, in which the head follows whoever of the
  conversation holds the floor, 0.2 to 0.8 s late; and others, a glance at
  somebody else, reading (25 to 45 degrees down), looking away, or turning
  to a noise that has just started. The share of attentive spells is
  solved for, by bisection, so that the head is on the active talker
  `gaze_share` of the time somebody else talks: 0.6 asked, 0.58 to 0.59 on
  the three examples. The head goes 70 to 95 per cent of the way to its
  target, the eyes being taken to do the rest (Freedman 2008), at 80 to
  220 degrees a second, and on its target keeps moving by 1 to 4 degrees,
  an amount that swells and fades over 17 to 43 s. A keyframe about every
  second.
- *A talker's yaw* (free: the directivity is applied at render). A talker
  turns to who speaks, three times in four, and to who it speaks to; plus
  a `sway` of 2 to 8 degrees.
- *The body's sway* (`sway`, free under the crossover by the rule below):
  five sinusoids, two slow ones of 11 to 41 s and three of 2.5 to 8 s, whose
  amplitudes sum to 1 to 3 cm. Standing sway of the body is of that order
  and under 1 Hz (Winter 1995).
- *Real displacements* stay on stations and rails, and are rare.

**The rule that makes a sway free: the low band is read where the body is
without its sway; the mirror follows the body.** A sway is held to 5 cm
(rule 14). What that costs under 1 kHz, against
[`rail-interpolation.md`](rail-interpolation.md): a response read `d` away
from where the source is errs, for the sound that travels along the
displacement, by `|1 - exp(i k d)|^2`.

| held `d` | 125 Hz | 250 Hz | 500 Hz | 1 kHz |
| --- | --- | --- | --- | --- |
| 1 cm | -32.8 dB | -26.8 | -20.8 | -14.8 |
| 3 cm | -23.3 dB | -17.3 | -11.2 | -5.3 |
| 5 cm | -18.8 dB | -12.8 | -6.9 | -1.1 |

**By the project's own yardstick this is worse than a rail.** Two
positions 6.5 cm apart read linearly hold -27.6 dB at 500 Hz and -23.3 dB
at 630 Hz (`rail-interpolation.md`), where a hold 3 cm off reads -11 dB:
the hold is the nearest position, not an interpolation. What the figure
measures here is a response that is exact for a source 3 cm away: there is
no comb, since one position is read and not two. What is heard is a talker
whose low band stands still while the band above it sways: the two bands
disagree by a delay of `d / c`, 87 microseconds at 3 cm, which at the
crossover is 31 degrees of phase and 0.33 dB of level (0.9 dB at 5 cm),
and by a direction of 2 degrees at 0.75 m. The mixture and the stems are
made by the same rule, so the label is consistent. It is unheard, and it
is question 1.

**5. Levels.** A talker's own level is 60 dB at 1 m, the normal effort and
the library's convention, give or take a normal draw of 2.5 dB cut at 5:
people differ, roles do not. It falls by up to 6 dB in a calm scene (the
relaxed effort from a calmness of 1.0), and one turn in seven is whispered
where the calmness is 0.85 or more and nobody else talks. It rises with
the noise at the talker by `lombard_level_db`. The interval's `gain_db`
carries the level and `effort` names the clip wanted; with `clarify_v1`
every clip is a normal voice, so a raised voice is today a normal voice
made louder, which is exactly the gain the owner did not want: the remedy
is the library's (decision 3). A noise has its class's level at 1 m
(section 3). **The criterion enforced** is rule 16: every turn of the
listener's conversation is at least `snr_floor_db` over the noise at the
listener in free field, 0 dB by default. The generator turns the loudest
masker of the worst turn down within its class, then off, and draws the
scene again where voices alone mask it. On the examples the least turn is
at +1.5 to +5.4 dB and the median at +5.5 to +8 dB.

**6. Fixed noises.** Every noise but a person's is one `dwell` for the
whole scene (rule 12), at a `fixture`: a station in the air by the object
it belongs to. `load_hssd_floor` now reads the objects the export labels:
on hssd_0076, 2 televisions, 1 washing machine and dryer, 2 counters, 2
showers, 3 toilets, 1 piano, tables, desks, shelves, nightstands, and, by
their meshes, 5 windows and 2 outer doors (an opening carries no category;
one that does not reach the floor is taken for a window). A television
faces the way its panel faces and is directional; an appliance stands 15 cm
above its machine. **What is missing**: no refrigerator, oven, hood, sink or
dishwasher is labelled, so a kitchen's noise goes to the counter; the
kitchen is not a room (ADR 0010 merges it with the living room); nothing
says whether a window opens; and an emitting point is not checked against
the voxel grid. A television or a radio is a `media_voice`.

**7. A person's noises.** A `noise` of `body` is carried at its person's
mouth, with a pattern turned any way (`yaw_offset_deg`, free); one of
`steps` is carried at the floor and sounds at each footfall of a walk.
Neither exists in `clarify_v1`, and the generator leaves them out and says
so. The directivity drawn is today the one pattern the assets hold,
`voice_v1`, turned at random: a family of patterns in the pack would be as
free.

**8. Outside.** A `noise` of `outside` stands at a window's or an outer
door's fixture and names it with its `state`, `open` or `closed`. Its level
is what comes in, a street's at the wall less what the opening takes
(section 3). How an opening radiates is another lot's.

**9. Duration** is `duration_s`. `SocialParameters.preset(name)` is three
minutes; `python -m reverberate.scenes generate --dwelling hssd_0076 --seed N
--preset medium --out recipe.json [--duration S]`.

## 3. The figures used, and where they are from

Confirmed for this note: ISO 9921's ladder. The others are quoted from
memory of the papers, with no network on most of this lot's work; **they
are to be checked against the papers before anything is published from
them**, and each is one constant of `SocialParameters` or of
`scenes.levels`.

| what | value used | source |
| --- | --- | --- |
| vocal effort at 1 m | relaxed 54, normal 60, raised 66, loud 72 dB(A) | ISO 9921:2003, annex A |
| a whisper | 45 dB at 1 m | not ISO's; an assumption, 15 dB under normal |
| Lombard rise | 0.5 dB a dB of noise above 45 dB at the talker, to the loud effort | Lazarus 1986, *Applied Acoustics* 19 (0.3 to 0.6 dB a dB); ISO 9921 |
| floor transfer offset | mean 0.2 s, s.d. 0.45 s; a third overlaps | Heldner and Edlund 2010, *J. Phonetics* 38 (about 40 per cent overlaps, the mode a short gap); Levinson and Torreira 2015, *Frontiers in Psychology* 6 (mode near 200 ms); Stivers et al. 2009, *PNAS* 106 |
| gaze of a listener at a talker | 0.6 of the time | the owner's rule; the literature reads higher for two people (Kendon 1967; Argyle and Cook 1976) |
| head against eyes | the head goes 70 to 95 per cent of a large shift | Freedman 2008, *Exp. Brain Res.* 190 |
| speech to noise at home | floor 0 dB; scenes sit at +5 to +8 dB in the median | Pearsons, Bennett and Fidell 1977 (EPA-600/1-77-025); Smeds, Wolters and Rung 2015, *JAAA* 26; sentences are half understood by normal hearing near -5 to -8 dB |
| television, at 1 m | 58 to 70 dB | a programme listened to at 55 to 65 dB at 3 m; an assumption |
| radio, music | 55 to 66, 50 to 68 dB | assumptions of the same kind |
| appliance, at 1 m | 42 to 68 dB | the sound power an EU energy label declares (EN 60704), less 8 dB for a hemisphere at 1 m: a washing machine 47 to 55 dB(A) washing and 70 to 77 spinning, a hood 52 to 70 |
| water, cooking | 50 to 66, 48 to 64 dB | assumptions; no label exists |
| outside, open window | 43 to 60 dB | a street at the wall, 55 to 72 dB, less 12 dB (WHO, *Night Noise Guidelines* 2009, takes 10 to 15 dB for a window ajar) |
| outside, closed | 27 to 44 dB | the same less 28 dB, ordinary double glazing |
| breath and clothes, steps | 25 to 40, 40 to 55 dB | assumptions |

## 4. The wearer's own voice: what the engines can do

The mouth is 0.10 m from where the field is taken.

- **The wave solver cannot answer it.** The array that hears a cell has a
  radius of 0.26 m (`trace.plan.ARRAY_RADIUS_M`): the source is inside it,
  and an expansion about the head holds only inside the ball that reaches
  the nearest source, 0.10 m. And the source moves with the listener: a
  solve a cell, 831 more on the first scene.
- **The order 7 signal cannot carry its direct sound** to the ears: they
  are 0.09 m from the centre, at the edge of that ball. Nor is the direct
  sound a room's affair: at the device it is the path round the head, and
  the bone, neither of which a room simulation without a head has.
- **The mirror can give the room's answer to it**: image sources and the
  tail from a source at the mouth, turned with the head, without the direct
  path. That is the engine's ordinary work at every step, and no solve.

**Recommended, the cheapest that is sound**: two things for one source. The
direct sound is not rendered into the field: the stem keeps the dry clip at
its level and the device stage applies a mouth to microphone response in
the head's frame, which no rotation and no room changes. The room's answer
is the mirror's alone, over the whole band, the direct path left out. Its
low band then has no modes; it stands 15 dB or more under the direct sound
at the wearer's ears, the reflections coming from metres and the mouth from
centimetres. **Better, at one solve a rest**: where the listener rests, a
solve from the mouth with the free field of the source taken out of the
array's records, which leaves an incoming field an expansion can hold; a
3 minute scene has one to three rests. It asks the solver for a source
inside an array and a subtraction; it was not tried. The recipe is the same
for both. Question 2.

## 5. The three examples

On hssd_0076 with `clarify_v1` and the assets of the first scene, three
minutes each; `python -m reverberate.scenes describe FILE --cost` prints
what is below. The files are in the lot's scratch area, not in the
repository.

| | quiet, seed 2 | medium, seed 5 | lively, seed 2 |
| --- | --- | --- | --- |
| `recipe_sha256` | `b1891312...` | `355c68f2...` | `aff157fa...` |
| voices, the listener apart | 2 | 5 | 9 |
| the listener's conversation | 1 partner, a second joins | walks from a group of 2 to a group of 3, 17.7 m, between 40 and 62 s | 3 partners, a fourth joins |
| fixed sources, dB SPL at 1 m | outside by a closed window, 31 | an appliance 54, music 59 | an appliance 62, music 68, water 65, outside by a closed window 44 |
| efforts of the turns | 29 normal, 7 whispered | 77 normal, 1 relaxed | 105 raised, 11 loud, 3 normal |
| speech over noise, least and median | +28.3, +44.2 dB | +1.5, +5.5 dB | +5.4, +8.0 dB |
| gaze on the talker | 0.59 | 0.58 | 0.58 |
| source positions, cells, pairs | 3, 1, 3 | 7, 119, 559 | 14, 1, 14 |
| 1 x RTX 3090 | 0.19 h, 0.03 USD | 0.36 h, 0.06 USD | 0.41 h, 0.07 USD |
| left out for want of clips | acknowledgements, laughter, body, steps | the same | the same, and a radio |

The recipes are 44 to 101 kB: a dwell a turn of a talker's head and a
keyframe a second.

## 6. What the engine must learn

Nothing under `render/`, `mirror/`, `wave/` or `viz/` was touched. A trace
of today reads `source.segments`, `source_state(...).position` and
`listener_state(...).position`, and so fails on a carried source and
ignores every sway. In order of need:

1. **`trace.plan._slots`, `tail_sites_of`, `tracks_of`**: a source with
   `attach` has no segments. Its low band positions and its tail sites are
   its carrier's (`kinematics.low_band_source_positions` already says
   which); at the floor, its footfalls; for `own_voice`, none.
2. **The mirror's source and head are `mouth` and `head`**, not `position`:
   `SourceState.mouth`, `ListenerState.head`. The low band, the cells
   (`listening_cells`, which reads a head that moves by 1e-9 m as a walk)
   and the tail keep reading `position`. Until then a version 2 scene is
   rendered with nobody swaying, which is valid and stiller than written.
   With it, no two audible steps share a position of the early trace:
   `step_pairs` becomes every audible step, 1.9 ms each on the host.
3. **`own_voice`**: section 4. The trace needs a rule for a source 0.10 m
   from the head (rule 8's clearance does not apply to it), the render a
   stem that is not in the field.
4. **A fixture's source** is by an object, up to 0.25 m from a surface and
   possibly outside the free floor: the solver's placement of a source near
   a boundary, and a refusal where the point is in a solid.
5. **A footstep** is 5 cm above the floor and hops: at the instant it hops
   the response changes under a sound that still rings. The quasi-static
   render lays the tail of the last footfall on the new place's response.
6. **`kind` and `subtype`** have new values (`voice`, `own_voice`,
   `media_voice`; `body`, `steps`, `outside`, `radio`) wherever the pack,
   the stems and the page name them; `roles` are to be carried to the
   training label. `opening.state` is for the lot that renders openings.
7. **The page** (`viz/scene_api.py`, `scene.js`) reads the first
   generator's `Parameters` and draws segments: it does not show a carried
   source, a role or the gaze.
8. `level_spl_1m_db` asks nothing: `gain_db` is what the engine applies, as
   before.

## 7. Open questions for the owner

1. **Is a sway heard as it should be when the low band holds still?**
   Recommended: yes, hold, at up to 3 cm, and listen to one scene traced
   both ways once the mirror follows the mouth. The alternative that meets
   the project's -20 dB is a second solved position 8 cm from each station
   and a linear reading between the two: one more solve a station occupied,
   which doubles a scene's solves and leaves it under 0.2 USD.
2. **The own voice: mirror alone, or a solve a rest?** Recommended: the
   mirror alone first, the direct sound left to the device stage; a solve a
   rest if the room's answer to one's own voice is heard to lack its low
   end.
3. **Footsteps under the crossover?** They are most of a full scene's
   positions. Recommended: keep them in the recipe as they are, and let the
   trace render them with the mirror alone at first, as the own voice; a
   footfall's thud is under 300 Hz, so listen before deciding.
4. **The floor of speech to noise: 0 dB in free field?** A room adds to
   both, and a wall takes from the noise alone, so the true ratio is
   seldom lower. Recommended: 0 dB, and a distribution over scenes rather
   than one value: most scenes at +5 to +15 dB, a tenth near the floor.
5. **A walk is silent but for footsteps.** A turn ends 1.3 s before its
   talker leaves. Talking while walking is what the first scene paid for:
   12.5 solves a metre. Recommended: keep it out by default, and add it as
   a share when the rails are read at 16 cm with eight positions
   (`rail-interpolation.md`), which halves it.
6. **Turns are dealt evenly** among those present, and the wearer speaks as
   often as anybody. A real conversation has a talkative member. Say if the
   wearer's share should be a parameter.
7. **The listener walks rails**, between stations, as sources do: its
   cells are then places other recipes of the dwelling walk too. It no
   longer wanders the free floor. Say if that loses something wanted.
8. **The library** decides how much of decisions 3, 5 and 7 is heard:
   efforts, acknowledgements, laughter, a person's noises, footsteps, a
   radio. Lot L31 audits it; the tags the generator reads are in
   `clip-library.md`.
9. **The published figures** of section 3 marked as from memory are to be
   checked, the floor transfer offset's law above all.
