# 0006: the solver run as the acoustic mode of the apartment viewer

Status: accepted. Supersedes an earlier draft of this record, which decided the
opposite and was wrong; the reasoning is kept below under "what was rejected"
because the mistake is instructive.

## Context

The project could not see or hear what the wave solver produced.

`reverberate.viz` already served every HSSD apartment in a browser with three
modes: colour, label and acoustic, behind a selector over 168 scenes. So the gap
was never "nothing has been seen". The acoustic mode colours **authored HSSD
instances** by a `pyroomacoustics` absorption palette, computed through
`geometry.absorption.compensate`. That describes a ray solver, and the engine is
a finite difference wave solver whose input is a serialised surface list,
`model_json`, that the apartment browser never reads. Looking at the acoustic
mode therefore told you nothing about any wave run: it is a picture of a
different simulation of the same flat.

Roadmap constraint 9 asks that the solver and the viewer read the same
serialised surface list, so that a picture and a response can be shown to be of
the same object.

## Decision

**The solver run replaces the acoustic mode of the apartment viewer**, leaving
three modes: colour, label and acoustic. It is reached from the same selector
and drawn with the same camera into the same canvas. A rendered run names the scene and room it was
simulated in, so it belongs to an apartment; it is another way of looking at
that apartment rather than a separate application.

The mode renders `model_json` group by group, in that file's own points and
triangle indices, and nothing else. It draws the source and receiver markers at
the placement the run recorded, and plays the dry voice and the twelve wet
convolutions published with the run.

**The split is data against presentation, not run against apartment.**
`reverberate.viz.run_view` reduces a run to one JSON payload plus its audio and
knows nothing about the page. `web/solver_mode.js` draws that payload and owns
the panel. `serve_room` discovers runs, builds their payloads at startup and
tells each apartment which runs it has, so a run is discoverable from the
selector rather than only by already knowing it exists.

**All analysis happens in Python at build time** and is embedded as JSON: the
waveform envelope, the Schroeder decay curve, the spectrogram, the per band
measures. The browser draws and plays; it performs no signal processing.
Duplicating the analysis in JavaScript would put it where none of the tests can
reach it.

**The ray-era acoustic mode is deleted rather than kept beside its
replacement.** An earlier revision kept it, on the grounds that it was the only
view of the `pyroomacoustics` palette. That left two modes describing two
different simulations of the same flat, one of them of an engine the project no
longer runs, and the viewer's own naming could not tell you which was which. Its
producer in `viz/scene_manifest.py` is untouched: `collider_url` is the geometry
the simulator export uses and has consumers beyond the viewer, so only the
consumer went.

**Surfaces are coloured by their measured absorption at 1 kHz**, on a ramp from
red for reflective to blue for absorbent, rather than by the colour in
`model_json`. `experiments.scene_export` writes `[128, 128, 128]` into every
group because PFFDTD ignores the field, so passing it through rendered all
thirteen materials of the bedroom as one flat grey. The coefficients come from
the `manifest.json` the export writes beside the model, which is the only place
labels and coefficients are tied together. Measured on the bedroom: thirteen
groups, ten distinct colours, the collisions being genuine equalities such as
basket, decoration, lamp and mirror all at 0.03. The ramp is stopped often
enough that no leg of it passes through grey, so grey keeps its one meaning of
"no coefficient found", and each legend entry carries its number because colour
alone cannot separate materials that share an absorption.

**Each simulated room leads the selector and is what the viewer opens on.** It
is a way into the apartment it belongs to, not a second scene, so all three
modes and the walk behave there as they do everywhere else. Its apartment
assembly is started but not awaited, so the run is on screen while the rest of
the flat is still being built.

## What was rejected

**A separate application, which is what was built first.** The argument was that
the two viewers answer different questions: the browser answers "what does this
HSSD apartment look like" over 168 scenes assembled on demand, and a run viewer
answers "what did the solver receive and what came back" for one run. That is
true, and it is not a reason to build twice. A run *is* an apartment, seen
differently. Splitting them duplicated the camera, the canvas, the surface
legend and the caveat rendering, and it made the run reachable only by someone
who already knew the run directory's path. The separate page also invented a
grid layout whose row height was unbounded, which no other page in the project
needed, and which grew the document to 26 million pixels on a 2.5x display.

**A fixed overlay panel rather than a layout column.** Kept, and worth stating
as a decision: the panel is taller than the screen, so putting it in the layout
lets its content determine the page height, and the renderer is then sized from
a box the panel can grow. As an overlay it cannot do that, and the renderer
takes the window, which nothing drawn into it can enlarge.

**Streaming the responses to the browser and analysing them there.** A 72 kHz
response is about 108,000 samples per receiver, and twelve of them would be
roughly 10 MB of page for plots a few hundred pixels wide. The envelope keeps
both extremes of each bucket rather than sampling on a stride, because sampling
would alias the waveform into whatever the stride happened to hit and draw a
response quieter than it is.

## Consequences

One outline fences the walk in every mode, and switching mode does not move the
camera. An earlier revision gave each mode its own walkable region, on the
stated belief that the solver mesh and the apartment were in different frames.
That belief was wrong, and measuring it settled the matter: the exported
apartment spans (-21.07, -0.12, -13.11) to (2.51, 2.86, 5.33) and the bedroom
the solver read spans (-2.46, -0.02, -2.41) to (1.92, 2.80, 0.74), which is
inside it. The mesh is the apartment's own bedroom, cropped and not recentred.
Three representations of one room in one frame means a mode switch is a change
of representation and not a change of place, so the viewer keeps whatever they
had walked over to look at. The panel still offers "stand at this receiver" as
an explicit move, so what is heard and what is seen can be made to come from the
same point on request.

Until an apartment arrives there is no outline, and the run's bounding box
stands in for it, which is what makes the run walkable during the assembly.

Switching apartments points the run at the **incoming** apartment before
awaiting its manifest. Assembly takes minutes on a first visit, and a run
repointed afterwards leaves the button offering the outgoing apartment's run for
the whole wait, ready to open it over a room it was never simulated in. Doing it
beforehand also makes the run available immediately, which is correct: a run
payload is published by the solver and owes nothing to the HSSD assembly.

The server sends `Cache-Control: no-store`. The site is rebuilt from the source
tree on every start, and a browser holding a previous ES module reports the
behaviour of an older revision, which looks exactly like an edit having had no
effect.

The panel carries the caveats the report carries, and the tests assert that it
does: the responses are not binaural, air absorption is absent, the two theory
bounds are shown as bounds rather than as one prediction, and the bands above
the simulated `fmax` are marked so that a decay time measured in the low pass
skirt cannot be read as a property of the room.

Run it with:

```
python -m reverberate.viz.serve_room <hssd root> --runs <runs directory>
```

Simulated rooms lead the selector, and the viewer opens on the first of them.
