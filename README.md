# reverberate

## The goal

`reverberate` is a hybrid room acoustics simulation pipeline. It turns real
furnished apartment geometry into physically credible, spatially resolved room
impulse responses, at a cost that makes tens of thousands of them affordable.

Two engines share one scene description: a wave solver below the crossover
frequency, where diffraction, room modes and partially transparent objects
matter, and a geometric engine above it, where the wavelength is smaller than
any real object and the geometric approximation is honest.

The thesis being defended: absorption belongs on the surfaces where it
physically sits, not redistributed into a global reverberation time. A room
average misses the spatial cues that matter, and every simplification here is
judged against that.

## What is covered

- A scene interchange format, derived from real furnished 3D scenes, with
  automatic assignment of acoustic materials, and read identically by both
  engines and the viewer.
- The hybrid simulation itself, emitting full impulse responses rather than
  decay envelopes. Decay curves and scalars are derived from them as metrics.
- A viewer that shows exactly what the simulators were handed, and never
  simulates anything itself.
- The generated dataset, and a validation report stating how far the result can
  be trusted.

## What is not covered

Predicting acoustics from geometry with a learned surrogate, its baselines and
any side by side demo are out of scope. Simulating the room correctly is a hard
enough problem on its own. Prediction becomes a downstream project that consumes
this one's dataset.

## How it works

A room is represented by its 3D mesh and by the acoustic properties of each of
its surfaces, walls, floor and furniture. Rooms are extruded from their authored
floor polygons so they are watertight by construction, furniture is reduced to
fitted envelopes rather than convex decompositions, and each surface carries an
absorption curve per octave band together with a transmission coefficient.

That single scene description is handed to both solvers. Each covers the
frequency range where its approximation holds, and the two are recombined across
the crossover band into one impulse response per source and listener pair.
