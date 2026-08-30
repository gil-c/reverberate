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

## Running the wave engine on a rented GPU

The wave engine below the crossover is [PFFDTD](https://github.com/bsxfun/pffdtd),
which needs a CUDA card. We rent one rather than owning one, which makes an
instance left running the main way this project can waste money.

Two pieces of tooling exist so that nobody has to be careful:

- `reverberate.gpu.vast` rents from Vast.ai. Every rental takes a deadline in
  hours and arms a detached watchdog before it returns, so the instance is
  destroyed even if the session that created it dies. Spend is appended to
  `<data root>/runs/gpu_spend.jsonl` and a rental that would pass the project
  ceiling is refused rather than warned about. The API key is read from
  `VASTAI_API_KEY` in the environment, loaded with
  `reverberate.auth.inject(["VASTAI_API_KEY"])`; it is never written to disk.

  ```
  python -m reverberate.gpu.vast search      # what is available and at what rate
  python -m reverberate.gpu.vast list        # live instances and spend to date
  python -m reverberate.gpu.vast destroy ID  # destroy, verify, and record the cost
  ```

  Renting is deliberately not a one-liner: state the offer, the rate, the hours
  and the total cost, get agreement, then call `reverberate.gpu.vast.rent()`.

- `scripts/build_pffdtd.sh` builds PFFDTD on the rented machine. PFFDTD was last
  touched in 2021 and does not compile or run against a current CUDA and numpy
  stack; the script pins the commit our published cost figures were measured on
  and applies the four fixes it needs. Run it on the instance, not locally.

Measured cost per room, and the grid resolution the numbers justify, are in
`data/runs/b1_pffdtd_cost/`.
