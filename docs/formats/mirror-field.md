# The mirror solver's files

Written by `python -m reverberate.mirror` in a run directory beside the wave
field `field/<S>.h5` (`ambisonic-field.md`).

## `run`

| Path | What |
| --- | --- |
| `field_mirror/<S>.h5` | The wave field's lattice and per point datasets, `/ir` from the mirror on the wave field's clock and scale. `/mirror_silent` lists the points left silent; `provenance_json` holds the scene key, the settings and the applied gain. |
| `mirror/scene.npz`, `scene.json`, `scene.model` | The derived geometry: reflector facets and triangles, occluder triangles, labels, materials, rules, census, key; and the digest of the model files it was derived from. |
| `mirror/paths_<S>.npz` | Every point's validated paths, concatenated: `offsets` (points + 1), `receivers` (points, 3), and per path `image`, `order`, `length_m`, `direction` (3), `gain` (bands), `points` (vertices, 3), `sequence` (facets). |
| `mirror/signature_<S>.npy` | The source signature: minimum phase FIR taps of the wave field's median direct spectrum. |
| `mirror/report_<S>.json` | Devices, settings, timings, trace and render records, alignment. |

## `calibrate`

`mirror/calibration/<key>.json`: `parameters` (the record of
`reverberate.mirror.parameters.Parameters`, including `rendered_with`, the ray
and render settings it was fitted under), `key`, the calibration `points`, and
the `trajectory` of evaluations. It reads `mirror/scene`, `mirror/paths_<S>.npz`
and the wave field, or `mirror/reference_subset_<S>.h5` (`ir`, `point_index`,
`sample_rate_hz`, `order`) where the field is not.

## `judge`

`mirror/judged_<field>_<S>.json`: every n-th point of `<field>/<S>.h5` judged
against the wave field; the medians and pass shares of the sixteen criteria and
`criteria_met`.

## `hybrid`

`field_hybrid/<S>.h5`: the wave field's lattice, `/ir` the wave field under
the crossover and the mirror over it; `provenance_json` holds the crossover and
the seam's median and deciles in dB before the mirror was levelled.
