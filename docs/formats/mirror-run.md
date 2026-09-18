# The mirror's files in a run

What the geometric mirror (ADR 0014) writes beside a wave field, so a run
that holds a field can also hold its mirror, its audit and its judgement.
Paths are relative to the run directory; `<S>` is the source's id as in
`walk.json`. A second mirror beside the first (`--tag c`) writes the same
files with the suffix `_<tag>` after `<S>` or after the directory's name
(`paths_S1_c.npz`, `card_S1_c.json`, `field_mirror_c/S1.h5`,
`mirror/metrics_c/S1.json`, `report_S1_c.json`, `signature_S1_c.npy`), and
`walk.json` gains `field_mirror_<tag>` and `metrics_<tag>`: the app offers it
as the third field.

## Written by the card phase (where the GPU is)

| Path | What |
| --- | --- |
| `mirror/scene.npz`, `mirror/scene.json` | The derived geometry: facets, occluder triangles, material table; the JSON holds the rules, the census and the key (a digest of the arrays). |
| `mirror/audit/layers.json` | The census and the two layers' payload names for the app. |
| `mirror/audit/reflectors.*`, `mirror/audit/occluders.*` | Quad payloads in the grid viewer's own format (`f32` vertices, `u32` index, `i16` label). |
| `mirror/paths_<S>.npz` | The validated paths of every point, concatenated with `offsets`; one row is a path with its image, order, length, direction, gain per band, vertices and facet sequence. |
| `mirror/paths/<S>.json` | The same paths for the app: per point, `[order, time_ms, kinds, vertices]`. |
| `mirror/histogram_<S>.npz` | The rays' histogram: `energy [point, bin, band]`, `moments [point, bin, band, channel]`, `hits [point, bin]`, plus `bin_s`, `bands_hz`, `order`, `rays`. |
| `mirror/card_<S>.json` | The card phase's report: settings, parameters, scene key, tree, paths and rays counts, timings. |
| `mirror/lattice_<S>.npz` | Only where the field is not: `positions`, `sample_rate_hz`, `order`, so the card phase knows the lattice. |

## Written by the host phase (where the reference field is)

| Path | What |
| --- | --- |
| `field_mirror/<S>.h5` | The mirror field in the reference's format (`ambisonic-field.md`): same lattice datasets and attributes, `ir` rendered by the mirror, aligned to the reference's clock and scale; `mirror_silent` lists the points with no response; `provenance_json` holds the scene key, the settings, the alignment and the tree. |
| `mirror/metrics/<S>.json` | The criteria's record, the solver's floor, the summary over the storey (medians and percentiles of every error, verdict rates) and one judgement per point. The criteria's `focus_low_hz` (1000 from 2026-09-17) says which octave bands the decay, colour, seam and echogram errors read and where the reflections and the late field's directions are high passed; every band's values stay in the point's record. The interaural coherences are energy weighted means of 20 ms frames from that date. |
| `mirror/report_<S>.json` | The whole stage's report: the card's part, the alignment, the summary, the timings, the diffraction record (`asked`, `found`, grid, median detour, `edge_trees` and `reflected_paths` from 2026-09-18, when a shadowed point's corner gained its own image tree). |
| `mirror/signature_<S>.npy`, `.json` | The source signature: minimum phase FIR taps from the reference's median direct spectrum, and the record of how it was read. |
| `walk.json` | Updated: the source's entry gains `field_mirror` and `metrics`; a top level `mirror` entry names the scene, its key, the audit directory and the paths file per source. |

The summary of `mirror/metrics/<S>.json` reads one set of thresholds, the
difference a listener hears (`criteria.targets`). `pass_fraction` is the
share of points meeting each. A threshold does not move because one point's
reading is noisy: where the measurement's own floor is the limit, the
judgement fails and says so.

## Written by `mirror hybrid` (where both fields are)

| Path | What |
| --- | --- |
| `field_hybrid/<S>.h5` | One field from two solvers: the wave field under the cutoff, the mirror over it, in the reference's own format. Everything that is not `ir` is copied from the low side. `provenance_json` names both files, the crossover (cutoff, width in octaves, how long the onset is joined in pressure) and the seam's median and deciles: how far apart the two were over the band they share, before one scalar per point levelled them. |
| `walk.json` | Updated: the source's entry gains `field_hybrid`, which the app offers as a fourth field beside the reference and the two mirrors. |

## Written by the calibration (where the card and the references are)

| Path | What |
| --- | --- |
| `mirror/calibration/<key>.json` | The parameters (absorption scale per band, scattering scale, tail gain per band, and when set the image sources' own absorption scale per band and the shell's own scattering) and the trajectory of the search, one record per evaluation with its cost and the medians of the criteria. The key of a file without an image scale is what it was before that field existed. |
| `mirror/calibration/latest.json` | The key and the file of the last calibration, and the points it read. |
| `mirror/reference_subset_<S>.h5` | Only where the field is not: `ir [n, channels, samples]` of the chosen points at a low order, `point_index`, the rate and the order. |

A field rendered with a calibration carries its parameters in
`provenance_json` under `settings.parameters`; the catalogue's values are
all ones and zeros with the note "catalogue values, nothing calibrated".
