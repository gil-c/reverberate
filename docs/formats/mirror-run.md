# The mirror's files in a run

What the geometric mirror (ADR 0014) writes beside a wave field, so a run
that holds a field can also hold its mirror, its audit and its judgement.
Paths are relative to the run directory; `<S>` is the source's id as in
`walk.json`.

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
| `mirror/metrics/<S>.json` | The criteria's record, the solver's floor, the summary over the storey (medians and percentiles of every error, verdict rates) and one judgement per point. |
| `mirror/report_<S>.json` | The whole stage's report: the card's part, the alignment, the summary, the timings. |
| `walk.json` | Updated: the source's entry gains `field_mirror` and `metrics`; a top level `mirror` entry names the scene, its key, the audit directory and the paths file per source. |

## Written by the calibration (where the card and the references are)

| Path | What |
| --- | --- |
| `mirror/calibration/<key>.json` | The parameters (absorption scale per band, scattering scale, tail gain per band) and the trajectory of the search, one record per evaluation with its cost and the medians of the criteria. |
| `mirror/calibration/latest.json` | The key and the file of the last calibration, and the points it read. |
| `mirror/reference_subset_<S>.h5` | Only where the field is not: `ir [n, channels, samples]` of the chosen points at a low order, `point_index`, the rate and the order. |

A field rendered with a calibration carries its parameters in
`provenance_json` under `settings.parameters`; the catalogue's values are
all ones and zeros with the note "catalogue values, nothing calibrated".
