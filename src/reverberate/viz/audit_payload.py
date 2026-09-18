"""What the app's two audit views show, checked against what each solver read.

The audit is the one check between a scene and a solve that geometry and
materials are right, so nothing here is approximated or guessed:

* **Mirror.** The geometric engine reads the derived scene,
  ``mirror/scene.npz`` and ``mirror/scene.json``, through
  :func:`reverberate.mirror.geometry.load_derived`. The app draws the quad
  files that :func:`reverberate.mirror.audit.write_geometry_layers` wrote from
  that scene. :func:`mirror_audit` compares them array for array: the
  reflector and occluder triangles (to float32, the precision of the
  drawing), the label of every triangle, the facet of every reflector
  triangle, the facet kinds, the key. A difference refuses the layers and
  names the difference. The materials come from the scene's own arrays, and
  for each mirror field the run offers (B, C) the absorption and scattering
  the rays and the image sources were handed, computed by the engine's own
  :func:`~reverberate.mirror.calibrate.apply_parameters` and
  :func:`~reverberate.mirror.calibrate.image_scene` from the parameters of
  that field's report.
* **Wave.** The wave solver reads the voxel cache named by the payload's
  ``cache_key``; its materials are the export's per label absorption on
  PFFDTD's eleven bands, in the ``manifest.json`` beside the model the cache
  records. :func:`wave_materials` reads that table and checks the model's
  digest against the cache's record.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.settings import data_root

__all__ = ["PFFDTD_BANDS_HZ", "mirror_audit", "wave_materials"]

#: PFFDTD's eleven octave bands, 15.625 Hz to 16 kHz, on which the export
#: manifest gives each label's absorption.
PFFDTD_BANDS_HZ = [round(1000.0 * 2.0**k, 3) for k in range(-6, 5)]

#: The mirror fields a run may offer, by the suffix the engine gives their
#: files, and the letter the app shows.
MIRROR_VARIANTS = (("", "B"), ("_c", "C"))


def _npz_arrays(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Only the named members of an uncompressed ``.npz``."""
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in names:
            with archive.open(f"{name}.npy") as member:
                out[name] = np.lib.format.read_array(member)
    return out


def _quads_of(directory: Path, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A quad payload back as ``[n, 4, 3]`` corners, the index, and a label per quad."""
    corners = np.fromfile(directory / meta["corners_url"], dtype=np.float32)
    index = np.fromfile(directory / meta["index_url"], dtype=np.uint32)
    labels = np.fromfile(directory / meta["label_url"], dtype=np.int16)
    count = int(meta["quads"])
    if corners.size != count * 12 or labels.size != count * 4 or index.size != count * 6:
        raise ValueError(
            f"{meta['corners_url']}: sizes {corners.size}, {index.size}, {labels.size}"
            f" do not hold {count} quads"
        )
    per_quad = labels.reshape(count, 4)
    if np.any(per_quad != per_quad[:, :1]):
        raise ValueError(f"{meta['label_url']}: a quad carries two labels")
    return corners.reshape(count, 4, 3), index, per_quad[:, 0]


def _check_layer(
    name: str,
    directory: Path,
    meta: dict[str, Any],
    vertices: np.ndarray,
    labels: np.ndarray,
    problems: list[str],
) -> None:
    """The drawn triangles and labels of one layer against the scene's arrays."""
    try:
        corners, index, drawn_labels = _quads_of(directory, meta)
    except (OSError, ValueError, KeyError) as error:
        problems.append(f"{name}: {error}")
        return
    count = int(vertices.shape[0])
    if corners.shape[0] != count:
        problems.append(f"{name}: {corners.shape[0]} drawn, the scene holds {count}")
        return
    expected = (
        np.arange(count, dtype=np.uint32)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3])
    ).ravel()
    if not np.array_equal(index, expected.astype(np.uint32)):
        problems.append(f"{name}: the index does not draw each quad as its own triangle")
    if not np.array_equal(corners[:, :3, :], vertices.astype(np.float32)):
        moved = int(np.any(corners[:, :3, :] != vertices.astype(np.float32), axis=(1, 2)).sum())
        problems.append(f"{name}: {moved} triangles differ from the scene's")
    if not np.array_equal(corners[:, 3, :], corners[:, 2, :]):
        problems.append(f"{name}: a quad's fourth corner is not its third")
    if not np.array_equal(drawn_labels, labels.astype(np.int16)):
        wrong = int((drawn_labels != labels.astype(np.int16)).sum())
        problems.append(f"{name}: {wrong} triangles carry another label than the scene's")


def _variant(
    run_path: Path,
    source_id: str,
    suffix: str,
    scene_key: str,
    catalogue: Any,
    problems: list[str],
) -> dict[str, Any] | None:
    """The parameters a mirror field was made with and the materials they give."""
    from reverberate.mirror.calibrate import Parameters, apply_parameters, image_scene

    report_path = run_path / "mirror" / f"report_{source_id}{suffix}.json"
    card_path = run_path / "mirror" / f"card_{source_id}{suffix}.json"
    if not report_path.is_file():
        problems.append(f"{source_id}{suffix}: no {report_path.name}, parameters unknown")
        return None
    report = json.loads(report_path.read_text())
    record = report.get("parameters")
    if record is None:
        problems.append(f"{report_path.name}: no parameters")
        return None
    if card_path.is_file():
        card = json.loads(card_path.read_text())
        if card.get("parameters") != record:
            problems.append(f"{card_path.name} and {report_path.name} name other parameters")
    key = (report.get("scene") or {}).get("key")
    if key != scene_key:
        problems.append(f"{report_path.name}: scene {key}, the derived scene is {scene_key}")
    parameters = Parameters.from_record(record)
    rays = apply_parameters(catalogue, parameters).materials
    images = image_scene(catalogue, parameters).materials
    return {
        "source": source_id,
        "report": str(report_path.relative_to(run_path)),
        "parameters": record,
        "key": parameters.key,
        "rays": {
            "absorption": rays.absorption.round(6).tolist(),
            "scattering": rays.scattering.round(6).tolist(),
        },
        "images": {
            "absorption": images.absorption.round(6).tolist(),
            "scattering": images.scattering.round(6).tolist(),
        },
    }


def mirror_audit(
    run_path: Path, mirror: dict[str, Any], sources: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check the audit layers against the derived scene; the materials each field read.

    Returns ``problems``, empty when the layers are the scene's; the caller
    links the layers only then. What could not be traced about the materials
    is in ``material_problems`` and refuses nothing: the table says it.
    """
    from reverberate.mirror.geometry import DerivedScene, GeometryRules, MaterialTable

    problems: list[str] = []
    material_problems: list[str] = []
    scene_json = run_path / str(mirror.get("scene") or "mirror/scene.json")
    scene_npz = scene_json.with_suffix(".npz")
    audit_dir = run_path / str(mirror.get("audit") or "mirror/audit")
    record: dict[str, Any] = {
        "scene": str(scene_json.relative_to(run_path)),
        "problems": problems,
        "material_problems": material_problems,
    }
    if not scene_json.is_file() or not scene_npz.is_file():
        problems.append(f"{record['scene']}: the derived scene is not there")
        return record
    if not (audit_dir / "layers.json").is_file():
        problems.append(f"{audit_dir.relative_to(run_path)}: no layers.json")
        return record
    scene = json.loads(scene_json.read_text())
    layers = json.loads((audit_dir / "layers.json").read_text())
    key = str(scene["key"])
    record["key"] = key
    if mirror.get("key") not in (None, key):
        problems.append(f"walk.json names scene {mirror.get('key')}, the file is {key}")
    if layers.get("key") != key:
        problems.append(f"layers.json is of scene {layers.get('key')}, the scene is {key}")
    labels = [str(v) for v in scene["labels"]]
    if list(layers.get("labels") or []) != labels:
        problems.append("layers.json lists other labels than the scene")

    arrays = _npz_arrays(
        scene_npz,
        (
            "reflector_vertices",
            "reflector_facet",
            "occluder_vertices",
            "occluder_label",
            "facet_label",
            "facet_count",
            "facet_start",
            "facet_area",
            "facet_sides",
            "absorption",
            "scattering",
        ),
    )
    facet_label = arrays["facet_label"]
    kinds = [str(v) for v in scene["facet_kinds"]]
    reflector_labels = (
        facet_label[arrays["reflector_facet"]] if facet_label.size else np.zeros(0, dtype=np.int16)
    )
    _check_layer(
        "reflectors",
        audit_dir,
        layers["layers"]["reflectors"],
        arrays["reflector_vertices"],
        reflector_labels,
        problems,
    )
    _check_layer(
        "occluders",
        audit_dir,
        layers["layers"]["occluders"],
        arrays["occluder_vertices"],
        arrays["occluder_label"],
        problems,
    )
    facets = layers.get("facets") or []
    # The app gives each reflector triangle its facet from the facets' order
    # and counts: that order must be the scene's.
    expected_facet = np.repeat(np.arange(facet_label.size), arrays["facet_count"])
    starts: np.ndarray = np.concatenate([[0], np.cumsum(arrays["facet_count"])])[: facet_label.size]
    if (
        len(facets) != facet_label.size
        or not np.array_equal(arrays["reflector_facet"], expected_facet)
        or not np.array_equal(arrays["facet_start"], starts)
    ):
        problems.append("the reflector triangles are not in facet order")
    for i, facet in enumerate(facets[: facet_label.size]):
        if (
            facet["label"] != labels[int(facet_label[i])]
            or facet["kind"] != kinds[i]
            or int(facet["triangles"]) != int(arrays["facet_count"][i])
            or int(facet["sides"]) != int(arrays["facet_sides"][i])
        ):
            problems.append(f"facet {i}: layers.json does not describe the scene's facet")
            break

    materials = scene["materials"]
    sources_text = [str(materials["labels"][label]["source"]) for label in labels]
    catalogue_table = MaterialTable(
        tuple(labels),
        arrays["absorption"],
        arrays["scattering"],
        tuple(int(v) for v in materials["bands_hz"]),
        tuple(sources_text),
    )
    empty = np.zeros((0, 3, 3))
    catalogue = DerivedScene(
        labels=tuple(labels),
        materials=catalogue_table,
        facets=(),
        reflector_vertices=empty,
        reflector_facet=np.zeros(0, dtype=np.int32),
        occluder_vertices=empty,
        occluder_label=np.zeros(0, dtype=np.int16),
        occluder_sides=np.zeros(0, dtype=np.int8),
        rules=GeometryRules(**scene["rules"]),
    )
    variants: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_id = str(source.get("id"))
        for suffix, letter in MIRROR_VARIANTS:
            if not source.get(f"field_mirror{suffix}"):
                continue
            found = _variant(run_path, source_id, suffix, key, catalogue, material_problems)
            if found is not None:
                variants[f"{letter}:{source_id}"] = {"variant": letter, **found}

    census = scene.get("census") or {}
    record.update(
        {
            "summary": scene.get("summary"),
            "rules": scene.get("rules"),
            "labels": labels,
            "facets": [
                {**facet, "area_m2": round(float(arrays["facet_area"][i]), 4)}
                for i, facet in enumerate(facets[: facet_label.size])
            ],
            "census": census,
            "materials": {
                "bands_hz": [int(v) for v in materials["bands_hz"]],
                "absorption": arrays["absorption"].round(6).tolist(),
                "scattering": arrays["scattering"].round(6).tolist(),
                "source": sources_text,
            },
            "variants": variants,
        }
    )
    return record


_DIGESTS: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path) -> str:
    stat = path.stat()
    known = (str(path), stat.st_size, stat.st_mtime_ns)
    if known in _DIGESTS:
        return _DIGESTS[known]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    _DIGESTS[known] = digest.hexdigest()
    return _DIGESTS[known]


def wave_materials(cache_key: str | None, labels: list[str]) -> dict[str, Any]:
    """The wave solver's per label absorption for the voxel cache ``cache_key``.

    Read from the export manifest beside the model the cache was voxelised
    from; ``problems`` says what could not be traced or does not match.
    """
    problems: list[str] = []
    record: dict[str, Any] = {"bands_hz": PFFDTD_BANDS_HZ, "problems": problems}
    if not cache_key:
        problems.append("the payload names no voxel cache")
        return record
    cache_manifest = data_root() / "cache" / "vox" / cache_key / "manifest.json"
    if not cache_manifest.is_file():
        problems.append(f"voxel cache {cache_key} is not on this machine")
        return record
    cache = json.loads(cache_manifest.read_text())
    record["cache_key"] = cache_key
    record["h_m"] = cache.get("h_m")
    record["fmax_hz"] = cache.get("fmax")
    listed = list((cache.get("materials") or {}).keys())
    if listed != labels:
        problems.append("the voxel cache lists other materials than the payload")
    model = Path(str(cache.get("model_json", "")))
    export_manifest = model.parent / "manifest.json"
    if not model.is_file() or not export_manifest.is_file():
        problems.append(f"the model {model} the cache was made from is not on this machine")
        return record
    if cache.get("model_sha256") and _sha256(model) != cache["model_sha256"]:
        problems.append(f"{model.name} changed since the voxelisation")
    export = json.loads(export_manifest.read_text())
    table = export.get("materials") or {}
    missing = [label for label in labels if label not in table]
    if missing:
        problems.append(f"no absorption for {', '.join(missing)}")
    record["model"] = str(model)
    record["absorption"] = [
        [round(float(v), 6) for v in table[label]] if label in table else None for label in labels
    ]
    return record
