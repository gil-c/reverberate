"""Assemble every storey of HSSD once, and publish the lot.

Opening a flat used to assemble it first, at about a minute per distinct
template for the boolean union and flood fill of its collider. HSSD places
15 684 distinct templates, about 266 hours of one core, so it is paid once,
here, and the result goes to the object store. See ADR 0011.

``--warm i/n`` fills the per-template collider pool for one shard, on a rented
machine (``scripts/remote_assemble.py``). Without it every storey is assembled
from that pool and published: pools first, scenes second, catalogue last, so a
reader never finds a manifest before its meshes.

Run as ``python -m reverberate.viz.assemble_dataset <hssd_root> --workers N``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from pathlib import Path

from reverberate.geometry import collider_cache
from reverberate.geometry.hssd_assets import resolve_asset
from reverberate.geometry.scene_ids import local_name, scene_names
from reverberate.store import ObjectStore, shared_store
from reverberate.viz import scene_store
from reverberate.viz.scene_cache import ensure_scene, entry_for

__all__ = ["Assembled", "Storeys", "every_storey", "every_template", "main", "merge_catalogue"]

#: Times the pool is rebuilt after a worker dies, halving the workers each time.
WARM_ATTEMPTS = 12

#: The fewest workers a restart falls back to.
MIN_WARM_WORKERS = 4


@dataclass(frozen=True)
class Storeys:
    """One storey of one scene, under this project's name for it."""

    local: str
    scene_id: str
    storey_index: int


@dataclass
class Assembled:
    """What one storey produced, for the catalogue and the log."""

    local: str
    scene_id: str
    storey_index: int
    key: str = ""
    summary: str = ""
    storey: str = ""
    seconds: float = 0.0
    error: str = ""

    def line(self) -> str:
        if self.error:
            return f"{self.local} ({self.scene_id}) FAILED after {self.seconds:.0f}s: {self.error}"
        return f"{self.local} ({self.scene_id}) {self.seconds:.0f}s {self.key}: {self.summary}"


def every_storey() -> list[Storeys]:
    """Every storey the frozen name table knows, in name order."""
    rows = [
        Storeys(
            local=local_name(entry.scene_id, index + 1 if entry.storeys > 1 else None),
            scene_id=entry.scene_id,
            storey_index=index,
        )
        for entry in scene_names().values()
        for index in range(entry.storeys)
    ]
    return sorted(rows, key=lambda row: row.local)


def scene_files(hssd_root: Path) -> list[Path]:
    """Every scene description, skipping the AppleDouble ``._*`` files macOS tar adds.

    Unpacked on Linux they glob as scenes -- 336 where there are 168 -- and are
    163 bytes of resource fork that no JSON reader can take.
    """
    return sorted(
        path
        for path in (hssd_root / "scenes").glob("*.scene_instance.json")
        if not path.name.startswith("._")
    )


def every_template(hssd_root: Path, shard: int = 0, shards: int = 1) -> list[str]:
    """Every template the dataset places, or shard ``shard`` of ``shards``.

    Sharded by template, not by scene: the same wardrobe stands in dozens of
    flats, and two machines given half the scenes each would both carve it.
    """
    names: set[str] = set()
    for path in scene_files(hssd_root):
        record = json.loads(path.read_text(encoding="utf-8"))
        names.update(
            str(placed["template_name"])
            for placed in record.get("object_instances", []) or []
            if placed.get("template_name")
        )
    return [name for index, name in enumerate(sorted(names)) if index % shards == shard]


def _pending(templates: list[str], hssd_root: Path | None = None) -> list[str]:
    """The templates the pool lacks, leaving out those the dataset cannot resolve.

    An unresolvable template never gets an entry; counting it as pending kept a
    finished shard restarting on two door templates that do not exist.
    """
    objects = hssd_root / "objects" if hssd_root is not None else None
    pending = []
    for template in templates:
        mesh_file, record_file = collider_cache.entry_paths(template)
        if mesh_file.is_file() and record_file.is_file():
            continue
        if objects is not None and resolve_asset(objects, template) is None:
            continue
        pending.append(template)
    return pending


def _warm_one(arguments: tuple[str, str]) -> tuple[str, float, str]:
    from reverberate.geometry.sim_geometry import _load_and_union

    root, template = arguments
    start = time.time()
    try:
        status = "ok" if _load_and_union(Path(root), template) is not None else "unresolved"
    except Exception as error:  # noqa: BLE001 - one bad template must not stop the rest
        status = f"{type(error).__name__}: {error}"
    return template, time.time() - start, status


def _warm(arguments: argparse.Namespace) -> int:
    """Fill the collider pool for one shard.

    **A killed worker takes the pool with it**, and ``ProcessPoolExecutor``
    answers ``BrokenProcessPool`` for the whole run. The cause measured was a
    container memory cap that ``free`` does not show, so the pool restarts on
    what is left with half the workers. Entries on disk make a restart resume.
    """
    shard, _, count = arguments.warm.partition("/")
    templates = every_template(arguments.hssd_root, int(shard), int(count or 1))
    print(f"warming shard {arguments.warm}: {len(templates)} templates", flush=True)
    started = time.time()
    workers = max(arguments.workers, MIN_WARM_WORKERS)
    for attempt in range(1, WARM_ATTEMPTS + 1):
        remaining = _pending(templates, arguments.hssd_root)
        if not remaining:
            break
        done_before = len(templates) - len(remaining)
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                tasks = [(str(arguments.hssd_root), name) for name in remaining]
                futures = [pool.submit(_warm_one, task) for task in tasks]
                for done, future in enumerate(as_completed(futures), start=1):
                    name, seconds, status = future.result()
                    if done % 25 == 0 or status not in ("ok", "unresolved"):
                        total = done_before + done
                        rate = total / (time.time() - started) * 60
                        print(f"[{total}/{len(templates)}] {rate:.1f}/min {status} {name}")
        except BrokenProcessPool:
            workers = max(MIN_WARM_WORKERS, workers // 2)
            print(f"pool broke on attempt {attempt}; retrying with {workers} workers", flush=True)
    outstanding = _pending(templates, arguments.hssd_root)
    done = len(templates) - len(outstanding)
    print(f"warmed {done} of {len(templates)}, {len(outstanding)} outstanding", flush=True)
    return 0


def merge_catalogue(
    existing: list[dict[str, object]], fresh: list[dict[str, object]]
) -> list[dict[str, object]]:
    """The catalogue with ``fresh`` rows added, each replacing its namesake.

    A partial run adds to the published catalogue and never removes from it.
    """
    replaced = {str(row["local"]) for row in fresh}
    kept = [row for row in existing if str(row["local"]) not in replaced]
    return sorted(kept + fresh, key=lambda row: str(row["local"]))


def _assemble(arguments: tuple[str, Storeys]) -> Assembled:
    """One storey into the local cache; a failure is returned, never raised."""
    root, row = arguments
    result = Assembled(local=row.local, scene_id=row.scene_id, storey_index=row.storey_index)
    start = time.time()
    try:
        entry = ensure_scene(Path(root), row.scene_id, storey=row.storey_index)
        result.key, result.summary, result.storey = entry.key, entry.summary(), entry.storey()
    except Exception as error:  # noqa: BLE001 - one flat must not take the other 175 down
        result.error = f"{type(error).__name__}: {error}"
    result.seconds = time.time() - start
    return result


def _publish(store: ObjectStore, hssd_root: Path, good: list[Assembled]) -> None:
    """The two pools in one pass each, then each scene's own files."""
    render, sim = scene_store.pool_files(hssd_root, every_template(hssd_root))
    for files, prefix in ((render, scene_store.RENDER_PREFIX), (sim, scene_store.SIM_PREFIX)):
        sent = scene_store.publish_pool(store, files, prefix)
        print(f"{prefix}: {sent} sent, {len(files) - sent} already there", flush=True)

    def send(result: Assembled) -> None:
        entry = entry_for(hssd_root, result.scene_id, storey=result.storey_index)
        scene_store.publish_entry(store, entry, pools=False)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(send, good))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hssd_root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-publish", action="store_true", help="assemble locally only")
    parser.add_argument("--only", default="", help="comma separated local names")
    parser.add_argument("--report", type=Path, default=None, help="write the run's report here")
    parser.add_argument("--warm", default="", help="fill the collider pool for shard i/n and stop")
    arguments = parser.parse_args(argv)
    if arguments.warm:
        return _warm(arguments)

    store = None if arguments.no_publish else shared_store()
    if not arguments.no_publish and store is None:
        print("no store on this machine; run with --no-publish or provide credentials")
        return 2
    rows = every_storey()
    if arguments.only:
        wanted = {name.strip() for name in arguments.only.split(",") if name.strip()}
        rows = [row for row in rows if row.local in wanted]

    results: list[Assembled] = []
    with ProcessPoolExecutor(max_workers=max(1, arguments.workers)) as pool:
        futures = [pool.submit(_assemble, (str(arguments.hssd_root), row)) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
            print(f"[{len(results)}/{len(rows)}] {results[-1].line()}", flush=True)
    good = sorted((r for r in results if not r.error), key=lambda r: r.local)
    print(f"{len(good)} assembled, {len(results) - len(good)} failed", flush=True)

    if store is not None and good:
        _publish(store, arguments.hssd_root, good)
        fresh: list[dict[str, object]] = [
            {
                "local": r.local,
                "scene_id": r.scene_id,
                "storey_index": r.storey_index,
                "key": r.key,
                "summary": r.summary,
                "storey": r.storey,
            }
            for r in good
        ]
        catalogue = (
            merge_catalogue(scene_store.fetch_index(store), fresh) if arguments.only else fresh
        )
        scene_store.publish_index(store, catalogue)
        print(f"published {scene_store.INDEX} with {len(catalogue)} storeys", flush=True)

    if arguments.report is not None:
        arguments.report.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return 1 if len(good) < len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
