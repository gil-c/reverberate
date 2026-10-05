"""One queue for the whole trace: jobs that wait on jobs, pulled by a process a card and a core.

A stage of the trace is a set of jobs that do not need each other: a launch
of the wave solver, a block of a source's steps to trace, a tail site, a
block of pairs to level, a block of the pack's rows. A job says where it
runs (:data:`CARD`, :data:`HOST`, or in the process that holds the queue),
what it waits for, and what it needs of a card's memory. :class:`Pool` hands
each idle worker the first job it may take, so that no card and no core
waits while a job it could run is ready, whatever stage that job is of.

**Processes, not threads.** The stages a card does not bound are bound by
the interpreter, and a thread more of one interpreter buys little (six
threads levelled 2.2 times as fast as one). A worker is a process of its
own, started fresh (``spawn``: a process forked after CUDA is up is not
usable), that sees one card or none: ``CUDA_VISIBLE_DEVICES`` is set for it
before it starts, so a worker never opens a context on another worker's
card, and its transforms are told to use the threads it was given. A worker
builds what it reads from the files of the run (:func:`serve` calls
``make``), which the page cache shares between them: nothing of the scene
is pickled, and a job's message is its name and a few numbers.

**A result is a file under the job's name**, written whole or not at all,
and the stage's merge reads them in the jobs' order: which worker made a
result, and when, cannot change what is merged. That is what the pack's
identity across machines rests on (``tests/test_trace_pool.py``).

**A job that fails is tried elsewhere.** A worker that is refused its
memory hands the job back; the pool parts it where its stage knows how
(``split``), or gives it to another worker. A worker that raises, or dies,
loses the job to another, :data:`MAX_TRIES` times at most; then the run
stops with the worker's own traceback. A worker may also answer with the
job in parts (``parted``): the parts take its place and what waited for it
waits for them.

With ``inline`` there are no processes: the jobs run one after the other
in the caller's process, in the order workers of those kinds would have
taken them. The tests and a run on one core use it.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any

__all__ = [
    "CARD",
    "HOST",
    "MAX_TRIES",
    "PARENT",
    "THREAD",
    "Job",
    "Pool",
    "PoolError",
    "WorkerSpec",
    "serve",
]

#: Where a job runs: on a worker that holds a card (any worker on a machine without one),
#: on any worker with ``numpy``, in the process that holds the queue between two
#: hand-outs (a merge: it must be short), or in a thread of that process (a stage that
#: is one long call of its own, which the queue does not wait on).
CARD = "card"
HOST = "host"
PARENT = "parent"
THREAD = "thread"

#: A job is tried this many times, each on another worker where there is one.
MAX_TRIES = 3

#: The variables that bound the threads of a worker's transforms and products.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class PoolError(RuntimeError):
    """A job that no worker could finish, or jobs that nothing can start."""


@dataclass(frozen=True)
class WorkerSpec:
    """One worker: its number, the card it holds or ``None``, the threads it may use."""

    index: int
    card: int | None = None
    threads: int = 1
    #: What its card has free, bytes, as the machine was read; 0 when not known.
    free_bytes: float = 0.0


@dataclass
class Job:
    """One piece of a stage. ``stage`` and ``name`` together are its key and name its result."""

    stage: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    on: str = HOST
    #: Keys (``stage/name``) of the jobs whose results this one reads.
    after: tuple[str, ...] = ()
    #: Card memory it needs, bytes; 0 when it takes what it finds.
    bytes: float = 0.0
    #: Lower is handed out first among the jobs a worker may take.
    priority: int = 0
    #: Jobs that read the same things: a worker is handed, of equal priority, one of the
    #: group it last worked on first, and so keeps what it built for the one before.
    group: str = ""
    #: :data:`PARENT` and :data:`THREAD`: what to call. Never sent to a worker.
    work: Callable[[], Any] | None = None
    #: A job refused its memory, as smaller jobs; ``None`` when it cannot be made smaller.
    split: Callable[[Job], list[Job] | None] | None = None
    tries: int = 0
    avoid: set[int] = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.stage}/{self.name}"


def serve(
    spec: WorkerSpec, make: Callable[[WorkerSpec], Any], link: Connection, faults: dict[str, Any]
) -> None:
    """A worker's life: build what it reads, then a job at a time until told to stop."""
    try:
        handler = make(spec)
    except BaseException:  # noqa: BLE001 - the queue's process reports it
        link.send(("broken", traceback.format_exc()))
        return
    link.send(("ready", _free_of(handler, spec), getattr(handler, "measured", None)))
    while True:
        try:
            message = link.recv()
        except EOFError:
            return
        if message is None:
            return
        stage, name, payload, tries = message
        started = time.time()
        try:
            _fault(faults, f"{stage}/{name}", tries)
            record = handler.run_job(stage, name, payload)
            link.send(("done", record, time.time() - started, _free_of(handler, spec)))
        except MemoryError:
            # With where it was refused: the last line is the error, the lines before its place.
            link.send(("memory", traceback.format_exc()[-3000:], time.time() - started))
        except Exception:  # noqa: BLE001 - sent to the queue's process, which decides
            link.send(("failed", traceback.format_exc(), time.time() - started))


def _free_of(handler: Any, spec: WorkerSpec) -> float:
    free = getattr(handler, "free_bytes", None)
    return float(free()) if callable(free) else float(spec.free_bytes)


def _fault(faults: dict[str, Any], key: str, tries: int) -> None:
    """A failure a test asked for: ``{key: (kind, times)}``, raised on the first tries."""
    kind, times = faults.get(key, ("", 0))
    if tries < int(times):
        if kind == "memory":
            raise MemoryError(f"a test refused {key} its memory")
        raise RuntimeError(f"a test failed {key}")


@dataclass
class _Worker:
    spec: WorkerSpec
    process: Any = None
    link: Any = None
    ready: bool = False
    job: Job | None = None
    started: float = 0.0
    free_bytes: float = 0.0
    busy_s: float = 0.0
    jobs: int = 0
    #: What the worker said of the device it holds when it was ready.
    said: Any = None
    #: The group of the job it took last.
    group: str = ""


class Pool:
    """The queue and its workers. ``add`` jobs, then ``run`` until every one is done."""

    def __init__(
        self,
        workers: list[WorkerSpec],
        make: Callable[[WorkerSpec], Any] | None = None,
        *,
        inline: Any = None,
        say: Callable[[str], None] | None = None,
        faults: dict[str, Any] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if (make is None) == (inline is None):
            raise ValueError("a pool starts processes (make) or runs in this one (inline)")
        if not workers:
            raise ValueError("a pool needs a worker")
        self.workers = [_Worker(spec, free_bytes=spec.free_bytes) for spec in workers]
        self.make, self.inline = make, inline
        self.say = say or (lambda message: None)
        self.faults = dict(faults or {})
        self.progress = progress
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.done: dict[str, dict[str, Any]] = {}
        #: A job answered in parts: the keys that stand for it.
        self.parts: dict[str, list[str]] = {}
        self.waiting: list[str] = []
        self.threads: dict[str, tuple[threading.Thread, dict[str, Any]]] = {}
        #: A line a job as it ended: stage, name, worker, card, when, seconds, tries.
        self.ledger: list[dict[str, Any]] = []
        self.started = time.time()
        self._turn = 0
        self._cards = any(w.spec.card is not None for w in self.workers)

    # ---- the jobs ------------------------------------------------------------------------

    def add(self, *jobs: Job) -> None:
        for job in jobs:
            if job.key in self.jobs:
                raise ValueError(f"two jobs are named {job.key}")
            self.jobs[job.key] = job
            self.order.append(job.key)
            self.waiting.append(job.key)

    def finished(self, key: str) -> bool:
        """Whether ``key`` is done: itself, or every part it was answered in."""
        if key in self.parts:
            return all(self.finished(part) for part in self.parts[key])
        return key in self.done

    def _startable(self, job: Job) -> bool:
        for key in job.after:
            if key not in self.jobs:
                raise PoolError(f"{job.key} waits for {key}, which is no job of this pool")
            if not self.finished(key):
                return False
        return True

    def _takes(self, worker: _Worker, job: Job) -> bool:
        if job.on == CARD and self._cards and worker.spec.card is None:
            return False
        return worker.spec.index not in job.avoid

    def _fits(self, worker: _Worker, job: Job) -> bool:
        if job.on != CARD or not job.bytes or not worker.free_bytes:
            return True
        if job.bytes <= worker.free_bytes:
            return True
        # More than any card has: those with most are given it, and part it.
        most = max((w.free_bytes for w in self.workers if w.spec.card is not None), default=0.0)
        return job.bytes > most and worker.free_bytes >= 0.9 * most

    def _choose(self, worker: _Worker) -> Job | None:
        """The first job this worker may take: a card's own kind before the host's."""
        best: tuple[int, int, int, int] | None = None
        chosen: Job | None = None
        for place, key in enumerate(self.waiting):
            job = self.jobs[key]
            if job.on in (PARENT, THREAD) or not self._takes(worker, job):
                continue
            if not self._fits(worker, job) or not self._startable(job):
                continue
            other = 1 if (worker.spec.card is not None and job.on != CARD) else 0
            apart = 0 if (job.group and job.group == worker.group) else 1
            rank = (other, job.priority, apart, place)
            if best is None or rank < best:
                best, chosen = rank, job
        return chosen

    # ---- the workers -----------------------------------------------------------------------

    def start(self) -> None:
        """Every worker's process, each with its card and its threads in its environment."""
        if self.make is None:
            for worker in self.workers:
                worker.ready = True
            return
        context = multiprocessing.get_context("spawn")
        names = (*_THREAD_VARIABLES, "CUDA_VISIBLE_DEVICES", "REVERBERATE_NO_GPU")
        kept = {name: os.environ.get(name) for name in names}
        try:
            for worker in self.workers:
                spec = worker.spec
                for name in _THREAD_VARIABLES:
                    os.environ[name] = str(max(1, spec.threads))
                if spec.card is None:
                    os.environ["REVERBERATE_NO_GPU"] = "1"
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ.pop("REVERBERATE_NO_GPU", None)
                    # The card's own number on the machine, as ``nvidia-smi`` gives it.
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.card)
                here, there = context.Pipe()
                worker.process = context.Process(
                    target=serve, args=(spec, self.make, there, self.faults), daemon=True
                )
                worker.process.start()
                there.close()
                worker.link = here
        finally:
            for name, value in kept.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def said(self) -> list[Any]:
        """What every worker said of its device, once each is ready; waits for them."""
        if not any(w.ready or w.link is not None for w in self.workers):
            self.start()
        while True:
            starting = [w for w in self.workers if w.link is not None and not w.ready]
            if not starting:
                break
            for link in wait([w.link for w in starting], timeout=5.0):
                self._hear(next(w for w in starting if w.link is link))
        return [w.said for w in self.workers if w.said is not None]

    def close(self) -> None:
        for worker in self.workers:
            if worker.link is not None:
                with contextlib.suppress(OSError, ValueError):
                    worker.link.send(None)
        for worker in self.workers:
            if worker.process is not None:
                worker.process.join(timeout=10)
                if worker.process.is_alive():
                    worker.process.terminate()
            if worker.link is not None:
                worker.link.close()
            worker.process = worker.link = None
            worker.ready = False

    # ---- a job's end -------------------------------------------------------------------------

    def _ended(self, job: Job, worker: _Worker | None, seconds: float, record: Any) -> None:
        record = dict(record) if isinstance(record, dict) else {"result": record}
        parted = record.pop("parted", None)
        self.ledger.append(
            {
                "stage": job.stage,
                "name": job.name,
                "worker": None if worker is None else worker.spec.index,
                "card": None if worker is None else worker.spec.card,
                "ended_s": round(time.time() - self.started, 3),
                "seconds": round(seconds, 4),
                "tries": job.tries + 1,
                "parted": len(parted) if parted else 0,
            }
        )
        if parted:
            fresh = [
                Job(
                    stage=job.stage,
                    name=str(part["name"]),
                    payload=dict(part["payload"]),
                    on=job.on,
                    after=job.after,
                    bytes=float(part.get("bytes", 0.0)),
                    priority=job.priority,
                    split=job.split,
                )
                for part in parted
            ]
            self.parts[job.key] = [part.key for part in fresh]
            self.add(*fresh)
        self.done[job.key] = record
        if self.progress is not None:
            self.progress({"done": len(self.done), "jobs": len(self.jobs), "stage": job.stage})

    def _refused(self, job: Job, worker: _Worker | None, kind: str, said: str) -> None:
        """A job that did not end: parted, or tried elsewhere, or the run's end."""
        where = "this process" if worker is None else f"worker {worker.spec.index}"
        if kind == "memory" and job.split is not None:
            parts = job.split(job)
            if parts:
                self.say(f"{job.key} was refused its memory on {where}; in {len(parts)} parts")
                self.parts[job.key] = [part.key for part in parts]
                self.done[job.key] = {"parted_by_the_pool": len(parts)}
                self.add(*parts)
                return
        job.tries += 1
        if worker is not None:
            job.avoid.add(worker.spec.index)
        if job.tries >= MAX_TRIES:
            raise PoolError(f"{job.key} failed {job.tries} times, the last on {where}:\n{said}")
        if all(not self._takes(w, job) for w in self.workers if w.ready or w.job is not None):
            job.avoid.clear()
        self.say(f"{job.key} failed on {where} ({said.strip().splitlines()[-1][:160]}); again")
        self.waiting.insert(0, job.key)

    def _lost(self, worker: _Worker, said: str) -> None:
        """A worker that is gone: its job goes to another, and it takes no more."""
        job, worker.job = worker.job, None
        worker.ready = False
        if worker.link is not None:
            worker.link.close()
        worker.link = None
        self.say(f"worker {worker.spec.index} is gone: {said.strip().splitlines()[-1][:200]}")
        if not any(w.link is not None for w in self.workers):
            raise PoolError(f"every worker is gone; the last said:\n{said}")
        if job is not None:
            self._refused(job, worker, "lost", said)

    def _hear(self, worker: _Worker) -> None:
        try:
            message = worker.link.recv()
        except (EOFError, OSError):
            self._lost(worker, "its process ended")
            return
        kind = message[0]
        if kind == "ready":
            worker.ready, worker.free_bytes = True, float(message[1]) or worker.free_bytes
            worker.said = message[2]
            return
        if kind == "broken":
            self._lost(worker, str(message[1]))
            return
        job, worker.job = worker.job, None
        assert job is not None
        worker.busy_s += float(message[2])
        worker.jobs += 1
        if kind == "done":
            worker.free_bytes = float(message[3]) or worker.free_bytes
            self._ended(job, worker, float(message[2]), message[1])
        else:
            self._refused(job, worker, kind, str(message[1]))

    # ---- the run ---------------------------------------------------------------------------

    def _in_this_process(self) -> bool:
        """The merges that are ready, and the threads to start; whether anything moved."""
        moved = False
        for key in list(self.waiting):
            job = self.jobs[key]
            if job.on not in (PARENT, THREAD) or not self._startable(job):
                continue
            assert job.work is not None
            self.waiting.remove(key)
            moved = True
            if job.on == PARENT or self.make is None:
                started = time.time()
                self._ended(job, None, time.time() - started, job.work() or {})
                self.ledger[-1]["seconds"] = round(time.time() - started, 4)
                continue
            held: dict[str, Any] = {"started": time.time()}

            def call(job: Job = job, held: dict[str, Any] = held) -> None:
                try:
                    assert job.work is not None
                    held["record"] = job.work() or {}
                except BaseException as error:  # noqa: BLE001 - raised by the queue's loop
                    held["error"] = error

            thread = threading.Thread(target=call, name=key, daemon=True)
            thread.start()
            self.threads[key] = (thread, held)
        for key, (thread, held) in list(self.threads.items()):
            if thread.is_alive():
                continue
            del self.threads[key]
            if "error" in held:
                raise held["error"]
            self._ended(self.jobs[key], None, time.time() - held["started"], held["record"])
            moved = True
        return moved

    def _hand_out(self) -> bool:
        moved = False
        idle = [w for w in self.workers if w.ready and w.job is None]
        # A worker without a card first: it leaves the cards to what only a card does.
        idle.sort(key=lambda w: (w.spec.card is not None, w.spec.index))
        if self.make is None and idle:
            idle = [idle[self._turn % len(idle)]]
            self._turn += 1
        for worker in idle:
            job = self._choose(worker)
            if job is None and self.make is None:
                # In one process a job no worker's turn fits still runs: on any that takes it.
                for other in self.workers:
                    job = self._choose(other)
                    if job is not None:
                        worker = other
                        break
            if job is None:
                continue
            self.waiting.remove(job.key)
            moved = True
            worker.group = job.group
            if self.make is None:
                self._inline(worker, job)
                continue
            worker.job, worker.started = job, time.time()
            worker.link.send((job.stage, job.name, job.payload, job.tries))
        return moved

    def _inline(self, worker: _Worker, job: Job) -> None:
        started = time.time()
        try:
            _fault(self.faults, job.key, job.tries)
            record = self.inline.run_job(job.stage, job.name, job.payload)
        except MemoryError as error:
            worker.busy_s += time.time() - started
            self._refused(job, worker, "memory", repr(error))
            return
        except PoolError:
            raise
        except Exception:  # noqa: BLE001 - tried again, as a worker's would be
            worker.busy_s += time.time() - started
            self._refused(job, worker, "failed", traceback.format_exc())
            return
        worker.busy_s += time.time() - started
        worker.jobs += 1
        self._ended(job, worker, time.time() - started, record)

    def run(self) -> dict[str, dict[str, Any]]:
        """Every job added, and those they add; each one's record by its key."""
        if not any(w.ready or w.link is not None for w in self.workers):
            self.start()
        while self.waiting or self.threads or any(w.job is not None for w in self.workers):
            moved = self._in_this_process()
            moved = self._hand_out() or moved
            busy = [w for w in self.workers if w.link is not None and (w.job or not w.ready)]
            if busy:
                for link in wait([w.link for w in busy], timeout=0.2 if self.threads else 5.0):
                    self._hear(next(w for w in busy if w.link is link))
            elif self.threads:
                time.sleep(0.05)
            elif not moved and self.waiting:
                stuck = ", ".join(self.waiting[:5])
                raise PoolError(f"{len(self.waiting)} job(s) can never start: {stuck}")
        return self.done

    def account(self) -> dict[str, Any]:
        """What the run was made of: by stage and by worker, seconds of work and of wall."""
        stages: dict[str, dict[str, Any]] = {}
        for line in self.ledger:
            held = stages.setdefault(
                line["stage"], {"jobs": 0, "work_s": 0.0, "first_s": None, "last_s": 0.0}
            )
            held["jobs"] += 1
            held["work_s"] = round(held["work_s"] + line["seconds"], 3)
            began = line["ended_s"] - line["seconds"]
            held["first_s"] = began if held["first_s"] is None else min(held["first_s"], began)
            held["last_s"] = max(held["last_s"], line["ended_s"])
        for held in stages.values():
            held["wall_s"] = round(held["last_s"] - (held["first_s"] or 0.0), 3)
            held["first_s"] = round(held["first_s"] or 0.0, 3)
        return {
            "stages": stages,
            "workers": [
                {
                    "index": w.spec.index,
                    "card": w.spec.card,
                    "jobs": w.jobs,
                    "busy_s": round(w.busy_s, 3),
                }
                for w in self.workers
            ],
            "wall_s": round(time.time() - self.started, 3),
        }
