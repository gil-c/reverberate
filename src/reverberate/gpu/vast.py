"""Renting a GPU from Vast.ai, with the teardown guaranteed rather than hoped for.

Roadmap section 13 is a list of ways to lose money: an instance left billing
overnight, a spend figure nobody tracked, a credential written to the rented
machine's disk. This module exists so that none of those depends on an agent or
a human remembering.

Three things are enforced here rather than documented:

- **Every rental carries a deadline.** :func:`rent` refuses to create an
  instance without one and arms a detached watchdog process before returning.
  The watchdog outlives the session that started it, retries, and verifies the
  instance is gone rather than assuming it.
- **Spend is ledgered.** Every rental and teardown is appended to
  ``<data root>/runs/gpu_spend.jsonl``, and :func:`rent` refuses to start when
  the total would pass :data:`SPEND_CEILING_USD`.
- **The credential is read from the environment and nowhere else**, per section
  10. It is never logged, never written to the ledger, and never sent to the
  rented machine.

The pure parts (query building, offer ranking, cost arithmetic, ledger totals)
take their inputs explicitly and are covered by tests. Only :class:`VastClient`
touches the network.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reverberate.settings import runs_dir

__all__ = [
    "API_KEY_ENV",
    "SPEND_CEILING_USD",
    "Instance",
    "Offer",
    "Rental",
    "VastClient",
    "VastError",
    "cheapest",
    "estimate_cost_usd",
    "ledger_total_usd",
    "rent",
    "search_query",
    "teardown",
]

#: Vast.ai credential, read from the process environment only (section 10).
#: Populate it with ``reverberate.auth.inject(["VASTAI_API_KEY"])``.
API_KEY_ENV = "VASTAI_API_KEY"

#: Hard ceiling on GPU spend across the whole project, in US dollars
#: (roadmap section 13.2).
SPEND_CEILING_USD = 1000.0

#: Name of the spend ledger inside the runs directory.
LEDGER_NAME = "gpu_spend.jsonl"

_API = "https://console.vast.ai/api/v0"


class VastError(RuntimeError):
    """Any failure talking to Vast.ai, or any refusal to spend."""


@dataclass(frozen=True)
class Offer:
    """One rentable machine, as advertised by Vast.ai."""

    id: int
    gpu_name: str
    num_gpus: int
    #: Total price in US dollars per hour, storage and bandwidth included.
    dph_total: float
    gpu_ram_gb: float
    cpu_cores: float
    ram_gb: float
    disk_gb: float
    cuda_max: float
    reliability: float
    inet_down_mbps: float
    location: str

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Offer:
        return cls(
            id=int(raw["id"]),
            gpu_name=str(raw.get("gpu_name", "")),
            num_gpus=int(raw.get("num_gpus", 1)),
            dph_total=float(raw.get("dph_total", 0.0)),
            gpu_ram_gb=float(raw.get("gpu_ram", 0.0)) / 1024.0,
            cpu_cores=float(raw.get("cpu_cores_effective", 0.0)),
            ram_gb=float(raw.get("cpu_ram", 0.0)) / 1024.0,
            disk_gb=float(raw.get("disk_space", 0.0)),
            cuda_max=float(raw.get("cuda_max_good", 0.0)),
            reliability=float(raw.get("reliability2", 0.0)),
            inet_down_mbps=float(raw.get("inet_down", 0.0)),
            location=str(raw.get("geolocation") or "unknown"),
        )

    def describe(self) -> str:
        """One line, in the shape section 13.1 wants stated before renting."""
        return (
            f"offer {self.id}: {self.num_gpus}x {self.gpu_name} "
            f"{self.gpu_ram_gb:.0f} GB, {self.dph_total:.3f} USD/h, "
            f"{self.cpu_cores:.0f} vCPU, {self.ram_gb:.0f} GB RAM, "
            f"{self.disk_gb:.0f} GB disk, CUDA {self.cuda_max}, "
            f"reliability {self.reliability:.3f}, {self.location}"
        )


@dataclass(frozen=True)
class Instance:
    """A rented machine, as reported by Vast.ai."""

    id: int
    status: str
    dph_total: float
    ssh_host: str
    ssh_port: int
    gpu_name: str
    #: Unix timestamp the contract started, or ``None`` before it does.
    start_date: float | None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Instance:
        start = raw.get("start_date")
        return cls(
            id=int(raw["id"]),
            status=str(raw.get("actual_status") or raw.get("cur_state") or "unknown"),
            dph_total=float(raw.get("dph_total", 0.0)),
            ssh_host=str(raw.get("ssh_host") or ""),
            ssh_port=int(raw.get("ssh_port") or 0),
            gpu_name=str(raw.get("gpu_name", "")),
            start_date=float(start) if start else None,
        )

    def uptime_hours(self, now: float | None = None) -> float:
        """Hours billed so far, 0.0 before the contract starts."""
        if self.start_date is None:
            return 0.0
        return max(0.0, ((now if now is not None else time.time()) - self.start_date) / 3600.0)


@dataclass(frozen=True)
class Rental:
    """What :func:`rent` returns: the instance and the deadline that kills it."""

    instance_id: int
    offer: Offer
    #: Unix timestamp at which the watchdog destroys the instance.
    deadline: float
    #: Process id of the detached watchdog.
    watchdog_pid: int


def search_query(
    gpu_name: str = "RTX_4090",
    num_gpus: int = 1,
    min_disk_gb: int = 60,
    min_reliability: float = 0.99,
    min_cuda: float = 12.0,
    min_inet_down_mbps: int = 300,
) -> str:
    """Build a Vast.ai offer query string.

    Kept separate from the request so the filter that picked a machine can be
    recorded in provenance verbatim, and asserted on in tests.
    """
    return " ".join(
        [
            f"gpu_name={gpu_name}",
            f"num_gpus={num_gpus}",
            "rentable=true",
            f"disk_space>{min_disk_gb}",
            f"reliability>{min_reliability}",
            f"cuda_vers>={min_cuda}",
            f"inet_down>{min_inet_down_mbps}",
        ]
    )


def estimate_cost_usd(dph: float, hours: float) -> float:
    """Dollars for ``hours`` at ``dph`` dollars per hour, never negative."""
    return max(0.0, dph) * max(0.0, hours)


def cheapest(offers: Iterable[Offer], min_gpu_ram_gb: float = 0.0) -> Offer | None:
    """The least expensive offer with at least ``min_gpu_ram_gb`` of VRAM."""
    eligible = [offer for offer in offers if offer.gpu_ram_gb >= min_gpu_ram_gb]
    return min(eligible, key=lambda offer: offer.dph_total) if eligible else None


def ledger_path() -> Path:
    """Where rentals are recorded. One JSON object per line, appended."""
    return runs_dir() / LEDGER_NAME


def ledger_total_usd(entries: Iterable[dict[str, Any]]) -> float:
    """Dollars spent, taking each rental's actual cost once it is torn down.

    A rental that has not been torn down yet counts at its full budgeted worst
    case, so the ceiling is never breached by a run still in flight.
    """
    actual: dict[int, float] = {}
    budget: dict[int, float] = {}
    for entry in entries:
        instance_id = int(entry.get("instance_id", 0))
        if entry.get("event") == "rent":
            budget[instance_id] = float(entry.get("budget_usd", 0.0))
        elif entry.get("event") == "teardown":
            actual[instance_id] = float(entry.get("cost_usd", 0.0))
    return sum(actual.get(instance_id, cost) for instance_id, cost in budget.items())


def read_ledger(path: Path | None = None) -> list[dict[str, Any]]:
    """Every ledger entry, or an empty list when nothing has been rented."""
    path = path or ledger_path()
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            entries.append(json.loads(stripped))
    return entries


def append_ledger(entry: dict[str, Any], path: Path | None = None) -> None:
    """Append one entry, with a UTC timestamp, creating the file if needed."""
    path = path or ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
    with path.open("a") as handle:
        handle.write(json.dumps(stamped) + "\n")


def _coerce(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_query(query: str) -> dict[str, Any]:
    """Turn ``"a=1 b>2"`` into the JSON filter the bundles endpoint expects."""
    ops = ((">=", "gte"), ("<=", "lte"), ("!=", "neq"), (">", "gt"), ("<", "lt"), ("=", "eq"))
    parsed: dict[str, Any] = {}
    for token in query.split():
        for symbol, name in ops:
            if symbol in token:
                field, _, value = token.partition(symbol)
                parsed[field] = {name: _coerce(value)}
                break
    parsed.setdefault("type", "on-demand")
    return parsed


class VastClient:
    """The network edge. Everything that talks to Vast.ai goes through here."""

    def __init__(self, api_key: str | None = None, timeout: float = 60.0) -> None:
        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise VastError(
                f"{API_KEY_ENV} is not set; load it with "
                f'reverberate.auth.inject(["{API_KEY_ENV}"]) before renting'
            )
        self._key = key
        self._timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"{_API}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)  # noqa: S310
        request.add_header("Authorization", f"Bearer {self._key}")
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            # The body can echo the request; never let it reach a log with the key in it.
            raise VastError(f"{method} {path} failed: HTTP {error.code}") from None
        except urllib.error.URLError as error:
            raise VastError(f"{method} {path} failed: {error.reason}") from None

    def search(self, query: str, limit: int = 20) -> list[Offer]:
        """Offers matching ``query``, cheapest first."""
        params = urllib.parse.urlencode({"q": json.dumps(parse_query(query)), "limit": limit})
        payload = self._request("GET", f"/bundles/?{params}")
        offers = [Offer.from_api(raw) for raw in payload.get("offers", [])]
        return sorted(offers, key=lambda offer: offer.dph_total)

    def instances(self) -> list[Instance]:
        """Every instance on the account, rented or still loading."""
        payload = self._request("GET", "/instances/")
        return [Instance.from_api(raw) for raw in payload.get("instances", [])]

    def instance(self, instance_id: int) -> Instance | None:
        """One instance, or ``None`` once it no longer exists."""
        for found in self.instances():
            if found.id == instance_id:
                return found
        return None

    def create(
        self,
        offer_id: int,
        image: str,
        disk_gb: int = 60,
        onstart_cmd: str = "touch /root/.onstart_done; sleep infinity",
    ) -> int:
        """Rent ``offer_id`` and return the new instance id."""
        payload = self._request(
            "PUT",
            f"/asks/{offer_id}/",
            {
                "client_id": "me",
                "image": image,
                "disk": disk_gb,
                "runtype": "ssh",
                "onstart": onstart_cmd,
            },
        )
        if not payload.get("success"):
            raise VastError(f"vast refused to create an instance on offer {offer_id}")
        return int(payload["new_contract"])

    def destroy(self, instance_id: int) -> None:
        """Ask Vast.ai to destroy an instance. Verify with :meth:`instance`."""
        self._request("DELETE", f"/instances/{instance_id}/")

    def destroy_and_verify(self, instance_id: int, attempts: int = 5, pause: float = 20.0) -> bool:
        """Destroy, then confirm it is gone. Section 13.5: do not assume."""
        for _ in range(attempts):
            with contextlib.suppress(VastError):
                self.destroy(instance_id)
            time.sleep(pause)
            if self.instance(instance_id) is None:
                return True
        return False


def arm_hard_stop(instance_id: int, hours: float) -> int:
    """Spawn a detached watchdog that destroys ``instance_id`` after ``hours``.

    Detached on purpose: the whole point is that it survives the session, the
    terminal and the agent that started it. Returns the watchdog's pid.
    """
    command = [
        sys.executable,
        "-m",
        "reverberate.gpu.vast",
        "hard-stop",
        str(instance_id),
        str(hours),
    ]
    log = runs_dir() / f"hard_stop_{instance_id}.log"
    with log.open("a") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return process.pid


def rent(
    client: VastClient,
    offer: Offer,
    hours: float,
    image: str,
    disk_gb: int = 60,
    ceiling_usd: float = SPEND_CEILING_USD,
) -> Rental:
    """Rent ``offer`` for at most ``hours``, with teardown already scheduled.

    Refuses a rental with no deadline, and refuses one that would take the
    project past ``ceiling_usd``. The watchdog is armed immediately after the
    instance exists and before this returns, so there is no window in which a
    forgotten instance bills unattended.
    """
    if hours <= 0:
        raise VastError("a rental needs a positive deadline in hours (section 13.5)")
    budget = estimate_cost_usd(offer.dph_total, hours)
    already = ledger_total_usd(read_ledger())
    if already + budget > ceiling_usd:
        raise VastError(
            f"rental would take project spend to {already + budget:.2f} USD, "
            f"past the {ceiling_usd:.0f} USD ceiling (already {already:.2f})"
        )
    instance_id = client.create(offer.id, image=image, disk_gb=disk_gb)
    deadline = time.time() + hours * 3600.0
    try:
        pid = arm_hard_stop(instance_id, hours)
    except OSError:
        client.destroy_and_verify(instance_id)
        raise VastError("could not arm the hard stop, so the instance was destroyed") from None
    append_ledger(
        {
            "event": "rent",
            "instance_id": instance_id,
            "offer_id": offer.id,
            "gpu": offer.gpu_name,
            "dph_total": offer.dph_total,
            "hours": hours,
            "budget_usd": round(budget, 4),
            "image": image,
            "watchdog_pid": pid,
        }
    )
    return Rental(instance_id=instance_id, offer=offer, deadline=deadline, watchdog_pid=pid)


def teardown(client: VastClient, instance_id: int) -> bool:
    """Destroy an instance, verify it, and record what it actually cost."""
    found = client.instance(instance_id)
    hours = found.uptime_hours() if found else 0.0
    cost = estimate_cost_usd(found.dph_total, hours) if found else 0.0
    gone = client.destroy_and_verify(instance_id)
    append_ledger(
        {
            "event": "teardown",
            "instance_id": instance_id,
            "hours": round(hours, 4),
            "cost_usd": round(cost, 4),
            "verified_gone": gone,
        }
    )
    return gone


def _hard_stop(instance_id: int, hours: float) -> int:
    """Watchdog body: sleep, then destroy and verify. Run detached."""
    time.sleep(max(0.0, hours * 3600.0))
    from reverberate import auth

    auth.inject([API_KEY_ENV])
    client = VastClient()
    if client.instance(instance_id) is None:
        print(f"hard stop: instance {instance_id} already gone")
        return 0
    gone = teardown(client, instance_id)
    print(f"hard stop: instance {instance_id} destroyed={gone}")
    return 0 if gone else 1


def _main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print(
            "usage: python -m reverberate.gpu.vast [search|list|destroy|hard-stop] ...",
            file=sys.stderr,
        )
        return 2
    command, rest = args[0], args[1:]

    if command == "hard-stop":
        return _hard_stop(int(rest[0]), float(rest[1]))

    from reverberate import auth

    auth.inject([API_KEY_ENV])
    client = VastClient()

    if command == "search":
        for offer in client.search(search_query())[:10]:
            print(offer.describe())
    elif command == "list":
        for found in client.instances():
            print(
                f"{found.id} {found.status} {found.gpu_name} "
                f"{found.dph_total:.3f} USD/h up {found.uptime_hours():.2f} h "
                f"ssh {found.ssh_host}:{found.ssh_port}"
            )
        print(f"project spend so far: {ledger_total_usd(read_ledger()):.2f} USD")
    elif command == "destroy":
        gone = teardown(client, int(rest[0]))
        print(f"instance {rest[0]} destroyed={gone}")
        return 0 if gone else 1
    else:
        print(f"unknown command {command!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
