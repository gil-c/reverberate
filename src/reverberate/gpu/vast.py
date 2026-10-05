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
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reverberate.settings import runs_dir

if TYPE_CHECKING:  # pragma: no cover - import cycle, needed for the annotation only
    from reverberate.wave.remote import Machine

__all__ = [
    "API_KEY_ENV",
    "SPEND_CEILING_USD",
    "Instance",
    "Offer",
    "Rental",
    "VastClient",
    "VastError",
    "cheapest",
    "direct_address",
    "estimate_cost_usd",
    "ledger_total_usd",
    "account_identity",
    "rent",
    "search_query",
    "teardown",
    "wait_for_ssh",
]

#: Vast.ai credential, read from the process environment only (section 10).
#: Populate it with ``reverberate.auth.inject(["VASTAI_API_KEY"])``.
API_KEY_ENV = "VASTAI_API_KEY"

#: Hard ceiling on GPU spend across the whole project, in US dollars
#: (roadmap section 13.2).
SPEND_CEILING_USD = 1000.0

#: Name of the spend ledger inside the runs directory.
LEDGER_NAME = "gpu_spend.jsonl"

#: The disk an offer's hourly price is stated with, GB, and the hours of the month its
#: storage is priced by: Vast's own search prices 5 GB unless asked otherwise.
OFFER_DISK_GB = 5.0
HOURS_A_MONTH = 730.0

_API = "https://console.vast.ai/api"
_API_VERSION = "v0"


class VastError(RuntimeError):
    """Any failure talking to Vast.ai, or any refusal to spend.

    ``status`` carries the HTTP code when there was one, so a caller can tell a
    machine that does not exist from an API it could not reach. That difference
    decides whether a rented card gets destroyed.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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
    #: Clock of one core, in GHz, as the offer advertises it. Carried because
    #: **this workload is bought by the core, not by the count.** W35 measured
    #: 192 vCPU voxelising at the same speed as ten, and 2026-09-07 measured a
    #: 48 vCPU box at 3.0 GHz taking about twice the laptop's time on the same
    #: flat at 4 kHz. Nothing in this module could act on that until the field
    #: was read.
    cpu_ghz: float
    #: What the offer calls the part, for the record. Two boxes at the same
    #: advertised clock are not the same machine.
    cpu_name: str
    ram_gb: float
    disk_gb: float
    cuda_max: float
    reliability: float
    inet_down_mbps: float
    location: str
    #: What the host says its line sends, Mbit/s: what a pack's way home is bounded by
    #: once it does not go through the proxy. 0 when the API does not say.
    inet_up_mbps: float = 0.0
    #: What the host asks for a gigabyte of disk a month, USD (``storage_cost``). **An
    #: offer's ``dph_total`` is not what the rental is billed**: it is priced with a few
    #: gigabytes of disk, and the two whole scenes of 2026-10-05, rented with 219 GB,
    #: were billed 1.382 USD/h for an offer of 1.284 and 0.797 for one of 0.543.
    #: :meth:`billed_dph` adds the disk. 0 when the API does not say.
    storage_usd_gb_month: float = 0.0
    #: The host behind the offer. One host is advertised as several offers, one
    #: per count of its cards, so a host to avoid is named by this and not by
    #: ``id``. 0 when the API does not say.
    machine_id: int = 0

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Offer:
        return cls(
            id=int(raw["id"]),
            gpu_name=str(raw.get("gpu_name", "")),
            num_gpus=int(raw.get("num_gpus", 1)),
            dph_total=float(raw.get("dph_total", 0.0)),
            gpu_ram_gb=float(raw.get("gpu_ram", 0.0)) / 1024.0,
            cpu_cores=float(raw.get("cpu_cores_effective", 0.0)),
            cpu_ghz=float(raw.get("cpu_ghz", 0.0) or 0.0),
            cpu_name=str(raw.get("cpu_name", "") or "").strip(),
            ram_gb=float(raw.get("cpu_ram", 0.0)) / 1024.0,
            disk_gb=float(raw.get("disk_space", 0.0)),
            cuda_max=float(raw.get("cuda_max_good", 0.0)),
            reliability=float(raw.get("reliability2", 0.0)),
            inet_down_mbps=float(raw.get("inet_down", 0.0)),
            location=str(raw.get("geolocation") or "unknown"),
            inet_up_mbps=float(raw.get("inet_up", 0.0) or 0.0),
            storage_usd_gb_month=float(raw.get("storage_cost", 0.0) or 0.0),
            machine_id=int(raw.get("machine_id") or 0),
        )

    def billed_dph(self, disk_gb: float) -> float:
        """USD an hour with ``disk_gb`` of disk: the offer's hour and the disk's share of a month.

        The offer's hour is taken to hold the :data:`OFFER_DISK_GB` an
        offer is priced with; a month is 730 hours. An estimate until a
        rental's own rate has been set against it: the instance, once it
        exists, says what it bills.
        """
        more = max(0.0, float(disk_gb) - OFFER_DISK_GB)
        return self.dph_total + self.storage_usd_gb_month * more / HOURS_A_MONTH

    def describe(self) -> str:
        """One line, in the shape section 13.1 wants stated before renting."""
        return (
            f"offer {self.id}: {self.num_gpus}x {self.gpu_name} "
            f"{self.gpu_ram_gb:.0f} GB, {self.dph_total:.3f} USD/h, "
            f"{self.cpu_cores:.0f} vCPU at {self.cpu_ghz:.1f} GHz "
            f"({self.cpu_name or 'unnamed'}), {self.ram_gb:.0f} GB RAM, "
            f"{self.disk_gb:.0f} GB disk, CUDA {self.cuda_max}, "
            f"reliability {self.reliability:.3f}, {self.location}"
            f"{f', machine {self.machine_id}' if self.machine_id else ''}"
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
    #: Vast.ai's own account of what the instance is doing, and the only way to
    #: tell "still pulling a 5 GB image" from "this host is broken". Both show
    #: as ``loading``, and one is worth waiting twenty minutes for while the
    #: other is worth nothing at all. Empty when the API offers no message.
    status_msg: str = ""
    #: The instance's own address and the host port its container's 22 is mapped to,
    #: where it was created with direct ssh (:data:`RUNTYPE_DIRECT`) and the API gives
    #: both: ``public_ipaddr`` and ``ports["22/tcp"][0]["HostPort"]``. ``None`` otherwise,
    #: and ``ssh_host`` and ``ssh_port``, the proxy's, are then the only way in.
    direct: tuple[str, int] | None = None

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
            status_msg=" ".join(str(raw.get("status_msg") or "").split()),
            direct=direct_address(raw),
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


#: How an instance is asked for (``runtype`` of the request that creates it). With the
#: first, Vast maps the container's port 22 to a port of the host's own address, beside
#: the proxy's: ``ssh -p <HostPort> root@<public_ipaddr>``. The words are Vast's own
#: command line's for ``--ssh --direct`` (``ssh_direc``, without the t), and **no run of
#: this project has created an instance with them yet**: :meth:`VastClient.create` asks
#: again with the second where the first is refused.
RUNTYPE_DIRECT = "ssh_direc ssh_proxy"
RUNTYPE_PROXY = "ssh"


def direct_address(raw: dict[str, Any]) -> tuple[str, int] | None:
    """An instance's own address and its ssh port there, from the API's record; or ``None``.

    The record of an instance created with direct ssh carries
    ``public_ipaddr`` and, once its container is up, ``ports``, Docker's
    own map: ``{"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40022"}]}``.
    One without the mapping, or with a malformed one, has no direct
    address and is reached through the proxy.
    """
    address = str(raw.get("public_ipaddr") or "").strip()
    ports = raw.get("ports")
    if not address or not isinstance(ports, dict):
        return None
    mapped: Any = ports.get("22/tcp")
    try:
        port = int(mapped[0]["HostPort"])
    except (TypeError, KeyError, IndexError, ValueError):
        return None
    return (address, port) if port > 0 else None


def search_query(
    gpu_name: str = "RTX 4090",
    num_gpus: int = 1,
    min_disk_gb: int = 60,
    min_reliability: float = 0.99,
    min_cuda: float = 12.0,
    min_inet_down_mbps: int = 300,
    min_gpu_ram_gb: float = 0.0,
    min_cpu_cores: int = 0,
    min_cpu_ghz: float = 0.0,
) -> str:
    """Build a Vast.ai offer query string.

    Kept separate from the request so the filter that picked a machine can be
    recorded in provenance verbatim, and asserted on in tests.

    ``gpu_name`` is dropped from the filter when empty, which is how to ask for
    "any card with at least this much VRAM" rather than naming one.

    The card name carries a **space**, not an underscore. Vast.ai's own
    documentation writes ``RTX_4090`` and the web console shows it that way, but
    the field the API matches against stores ``RTX 4090``, so an underscore
    matches nothing and the search returns zero offers with no error at all. The
    default was written with an underscore and silently found nothing until it
    was checked against the raw offer records.
    """
    tokens = [
        f"num_gpus={num_gpus}",
        "rentable=true",
        f"disk_space>{min_disk_gb}",
        f"reliability>{min_reliability}",
        f"cuda_vers>={min_cuda}",
        f"inet_down>{min_inet_down_mbps}",
    ]
    if gpu_name:
        tokens.insert(0, f"gpu_name={gpu_name}")
    if min_gpu_ram_gb > 0:
        tokens.append(f"gpu_ram>={int(min_gpu_ram_gb * 1024)}")
    if min_cpu_cores > 0:
        tokens.append(f"cpu_cores_effective>={min_cpu_cores}")
    if min_cpu_ghz > 0:
        # The one filter that matters for a CPU-bound stage, and the one that
        # was missing: see Offer.cpu_ghz for what its absence cost.
        tokens.append(f"cpu_ghz>={min_cpu_ghz}")
    return " ".join(tokens)


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


def deadline_of(instance_id: int, entries: Iterable[dict[str, Any]] | None = None) -> float | None:
    """When the watchdog of a rental destroys it, from this machine's ledger; ``None`` unknown.

    A run resumed on its instance did not rent it and was not told its
    cap: the ledger's ``rent`` entry has when and for how many hours.
    """
    import calendar

    found = None
    for entry in read_ledger() if entries is None else entries:
        if entry.get("event") == "rent" and int(entry.get("instance_id", 0)) == int(instance_id):
            found = entry
    if found is None or not found.get("at") or not found.get("hours"):
        return None
    try:
        rented = calendar.timegm(time.strptime(str(found["at"]), "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None
    return float(rented) + float(found["hours"]) * 3600.0


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
    """Turn ``"a=1 b>2"`` into the JSON filter the bundles endpoint expects.

    A value may hold spaces, and one does: every card is named ``RTX 3090``
    and not ``RTX_3090``. A token with no operator in it therefore continues
    the value of the token before it. Splitting on whitespace alone sent
    ``gpu_name={"eq": "RTX"}`` to the endpoint, which matches no card at all
    and returns an empty list with no error; two sessions read that as "no
    card is free tonight".
    """
    ops = ((">=", "gte"), ("<=", "lte"), ("!=", "neq"), (">", "gt"), ("<", "lt"), ("=", "eq"))
    parsed: dict[str, Any] = {}
    last: tuple[str, str] | None = None
    for token in query.split():
        for symbol, name in ops:
            if symbol in token:
                field, _, value = token.partition(symbol)
                parsed[field] = {name: _coerce(value)}
                last = (field, name)
                break
        else:
            if last is not None and isinstance(parsed[last[0]][last[1]], str):
                parsed[last[0]][last[1]] = f"{parsed[last[0]][last[1]]} {token}"
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
        #: Set once a request for direct ssh was refused and the plain one taken: said by
        #: the caller, since every transfer of the run then goes through the proxy.
        self.direct_refused = False

    @staticmethod
    def _ssl_context() -> ssl.SSLContext:
        """Verify against ``certifi`` when it is installed, the system store otherwise.

        **The symptom this prevents looks like a network outage and is not one.**
        The python.org framework build of Python on macOS ships no certificate
        store of its own, so every call through ``urllib`` fails with
        ``CERTIFICATE_VERIFY_FAILED, unable to get local issuer certificate``,
        while the store client keeps working because ``boto3`` carries
        ``certifi``. Two sessions of this project met it and one of them read it
        as the API being unreachable.

        Verification is never turned off. If ``certifi`` is absent the system
        store is used, which is the standard behaviour.
        """
        try:
            import certifi
        except ImportError:  # pragma: no cover - certifi arrives with boto3
            return ssl.create_default_context()
        return ssl.create_default_context(cafile=certifi.where())

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        api_version: str = _API_VERSION,
    ) -> Any:
        url = f"{_API}/{api_version}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)  # noqa: S310
        request.add_header("Authorization", f"Bearer {self._key}")
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            context = self._ssl_context()
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=self._timeout, context=context
            ) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            # The body can echo the request; never let it reach a log with the key in it.
            raise VastError(f"{method} {path} failed: HTTP {error.code}", error.code) from None
        except urllib.error.URLError as error:
            hint = ""
            if isinstance(error.reason, ssl.SSLCertVerificationError):
                hint = (
                    ". This is a certificate store problem on this machine and not a "
                    "network one; install certifi, or run "
                    "'Install Certificates.command' from the Python framework"
                )
            raise VastError(f"{method} {path} failed: {error.reason}{hint}") from None

    def search(self, query: str, limit: int = 20) -> list[Offer]:
        """Offers matching ``query``, cheapest first.

        ``limit`` and ``order`` travel inside the ``q`` document. The bundles
        endpoint rejects them as URL parameters with HTTP 400, which is what it
        did to every search this module made before 2026-08.
        """
        filters = {**parse_query(query), "order": [["dph_total", "asc"]], "limit": limit}
        params = urllib.parse.urlencode({"q": json.dumps(filters)})
        payload = self.request("GET", f"/bundles/?{params}")
        offers = [Offer.from_api(raw) for raw in payload.get("offers", [])]
        return sorted(offers, key=lambda offer: offer.dph_total)

    def instances(self) -> list[Instance]:
        """Every instance on the account, rented or still loading.

        The listing lives under ``/api/v1``: since 2026-08 the ``v0`` form
        answers HTTP 410 and names its replacement.
        """
        payload = self.request("GET", "/instances/", api_version="v1")
        return [Instance.from_api(raw) for raw in payload.get("instances", [])]

    def instance(self, instance_id: int) -> Instance | None:
        """One instance, or ``None`` once it no longer exists.

        Asks for the single instance rather than filtering the listing, so a
        watchdog polling one rental stays cheap and keeps working even while
        the listing endpoint is being moved around.
        """
        try:
            payload = self.request("GET", f"/instances/{instance_id}/")
        except VastError as error:
            # **Absent and unreachable are not the same answer.** Swallowing
            # every failure here makes a transient API fault look exactly like
            # an instance that no longer exists, and two callers act on that:
            # ``destroy_and_verify`` would report a card destroyed while it is
            # still billing, and ``wait_for_ssh`` would abandon a rental that is
            # merely starting. Only a 404 means gone.
            if error.status == 404:
                return None
            raise
        raw = payload.get("instances")
        if not raw:
            return None
        if isinstance(raw, list):
            raw = next((item for item in raw if item.get("id") == instance_id), None)
            if raw is None:
                return None
        return Instance.from_api(raw)

    def create(
        self,
        offer_id: int,
        image: str,
        disk_gb: int = 60,
        onstart_cmd: str = "touch /root/.onstart_done; sleep infinity",
        direct: bool = True,
    ) -> int:
        """Rent ``offer_id`` and return the new instance id.

        ``direct`` asks for the instance's port 22 on the host's own
        address as well as through the proxy (:data:`RUNTYPE_DIRECT`): a
        pack came home at 2 MB/s a stream through the proxy on a line that
        carries 17 to 70. A request so worded that Vast refuses as
        malformed (HTTP 400) is made again as it always was, and
        ``direct_refused`` says so; any other refusal is the offer's.
        """
        body = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "runtype": RUNTYPE_DIRECT if direct else RUNTYPE_PROXY,
            "onstart": onstart_cmd,
        }
        try:
            payload = self.request("PUT", f"/asks/{offer_id}/", body)
        except VastError as error:
            if not direct or error.status != 400:
                raise
            self.direct_refused = True
            payload = self.request("PUT", f"/asks/{offer_id}/", {**body, "runtype": RUNTYPE_PROXY})
        if not payload.get("success"):
            raise VastError(f"vast refused to create an instance on offer {offer_id}")
        return int(payload["new_contract"])

    def destroy(self, instance_id: int) -> None:
        """Ask Vast.ai to destroy an instance. Verify with :meth:`instance`."""
        self.request("DELETE", f"/instances/{instance_id}/")

    def destroy_and_verify(self, instance_id: int, attempts: int = 5, pause: float = 20.0) -> bool:
        """Destroy, then confirm it is gone. Section 13.5: do not assume."""
        for _ in range(attempts):
            with contextlib.suppress(VastError):
                self.destroy(instance_id)
            time.sleep(pause)
            try:
                if self.instance(instance_id) is None:
                    return True
            except VastError:
                # Unreachable is not gone. Try again rather than report a
                # destruction that was never confirmed.
                continue
        return False


def arm_hard_stop(instance_id: int, deadline: float) -> int:
    """Spawn a detached watchdog that destroys ``instance_id`` at ``deadline``.

    Detached on purpose: the whole point is that it survives the session, the
    terminal and the agent that started it. Returns the watchdog's pid.

    ``deadline`` is an absolute unix timestamp, not a duration, and that is the
    W22 defect. The watchdog used to be handed a number of hours and spend them
    in a single ``time.sleep``. A ``sleep`` does not advance while the machine
    is suspended, so every minute the laptop spent asleep was a minute the
    rented instance kept billing unwatched -- which is how W2 ran 1.97 h against
    a 1.5 h cap. Against a wall-clock timestamp, suspending the machine can
    delay the kill by one poll interval and no more.
    """
    command = [
        sys.executable,
        "-m",
        "reverberate.gpu.vast",
        "hard-stop",
        str(instance_id),
        str(deadline),
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
        pid = arm_hard_stop(instance_id, deadline)
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


def account_identity(client: VastClient) -> Path:
    """The local private key whose public half Vast will install, or refuse.

    Checked before renting. The alternative is what it cost to learn: an
    instance comes up, ssh answers ``Permission denied (publickey)`` on every
    poll for the full timeout, and the run tears down having done nothing.
    """
    registered = {
        (key.get("public_key") or "").split()[1]
        for key in client.request("GET", "/ssh/")
        if len((key.get("public_key") or "").split()) > 1
    }
    if not registered:
        raise VastError("the Vast account has no ssh key registered; add one in the console")
    for public in sorted(Path.home().joinpath(".ssh").glob("*.pub")):
        blob = public.read_text().split()
        if len(blob) > 1 and blob[1] in registered:
            private = public.with_suffix("")
            if private.is_file():
                return private
    raise VastError(
        "no private key here matches a key registered on the Vast account, so ssh into "
        "the instance would be refused; nothing was rented"
    )


def wait_for_ssh(
    client: VastClient, instance_id: int, identity: Path, timeout: float = 900.0
) -> Machine:
    """Block until the instance answers a command, not merely until it exists.

    **Readiness is ssh answering.** The API reports ``running`` one second after
    create and then reverts to ``loading``, which cost three premature
    teardowns before it was believed; it supplies the host and the port and
    nothing else.
    """
    from reverberate.wave.remote import Machine, _run

    deadline = time.time() + timeout
    while time.time() < deadline:
        instance = client.instance(instance_id)
        if instance is None:
            raise VastError(f"instance {instance_id} vanished while starting")
        if instance.ssh_host and instance.status == "running":
            # Commands go through the proxy, the way every run so far has gone; the
            # instance's own address rides along for the transfers to try first.
            machine = Machine(
                host=instance.ssh_host,
                port=instance.ssh_port,
                identity=identity,
                direct=instance.direct,
            )
            try:
                _run(machine.ssh_command("true"), what="ssh probe", timeout=30)
                return machine
            except Exception:  # noqa: BLE001 - not up yet is the common case
                pass
        time.sleep(15)
    raise TimeoutError(f"instance {instance_id} never answered on ssh")


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


#: How often the watchdog wakes to compare the clock against its deadline.
#: Short enough that a suspend-and-resume cannot overshoot by much, long enough
#: that a watchdog living for hours costs nothing.
HARD_STOP_POLL_S = 30.0


def _hard_stop(instance_id: int, deadline: float, poll_s: float = HARD_STOP_POLL_S) -> int:
    """Watchdog body: wait for the wall clock to pass ``deadline``, then kill.

    The wait is a poll against :func:`time.time` rather than one long sleep,
    so that suspending the machine cannot postpone the deadline. See
    :func:`arm_hard_stop`.
    """
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(poll_s, remaining))
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


# --------------------------------------------------------------------------
# renting one machine that answers
# --------------------------------------------------------------------------


def enough_credit(credit: float, *, remaining_usd: float) -> bool:
    """Whether the account can pay the rest of the plan, not just one hour.

    The night of 2026-09-13 started with 12.5 USD, spent 10 before the encode
    and ran out under it. ``credit`` is NaN when the API did not answer; then
    the rental goes ahead, since a status read must not end a campaign.
    """
    return credit != credit or credit >= remaining_usd


def credit_of(client: Any) -> float:
    """The account's credit in USD, or NaN when the API does not answer."""
    try:
        return float(client.request("GET", "/users/current/").get("credit") or 0.0)
    except Exception:  # noqa: BLE001 - a status read must never end a campaign
        return float("nan")


#: Hosts known bad, by machine id: when it was seen and what it did. Never rented, whatever
#: ``--avoid`` says; a host is added here when its fault is the host's own and lasting.
KNOWN_BAD_HOSTS: dict[int, tuple[str, str]] = {
    35928: (
        "2026-10-05",
        "ssh3.vast.ai port 13832 refused every connection after the instance answered"
        " once: the rental was lost in its provisioning",
    ),
    152135: (
        "2026-10-05",
        "CUDA cannot be initialised (cuInit 999), nvidia-smi lists the cards",
    ),
}


def hosts_to_avoid(avoid: Iterable[int] = ()) -> set[int]:
    """The ids never to rent: those given and :data:`KNOWN_BAD_HOSTS`."""
    return {int(i) for i in avoid} | set(KNOWN_BAD_HOSTS)


def offer_ids(offer: Any) -> set[int]:
    """What names an offer in a list of hosts to avoid: its own id and its host's."""
    machine = int(getattr(offer, "machine_id", 0) or 0)
    return {int(offer.id), machine} if machine else {int(offer.id)}


def rent_one(
    client: Any,
    identity: Any,
    offers: list[Any],
    *,
    hours: float,
    disk_gb: int,
    image: str,
    remaining_usd: float = 0.0,
    avoid: set[int] | None = None,
    refuse: Callable[[Any], str | None] | None = None,
    say: Callable[[str], None] = print,
) -> tuple[Any, int]:
    """Down the list until one rents, answers and is accepted; the machine and its id.

    Every offer tried is removed from ``offers`` in place, rented or not, so a
    caller renting several boxes from one list never returns to a host that
    stayed silent (two did on 2026-09-12). The credit is read before each
    rental against ``remaining_usd``.

    ``avoid`` holds offer and machine ids never to rent; an offer named there
    is dropped without counting as a try. ``refuse`` looks at a machine that
    answered and returns why it is not to be used, or ``None``: a refused host
    is destroyed, verified destroyed, and the next offer is tried. A host that
    stayed silent or was refused is added to ``avoid``, under both its ids, so
    neither another of its offers nor a later rental of the same run takes it.
    """
    avoid = set() if avoid is None else avoid
    tried = 0
    while offers and tried < 8:
        candidate = offers.pop(0)
        if offer_ids(candidate) & avoid:
            say(f"  {candidate.id} is on a host to avoid; skipped")
            continue
        tried += 1
        credit = credit_of(client)
        if not enough_credit(credit, remaining_usd=max(remaining_usd, candidate.dph_total + 0.5)):
            raise SystemExit(
                f"credit {credit:.2f} USD is under the {remaining_usd:.2f} USD"
                " the rest of the plan needs"
            )
        say(f"  credit {credit:.2f} USD before renting {candidate.id}")
        try:
            rental = rent(client, candidate, hours=hours, image=image, disk_gb=disk_gb)
        except VastError as refusal:
            say(f"  {candidate.id} would not rent ({refusal})")
            continue
        say(f"rented {rental.instance_id} on {candidate.describe()}")
        try:
            machine = wait_for_ssh(client, rental.instance_id, identity, timeout=420.0)
        except (TimeoutError, VastError) as silence:
            say(f"  {rental.instance_id} never answered ({silence}); destroying, next")
            avoid |= offer_ids(candidate)
            teardown(client, rental.instance_id)
            continue
        reason = refuse(machine) if refuse is not None else None
        if reason is not None:
            avoid |= offer_ids(candidate)
            gone = teardown(client, rental.instance_id)
            say(f"  {rental.instance_id} refused ({reason}); destroyed={gone}")
            if not gone:
                # A second rental beside one that may still be billing is how
                # a night costs double; the watchdog holds the first meanwhile.
                raise SystemExit(
                    f"instance {rental.instance_id} was refused and is not verified destroyed"
                )
            continue
        return machine, rental.instance_id
    raise SystemExit("no offer produced a machine that answered")
