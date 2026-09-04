"""The object store, and the one place that knows a bucket exists.

Roadmap section 12.2 puts every artefact under one data root, and W24 records
that the voxelisation cache "currently lives per worktree with no sharing
mechanism". That is the hole this module fills. A voxelisation costs minutes of
CPU and about a gigabyte per room per band; a rented GPU instance has no way to
reach the laptop that computed one, so B0 paid for 52.3 minutes of voxelisation
on a machine rented for its card. **The remote store is what makes a rented
instance a first class client of the cache rather than a special case.**

**The remote is the source of truth and the local disk is a read-through
cache.** That ordering is deliberate: the alternative, a local truth mirrored
outwards, gives a rented instance nothing to read until somebody remembers to
push.

**Content addressing, and why a staging prefix.** Keys are derived from the
SHA-256 of what they hold, so an upload is idempotent and a re-run overwrites
nothing. An interrupted multi-gigabyte upload would otherwise leave a truncated
object under a key that ``exists()`` reports as present, and the cache would
serve it forever. So an object is written under ``staging/`` first and copied
to its real key only once its size is confirmed, and its digest travels in
object metadata so a reader can check what it got.

**Bucket ``Clarify`` is shared with another project of the owner's**, which is
why :data:`PREFIX` is not optional and every key this project writes sits under
it. The same credentials also read that project's speech library, which is what
:class:`ObjectStore` grows a ranged read for: its clips live at byte offsets
inside multi hundred megabyte zip shards, and fetching one clip must not
download the shard.

Credentials come from the environment, populated by
:mod:`reverberate.auth` from KeePassXC, and from nowhere else. No value from
this module is ever logged.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "BUCKET_ENV",
    "ENDPOINT_ENV",
    "KEY_ID_ENV",
    "PREFIX",
    "SECRET_ENV",
    "B2Store",
    "MemoryStore",
    "ObjectStore",
    "StoreError",
    "digest_of_bytes",
    "digest_of_file",
    "shared_store",
]

#: Environment variables carrying the credentials. Read here, never elsewhere.
#: These are variable *names*; no credential is stored in this repository.
KEY_ID_ENV = "B2_KEY_ID"
SECRET_ENV = "B2_APPLICATION_KEY"  # gitleaks:allow
BUCKET_ENV = "B2_BUCKET_NAME"
ENDPOINT_ENV = "B2_S3_ENDPOINT_URL"

#: Every key this project writes lives under this prefix, because the bucket is
#: shared. Not a parameter: a caller that could omit it eventually would.
PREFIX = "reverberate/"

#: Where a partially uploaded object waits until its size is confirmed.
STAGING = f"{PREFIX}staging/"

#: Object metadata key carrying the SHA-256 a reader checks against.
DIGEST_META = "sha256"

_CHUNK = 1 << 20


class StoreError(RuntimeError):
    """A store operation failed, or returned something other than was asked for."""


def digest_of_bytes(payload: bytes) -> str:
    """Lowercase hex SHA-256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def digest_of_file(path: Path) -> str:
    """Lowercase hex SHA-256 of a file, read in chunks so size does not matter."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class ObjectStore(Protocol):
    """What the rest of the project is allowed to assume about the store.

    Keys are relative to :data:`PREFIX` for this project's own objects, and
    absolute within the bucket when ``shared=True``, which is how the other
    project's speech library is read without giving this project write access
    to it.
    """

    def exists(self, key: str, *, shared: bool = False) -> bool: ...

    def put_bytes(self, key: str, payload: bytes) -> str: ...

    def get_bytes(self, key: str, *, shared: bool = False) -> bytes: ...

    def get_range(self, key: str, offset: int, length: int, *, shared: bool = False) -> bytes: ...

    def put_file(self, key: str, path: Path) -> str: ...

    def get_file(self, key: str, destination: Path, *, shared: bool = False) -> Path: ...

    def list(self, prefix: str, *, shared: bool = False) -> Iterator[str]: ...


@dataclass
class MemoryStore:
    """An in-memory store, so the suite never touches the network.

    Roadmap constraint 2. This is the same shape as the deterministic solver
    fake: the tests exercise the real call sequence against a fake transport
    rather than mocking out the module under test.
    """

    objects: dict[str, bytes] = field(default_factory=dict)
    #: Every key ever written, including staging keys later removed. The test
    #: for the staging discipline reads this rather than the live keys.
    written: list[str] = field(default_factory=list)

    def _key(self, key: str, shared: bool) -> str:
        return key if shared else f"{PREFIX}{key}"

    def exists(self, key: str, *, shared: bool = False) -> bool:
        return self._key(key, shared) in self.objects

    def put_bytes(self, key: str, payload: bytes) -> str:
        digest = digest_of_bytes(payload)
        staging = f"{STAGING}{digest}"
        self.objects[staging] = payload
        self.written.append(staging)
        if len(self.objects[staging]) != len(payload):  # pragma: no cover - cannot happen here
            raise StoreError(f"short upload for {key!r}")
        self.objects[self._key(key, False)] = payload
        self.written.append(self._key(key, False))
        del self.objects[staging]
        return digest

    def get_bytes(self, key: str, *, shared: bool = False) -> bytes:
        try:
            return self.objects[self._key(key, shared)]
        except KeyError:
            raise StoreError(f"no object at {self._key(key, shared)!r}") from None

    def get_range(self, key: str, offset: int, length: int, *, shared: bool = False) -> bytes:
        payload = self.get_bytes(key, shared=shared)
        chunk = payload[offset : offset + length]
        if len(chunk) != length:
            raise StoreError(f"range {offset}+{length} runs past the end of {key!r}")
        return chunk

    def put_file(self, key: str, path: Path) -> str:
        return self.put_bytes(key, path.read_bytes())

    def get_file(self, key: str, destination: Path, *, shared: bool = False) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.get_bytes(key, shared=shared))
        return destination

    def list(self, prefix: str, *, shared: bool = False) -> Iterator[str]:
        full = self._key(prefix, shared)
        for key in sorted(self.objects):
            if key.startswith(full) and not key.startswith(STAGING):
                yield key if shared else key[len(PREFIX) :]


class B2Store:
    """Backblaze B2 through its S3 compatible API.

    **The S3 API does work with these credentials**, contrary to a note left in
    the sibling project that says only the native b2 backend does. That note
    predates the current application key; a listing was run against the live
    bucket before this module was written. Recorded here so the next reader
    does not reach for ``b2sdk`` on the strength of it.
    """

    def __init__(self, client: Any | None = None, bucket: str | None = None) -> None:
        self._client = client if client is not None else _make_client()
        self._bucket = bucket if bucket is not None else _require(BUCKET_ENV)

    def _key(self, key: str, shared: bool) -> str:
        return key if shared else f"{PREFIX}{key}"

    def exists(self, key: str, *, shared: bool = False) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key(key, shared))
        except Exception as error:  # noqa: BLE001 - botocore raises a generated class
            if _is_missing(error):
                return False
            raise
        return True

    def put_bytes(self, key: str, payload: bytes) -> str:
        digest = digest_of_bytes(payload)
        staging = f"{STAGING}{digest}"
        self._client.put_object(
            Bucket=self._bucket,
            Key=staging,
            Body=payload,
            Metadata={DIGEST_META: digest},
        )
        self._promote(staging, self._key(key, False), len(payload), digest)
        return digest

    def put_file(self, key: str, path: Path) -> str:
        digest = digest_of_file(path)
        staging = f"{STAGING}{digest}"
        size = path.stat().st_size
        with path.open("rb") as handle:
            self._client.upload_fileobj(
                handle,
                self._bucket,
                staging,
                ExtraArgs={"Metadata": {DIGEST_META: digest}},
            )
        self._promote(staging, self._key(key, False), size, digest)
        return digest

    def _promote(self, staging: str, key: str, size: int, digest: str) -> None:
        """Copy a staged object to its real key once its size is confirmed.

        The size check is what makes a truncated upload visible. A full digest
        check would mean downloading a gigabyte back to verify a gigabyte just
        sent, so the digest travels in metadata instead and is checked by
        whoever reads the object.
        """
        head = self._client.head_object(Bucket=self._bucket, Key=staging)
        if int(head["ContentLength"]) != size:
            self._client.delete_object(Bucket=self._bucket, Key=staging)
            raise StoreError(f"short upload for {key!r}: {head['ContentLength']} of {size} bytes")
        self._client.copy_object(
            Bucket=self._bucket,
            Key=key,
            CopySource={"Bucket": self._bucket, "Key": staging},
            Metadata={DIGEST_META: digest},
            MetadataDirective="REPLACE",
        )
        self._client.delete_object(Bucket=self._bucket, Key=staging)

    def get_bytes(self, key: str, *, shared: bool = False) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=self._key(key, shared))
        payload: bytes = response["Body"].read()
        _check_digest(self._key(key, shared), payload, response.get("Metadata", {}))
        return payload

    def get_range(self, key: str, offset: int, length: int, *, shared: bool = False) -> bytes:
        if length <= 0:
            raise ValueError("length must be positive")
        end = offset + length - 1
        response = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key(key, shared),
            Range=f"bytes={offset}-{end}",
        )
        chunk: bytes = response["Body"].read()
        if len(chunk) != length:
            raise StoreError(f"range {offset}+{length} returned {len(chunk)} bytes for {key!r}")
        return chunk

    def get_file(self, key: str, destination: Path, *, shared: bool = False) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        with partial.open("wb") as handle:
            self._client.download_fileobj(self._bucket, self._key(key, shared), handle)
        head = self._client.head_object(Bucket=self._bucket, Key=self._key(key, shared))
        expected = head.get("Metadata", {}).get(DIGEST_META)
        if expected is not None and digest_of_file(partial) != expected:
            partial.unlink()
            raise StoreError(f"digest mismatch on {key!r}")
        shutil.move(str(partial), str(destination))
        return destination

    def list(self, prefix: str, *, shared: bool = False) -> Iterator[str]:
        full = self._key(prefix, shared)
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": full}
            if token is not None:
                kwargs["ContinuationToken"] = token
            page = self._client.list_objects_v2(**kwargs)
            for entry in page.get("Contents", []):
                key = str(entry["Key"])
                if key.startswith(STAGING):
                    continue
                yield key if shared else key[len(PREFIX) :]
            if not page.get("IsTruncated"):
                return
            token = page.get("NextContinuationToken")


def _check_digest(key: str, payload: bytes, metadata: dict[str, str]) -> None:
    expected = metadata.get(DIGEST_META)
    if expected is not None and digest_of_bytes(payload) != expected:
        raise StoreError(f"digest mismatch on {key!r}")


def _is_missing(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status in (403, 404)


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise StoreError(
            f"required environment variable {name!r} is not set; "
            "run reverberate.auth.inject() first"
        )
    return value


def _make_client() -> Any:
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=_require(ENDPOINT_ENV),
        aws_access_key_id=_require(KEY_ID_ENV),
        aws_secret_access_key=_require(SECRET_ENV),
    )


#: Resolved once per process by :func:`shared_store`. ``None`` is a real
#: answer -- "this machine has no credentials" -- so absence is spelled with a
#: sentinel rather than with ``None``.
_SHARED_UNSET = object()
_shared: Any = _SHARED_UNSET


def shared_store() -> ObjectStore | None:
    """The project's bucket, or ``None`` when this machine cannot reach it.

    Every caller that *shares* an artefact wants the same thing: the store if
    the credentials are there, and no store rather than an exception if they
    are not. A laptop with a locked vault, a checkout without KeePassXC and CI
    all have to keep working, and each of them would otherwise grow its own
    try/except around :class:`B2Store`.

    Returning ``None`` rather than raising is what makes the store optional at
    every call site without making it invisible: a caller that gets ``None``
    is expected to say so, because a silently local-only run is the failure
    mode this project has already paid for once -- the voxelisation cache of
    W29 lived in a worktree, was never published, and went with the worktree.

    Resolved once: the vault is read at most one time per process, and the
    answer -- store or no store -- is kept for every later call.
    """
    global _shared
    if _shared is _SHARED_UNSET:
        _shared = _open_shared_store()
    return _shared  # type: ignore[no-any-return]


def _open_shared_store() -> ObjectStore | None:
    import contextlib

    from reverberate import auth

    # A vault that will not open, or is not installed, is a machine without
    # credentials -- which is exactly what this function reports by returning
    # None. It is not an error to raise through a caller that asked whether a
    # store exists.
    with contextlib.suppress(Exception):
        auth.inject([KEY_ID_ENV, SECRET_ENV, BUCKET_ENV, ENDPOINT_ENV])
    names = (KEY_ID_ENV, SECRET_ENV, BUCKET_ENV, ENDPOINT_ENV)
    if not all(os.environ.get(name) for name in names):
        return None
    try:
        return B2Store()
    except (StoreError, ImportError):
        return None
