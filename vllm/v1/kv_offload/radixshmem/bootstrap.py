# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle of the node-wide radixshmem index and slot store.

* Owner worker (DP 0 / TP 0): in ``get_worker`` starts an in-process
  ``RadixServer`` (unless ``endpoint`` names an external one) and writes the
  layout sentinel, or adopts a live one on an external server.
* Other workers attach read-only; schedulers attach read-write. Both wait for
  the sentinel and compare the layout.

Workers run before schedulers exist, so no worker waits for a scheduler. The
server validates block size, pools, slot bytes, window and alignment; the
sentinel carries the vLLM-only layout fields.
"""

import atexit
import contextlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from filelock import FileLock

from vllm.logger import init_logger
from vllm.v1.kv_offload.radixshmem.geometry import (
    FINGERPRINT_LAYOUT_FIELDS,
    SlotGeometry,
    server_geometry,
)

logger = init_logger(__name__)

SENTINEL_VERSION = 2
DEFAULT_INDEX_SHM_NAME = "/vllm_kv_index"
DEFAULT_ATTACH_TIMEOUT_S = 300.0
_POLL_S = 0.05
_LOG_EVERY_S = 30.0

_OWNER_DESCRIPTION = "the pool owner (the DP rank 0 / TP rank 0 worker)"


def _import_radixshmem() -> Any:
    try:
        import radixshmem
    except ImportError as e:
        raise RuntimeError(
            "RadixShmem needs the `radixshmem` package (ai-dynamo/radixshmem)."
        ) from e
    if not hasattr(radixshmem, "_data"):
        raise RuntimeError("`radixshmem` was built without its `_data` extension.")
    return radixshmem


def component_of(rs: Any, kind: str) -> Any:
    """``radixshmem.ComponentType`` of a pool kind."""
    return getattr(rs.ComponentType, kind.upper())


def mask_of(rs: Any, kind: str) -> int:
    """The ``COMPONENT_MASK_*`` bit of a pool kind."""
    return int(getattr(rs, f"COMPONENT_MASK_{kind.upper()}"))


@dataclass(frozen=True)
class RadixShmemSettings:
    """Backend knobs read from ``kv_connector_extra_config``."""

    index_shm_name: str = DEFAULT_INDEX_SHM_NAME
    data_shm_name: str = ""
    # External radix-server; empty: the owner worker runs one.
    endpoint: str = ""
    remote: bool = False
    cache_namespace: str = ""
    remote_timeout_s: float = 30.0
    remote_max_pending: int = 128
    hugepage_path: str = ""
    prefault: bool = True
    max_nodes: int = 0
    data_pool_ratio: float = 4.0
    background_evict_ratio: float = 0.05
    rpc_workers: int = 4
    attach_timeout_s: float = DEFAULT_ATTACH_TIMEOUT_S

    @classmethod
    def from_extra_config(cls, extra: Mapping[str, Any]) -> "RadixShmemSettings":
        name = str(extra.get("index_shm_name", DEFAULT_INDEX_SHM_NAME))
        if not name.startswith("/"):
            raise ValueError(f"index_shm_name must start with '/', got {name!r}")
        settings = cls(
            index_shm_name=name,
            data_shm_name=str(extra.get("data_shm_name", "") or f"{name}_data"),
            endpoint=str(extra.get("endpoint", "") or ""),
            remote=bool(extra.get("remote", False)),
            cache_namespace=str(extra.get("cache_namespace", "") or ""),
            remote_timeout_s=float(extra.get("remote_timeout_s", 30.0)),
            remote_max_pending=int(extra.get("remote_max_pending", 128)),
            hugepage_path=str(extra.get("hugepage_path", "") or ""),
            prefault=bool(extra.get("prefault", True)),
            max_nodes=int(extra.get("max_nodes", 0)),
            data_pool_ratio=float(extra.get("data_pool_ratio", 4.0)),
            background_evict_ratio=float(extra.get("background_evict_ratio", 0.05)),
            rpc_workers=int(extra.get("rpc_workers", 4)),
            attach_timeout_s=float(
                extra.get("attach_timeout_s", DEFAULT_ATTACH_TIMEOUT_S)
            ),
        )
        if settings.remote:
            if not settings.endpoint or not settings.cache_namespace:
                raise ValueError(
                    "RadixShmem remote sharing requires a node-local external "
                    "`endpoint` and a `cache_namespace` identifying the model "
                    "weights/revision and adapters."
                )
            if settings.remote_timeout_s <= 0 or settings.remote_max_pending <= 0:
                raise ValueError("remote_timeout_s and remote_max_pending must be > 0")
        return settings

    @property
    def external_server(self) -> bool:
        return bool(self.endpoint)

    @property
    def sentinel_path(self) -> str:
        base = self.index_shm_name.lstrip("/").replace("/", "_")
        return f"/dev/shm/{base}.vllm.json"

    def resolved_endpoint(self) -> str:
        if self.endpoint:
            return self.endpoint
        from radixshmem import rpc

        return rpc.endpoint_for(self.index_shm_name)


def _unlink_quiet(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _pid_alive(pid: int) -> bool:
    return pid > 0 and os.path.exists(f"/proc/{pid}")


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_sentinel(path: str) -> dict[str, Any] | None:
    """The sentinel payload, or None when absent, unreadable or stale."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    if payload.get("version") != SENTINEL_VERSION:
        return None
    if not _pid_alive(int(payload.get("owner_pid", 0))):
        # Leftover from a crashed run; the new owner rewrites it.
        return None
    return payload


def _wait_for_sentinel(path: str, deadline: float, role: str) -> dict[str, Any]:
    next_log = time.monotonic() + _LOG_EVERY_S
    while True:
        payload = read_sentinel(path)
        if payload is not None:
            return payload
        now = time.monotonic()
        if now > deadline:
            raise TimeoutError(
                f"RadixShmem: {role} timed out waiting for {path}; check the "
                "DP 0 / TP 0 worker's logs."
            )
        if now > next_log:
            next_log = now + _LOG_EVERY_S
            logger.info("RadixShmem %s still waiting for %s", role, path)
        time.sleep(_POLL_S)


def compare_fingerprint(
    local: Mapping[str, Any],
    published: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    what: str,
) -> None:
    """Raise ``RuntimeError`` naming every field on which the two differ."""
    diffs = [
        f"{f}: local={local.get(f)!r} {what}={published.get(f)!r}"
        for f in fields
        if local.get(f) != published.get(f)
    ]
    if not diffs:
        return
    raise RuntimeError(
        f"RadixShmem layout differs from {what}:\n  " + "\n  ".join(diffs)
    )


def _connect(
    rs: Any,
    settings: "RadixShmemSettings",
    geo: Any,
    endpoint: str,
    deadline: float,
    *,
    read_only: bool,
) -> Any:
    """A client, retried while the server socket is not served yet."""
    next_log = time.monotonic() + _LOG_EVERY_S
    last: Exception | None = None
    while True:
        try:
            return rs.RadixClient(
                settings.index_shm_name,
                geo,
                endpoint=endpoint,
                local_read_only=read_only,
            )
        except RuntimeError as e:
            # Only "UNAVAILABLE" means no server yet.
            if not str(e).startswith("UNAVAILABLE"):
                raise
            last = e
        now = time.monotonic()
        if now > deadline:
            raise TimeoutError(
                f"RadixShmem: timed out connecting to {endpoint}; is the DP 0 / "
                "TP 0 worker (or the external radix-server) running?"
            ) from last
        if now > next_log:
            next_log = now + _LOG_EVERY_S
            logger.info("RadixShmem still connecting to %s", endpoint)
        time.sleep(_POLL_S)


def _check_published_geometry(info: Any, geometry: SlotGeometry) -> dict[str, int]:
    """Cross-check the server's geometry; return the planned slots per pool."""
    if not info.data_plane:
        raise RuntimeError("RadixShmem: the server has no SlotStore (--data-bytes)")
    g = info.geometry
    published = g["pools"]
    if set(published) != set(geometry.kinds):
        raise RuntimeError(
            f"RadixShmem: server pools {sorted(published)} != local pools "
            f"{sorted(geometry.kinds)}"
        )
    expected = {
        "block_size": geometry.tokens_per_chunk,
        "slot_align": geometry.slot_align,
    }
    compare_fingerprint(
        expected,
        {key: int(g[key]) for key in expected},
        tuple(expected),
        what="the server",
    )
    pool_slots: dict[str, int] = {}
    for pool in geometry.pools:
        pub = published[pool.kind]
        expected = {"slot_bytes": pool.slot_bytes}
        if pool.kind == "swa":
            expected["window_blocks"] = pool.window_blocks
        compare_fingerprint(
            expected,
            {key: int(pub.get(key, 0)) for key in expected},
            tuple(expected),
            what=f"the server's {pool.kind} pool",
        )
        num_slots = int(pub["num_slots"])
        if num_slots <= 0:
            raise RuntimeError(f"RadixShmem: the server planned zero {pool.kind} slots")
        pool_slots[pool.kind] = num_slots
    return pool_slots


def _check_store(
    rs: Any, store: Any, geometry: SlotGeometry, pool_slots: Mapping[str, int]
) -> None:
    data_bytes = len(store.data_view())
    for pool in geometry.pools:
        component = component_of(rs, pool.kind)
        if not store.has_pool(component):
            raise RuntimeError(f"RadixShmem: SlotStore has no {pool.kind} pool")
        sp = store.pool(component)
        # The package checks slot_bytes >=; the strided views need ==.
        compare_fingerprint(
            {"num_slots": pool_slots[pool.kind], "slot_bytes": pool.slot_stride},
            {"num_slots": int(sp.num_slots), "slot_bytes": int(sp.slot_bytes)},
            ("num_slots", "slot_bytes"),
            what=f"SlotStore's {pool.kind} pool",
        )
        if pool.kind == "full" and int(sp.base) != 0:
            raise RuntimeError(
                "RadixShmem: the FULL pool is not at offset 0 of the data region "
                f"(offset {sp.base})"
            )
        end = int(sp.base) + int(sp.num_slots) * int(sp.slot_bytes)
        if end > data_bytes:
            raise RuntimeError(
                f"RadixShmem: the {pool.kind} pool ends at {end} B, past the "
                f"{data_bytes} B data region"
            )


def describe_pools(geometry: SlotGeometry, pool_slots: Mapping[str, int]) -> str:
    parts = []
    for pool in geometry.pools:
        n = pool_slots[pool.kind]
        s = (
            f"{pool.kind.upper()} {n} slots x {pool.slot_stride} B "
            f"= {n * pool.slot_stride / 2**30:.2f} GiB"
        )
        if pool.kind == "swa":
            s += f" (window {pool.window_blocks} blocks)"
        parts.append(s)
    return ", ".join(parts)


def _layout_only(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    return {f: fingerprint.get(f) for f in FINGERPRINT_LAYOUT_FIELDS}


@dataclass
class RadixRegions:
    """An owned or attached radixshmem index and slot store."""

    client: Any
    store: Any
    pool_slots: dict[str, int]
    geometry: SlotGeometry
    settings: RadixShmemSettings
    server: Any = None
    owns_sentinel: bool = False
    _closed: bool = field(default=False, init=False)

    @property
    def is_owner(self) -> bool:
        """This process runs the server and destroys the pool on close."""
        return self.server is not None

    @property
    def num_slots(self) -> int:
        """Slots of the FULL pool."""
        return self.pool_slots["full"]

    @classmethod
    def open_worker(
        cls,
        geometry: SlotGeometry,
        settings: RadixShmemSettings,
        fingerprint: Mapping[str, Any],
        *,
        owner: bool,
    ) -> "RadixRegions":
        """The owner starts or configures the server; other workers attach."""
        lock = (
            FileLock(
                settings.sentinel_path + ".lock", timeout=settings.attach_timeout_s
            )
            if settings.remote and owner
            else contextlib.nullcontext()
        )
        with lock:
            return cls._open(
                geometry, settings, fingerprint, read_only=True, owner=owner
            )

    @classmethod
    def open_scheduler(
        cls,
        geometry: SlotGeometry,
        settings: RadixShmemSettings,
        fingerprint: Mapping[str, Any],
    ) -> "RadixRegions":
        """Attach read-write and compare the layout."""
        return cls._open(geometry, settings, fingerprint, read_only=False)

    @classmethod
    def _open(
        cls,
        geometry: SlotGeometry,
        settings: RadixShmemSettings,
        fingerprint: Mapping[str, Any],
        *,
        read_only: bool,
        owner: bool = False,
    ) -> "RadixRegions":
        rs = _import_radixshmem()
        layout = _layout_only(fingerprint)
        deadline = time.monotonic() + settings.attach_timeout_s
        role = "worker" if read_only else "DP scheduler"
        sentinel = settings.sentinel_path
        endpoint = settings.resolved_endpoint()
        regions = cls(
            client=None,
            store=None,
            pool_slots={},
            geometry=geometry,
            settings=settings,
        )
        try:
            writes_sentinel = owner and read_sentinel(sentinel) is None
            if owner and not settings.external_server:
                # Starting a server would unlink the socket of a live owner.
                existing = read_sentinel(sentinel)
                if existing is not None:
                    raise RuntimeError(
                        f"RadixShmem: pid {existing['owner_pid']} already owns "
                        f"{settings.index_shm_name}; set a distinct "
                        "`index_shm_name` or stop it."
                    )
                _unlink_quiet(sentinel)
                regions.server = _start_server(rs, geometry, settings)
                endpoint = regions.server.endpoint
                writes_sentinel = True
            geo = server_geometry(rs, geometry) if owner or not read_only else None
            regions.client = _connect(
                rs, settings, geo, endpoint, deadline, read_only=read_only
            )
            info = regions.client.wait_ready(max(deadline - time.monotonic(), 1.0))
            regions.pool_slots = pool_slots = _check_published_geometry(info, geometry)
            if writes_sentinel:
                _write_json_atomic(
                    sentinel,
                    {
                        "version": SENTINEL_VERSION,
                        "owner_pid": os.getpid(),
                        "num_slots": pool_slots["full"],
                        "pool_slots": pool_slots,
                        **layout,
                    },
                )
                regions.owns_sentinel = True
            else:
                published = _wait_for_sentinel(sentinel, deadline, role)
                compare_fingerprint(
                    {**layout, "pool_slots": pool_slots},
                    published,
                    FINGERPRINT_LAYOUT_FIELDS + ("pool_slots",),
                    what=_OWNER_DESCRIPTION,
                )
            regions.store = regions.client.store
            _check_store(rs, regions.store, geometry, pool_slots)
        except BaseException:
            regions.close()
            raise
        if regions.server is not None or regions.owns_sentinel:
            atexit.register(regions.close)
        logger.info(
            "RadixShmem %s attached: index=%s data=%s endpoint=%s (%s)",
            role,
            settings.index_shm_name,
            settings.data_shm_name,
            endpoint,
            describe_pools(geometry, pool_slots),
        )
        return regions

    def close(self) -> None:
        """Close the client, then (owner) the server, which destroys the pool,
        then (writer) the sentinel. Other processes' mappings stay valid."""
        if self._closed:
            return
        self._closed = True
        self.store = None
        client, self.client = self.client, None
        server, self.server = self.server, None
        try:
            if client is not None:
                client.close()
        finally:
            try:
                if server is not None:
                    server.close()
            finally:
                if self.owns_sentinel:
                    _unlink_quiet(self.settings.sentinel_path)


def _start_server(rs: Any, geometry: SlotGeometry, settings: RadixShmemSettings) -> Any:
    ratios = {f"{p.kind}_ratio": p.ratio for p in geometry.pools[1:]}
    cfg = rs.ServerConfig(
        name=settings.index_shm_name,
        data_name=settings.data_shm_name,
        data_bytes=geometry.cpu_bytes_to_use,
        hugepage_path=settings.hugepage_path,
        prefault=settings.prefault,
        slot_align=geometry.slot_align,
        max_nodes=settings.max_nodes,
        data_pool_ratio=settings.data_pool_ratio,
        background_evict_ratio=settings.background_evict_ratio,
        rpc_workers=settings.rpc_workers,
        **ratios,
    )
    try:
        return rs.RadixServer(cfg).start()
    except RuntimeError as e:
        if "cannot bind" not in str(e):
            raise
        raise RuntimeError(
            f"RadixShmem: cannot bind {cfg.resolved_endpoint}; another server uses "
            "it. Set a distinct `index_shm_name`, or share one `endpoint`."
        ) from e
