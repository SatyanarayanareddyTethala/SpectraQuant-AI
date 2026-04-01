"""Locking abstractions for orchestration critical sections."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LockLease:
    key: str
    token: str


class LockManager(Protocol):
    """Interface for lock managers (in-memory, Redis, Postgres, etc.)."""

    def try_acquire(self, key: str) -> LockLease | None: ...

    def release(self, lease: LockLease) -> None: ...


class InMemoryLockManager:
    """Process-local lock manager implementing the distributed-ready contract."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._leases: dict[str, str] = {}

    def try_acquire(self, key: str) -> LockLease | None:
        with self._guard:
            if key in self._leases:
                return None
            token = uuid.uuid4().hex
            self._leases[key] = token
            return LockLease(key=key, token=token)

    def release(self, lease: LockLease) -> None:
        with self._guard:
            token = self._leases.get(lease.key)
            if token == lease.token:
                self._leases.pop(lease.key, None)
