from __future__ import annotations

from spectraquant_v3.service.locking import InMemoryLockManager


def test_in_memory_lock_contention() -> None:
    mgr = InMemoryLockManager()
    first = mgr.try_acquire("equity:research")
    assert first is not None
    second = mgr.try_acquire("equity:research")
    assert second is None

    mgr.release(first)
    third = mgr.try_acquire("equity:research")
    assert third is not None
