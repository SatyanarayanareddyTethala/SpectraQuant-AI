"""Allocator registry for SpectraQuant-AI-V3.

Maps allocator name strings (used in :class:`~spectraquant_v3.strategies.strategy_definition.StrategyDefinition`)
to the concrete allocator class.

Usage::

    from spectraquant_v3.strategies.allocators.registry import AllocatorRegistry

    cls = AllocatorRegistry.get("vol_target_v1")
    allocator_instance = cls.from_config(cfg)

    AllocatorRegistry.list()  # ['equal_weight', 'vol_target_v1']
"""

from __future__ import annotations

from typing import Any


class AllocatorRegistry:
    """Registry mapping allocator name → allocator class.

    Methods
    -------
    register(name, cls)  Add or replace an allocator class.
    get(name)            Return the allocator class for *name*.
    list()               Sorted list of registered allocator names.
    """

    _allocators: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, allocator_cls: type) -> None:
        """Register an allocator class under *name*."""
        if not name:
            raise ValueError("Allocator name must be a non-empty string.")
        cls._allocators[name] = allocator_cls

    @classmethod
    def get(cls, name: str) -> type:
        """Return the allocator class for *name*.

        Raises:
            KeyError: If no allocator with that name has been registered.
        """
        try:
            return cls._allocators[name]
        except KeyError:
            registered = sorted(cls._allocators)
            raise KeyError(
                f"Allocator '{name}' is not registered. "
                f"Registered allocators: {registered}"
            ) from None

    @classmethod
    def list(cls) -> list[str]:
        """Return a sorted list of registered allocator names."""
        return sorted(cls._allocators)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove an allocator from the registry (useful in tests)."""
        cls._allocators.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Wipe the registry (useful in tests)."""
        cls._allocators.clear()

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> Any:
        """Instantiate the allocator class for *name* with the given keyword arguments."""
        allocator_cls = cls.get(name)
        return allocator_cls(**kwargs)


# ---------------------------------------------------------------------------
# Built-in allocator registrations
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from spectraquant_v3.pipeline.allocator import Allocator
    from spectraquant_v3.strategies.allocators.rank_vol_target_allocator import (
        RankVolTargetAllocator,
    )

    # The Allocator class supports both "equal_weight" and "vol_target"
    # modes via its AllocatorConfig.mode field.  We register the same
    # class under two canonical names for ergonomic strategy definitions.
    AllocatorRegistry.register("equal_weight", Allocator)
    AllocatorRegistry.register("vol_target_v1", Allocator)
    AllocatorRegistry.register("rank_vol_target_allocator", RankVolTargetAllocator)


_register_builtins()
