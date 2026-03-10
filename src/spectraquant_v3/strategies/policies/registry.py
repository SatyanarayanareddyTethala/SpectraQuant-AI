"""Policy registry for SpectraQuant-AI-V3.

Maps policy name strings (used in :class:`~spectraquant_v3.strategies.strategy_definition.StrategyDefinition`)
to the concrete policy classes.

Usage::

    from spectraquant_v3.strategies.policies.registry import PolicyRegistry

    cls = PolicyRegistry.get("confidence_filter_v1")
    policy_instance = cls.from_config(cfg)

    PolicyRegistry.list()  # ['confidence_filter_v1']
"""

from __future__ import annotations

from typing import Any


class PolicyRegistry:
    """Registry mapping policy name → policy class.

    Methods
    -------
    register(name, cls)  Add or replace a policy class.
    get(name)            Return the policy class for *name*.
    list()               Sorted list of registered policy names.
    """

    _policies: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, policy_cls: type) -> None:
        """Register a policy class under *name*."""
        if not name:
            raise ValueError("Policy name must be a non-empty string.")
        cls._policies[name] = policy_cls

    @classmethod
    def get(cls, name: str) -> type:
        """Return the policy class for *name*.

        Raises:
            KeyError: If no policy with that name has been registered.
        """
        try:
            return cls._policies[name]
        except KeyError:
            registered = sorted(cls._policies)
            raise KeyError(
                f"Policy '{name}' is not registered. "
                f"Registered policies: {registered}"
            ) from None

    @classmethod
    def list(cls) -> list[str]:
        """Return a sorted list of registered policy names."""
        return sorted(cls._policies)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a policy from the registry (useful in tests)."""
        cls._policies.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Wipe the registry (useful in tests)."""
        cls._policies.clear()

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> Any:
        """Instantiate the policy class for *name* with the given keyword arguments."""
        policy_cls = cls.get(name)
        return policy_cls(**kwargs)


# ---------------------------------------------------------------------------
# Built-in policy registrations
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from spectraquant_v3.pipeline.meta_policy import MetaPolicy

    PolicyRegistry.register("confidence_filter_v1", MetaPolicy)


_register_builtins()
