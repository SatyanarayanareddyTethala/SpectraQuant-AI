"""Configuration loader for SpectraQuant-AI-V3.

Loads YAML config files from config/v3/.
Supports per-asset-class overrides on top of base.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None

# Anchor: walk up from this file's directory to find a config/v3 sibling.
# This avoids fragile hard-coded `.parent` chains.
_THIS_FILE = Path(__file__).resolve()


def _find_config_dir() -> Path:
    """Resolve the config/v3 directory, checking env override first.

    Search strategy:
    1. ``SPECTRAQUANT_V3_CONFIG_DIR`` environment variable.
    2. Walk up the directory tree from this source file until a directory
       containing ``config/v3`` is found.
    3. Fall back to a ``config/v3`` sibling of the current working directory.
    """
    env_dir = os.getenv("SPECTRAQUANT_V3_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    for parent in _THIS_FILE.parents:
        candidate = parent / "config" / "v3"
        if candidate.is_dir():
            return candidate
    # Last-resort: relative to cwd
    return Path("config") / "v3"


def load_config(
    config_dir: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Load and merge YAML config from *config_dir*.

    Loading order (later keys win):
    1. ``base.yaml``   – shared defaults
    2. Asset-class-specific file is NOT merged here; callers use
       ``get_crypto_config()`` or ``get_equity_config()`` instead.

    Args:
        config_dir: Path to directory containing base.yaml.  When *None*,
            the directory is resolved via ``SPECTRAQUANT_V3_CONFIG_DIR``
            env variable or the ``config/v3/`` folder in the repo root.
        force_reload: Bypass the module-level cache.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If ``base.yaml`` cannot be found.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    base_dir = Path(config_dir) if config_dir else _find_config_dir()
    base_path = base_dir / "base.yaml"

    if not base_path.exists():
        raise FileNotFoundError(
            f"V3 base config not found at {base_path}. "
            "Set SPECTRAQUANT_V3_CONFIG_DIR or place config/v3/base.yaml in the repo root."
        )

    with open(base_path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    _CONFIG_CACHE = cfg
    return cfg


def get_crypto_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Return merged config with crypto.yaml layered on top of base.yaml."""
    base = load_config(config_dir)
    base_dir = Path(config_dir) if config_dir else _find_config_dir()
    cfg = dict(base)
    crypto_path = base_dir / "crypto.yaml"
    if crypto_path.exists():
        with open(crypto_path) as fh:
            overlay: dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, overlay)

    strategies_path = base_dir / "strategies.yaml"
    if strategies_path.exists():
        with open(strategies_path) as fh:
            strategies_overlay: dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, strategies_overlay)
    return cfg


def get_equity_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Return merged config with equities.yaml layered on top of base.yaml."""
    base = load_config(config_dir)
    base_dir = Path(config_dir) if config_dir else _find_config_dir()
    cfg = dict(base)
    equity_path = base_dir / "equities.yaml"
    if equity_path.exists():
        with open(equity_path) as fh:
            overlay: dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, overlay)

    strategies_path = base_dir / "strategies.yaml"
    if strategies_path.exists():
        with open(strategies_path) as fh:
            strategies_overlay: dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, strategies_overlay)
    return cfg


def reset_config_cache() -> None:
    """Reset module-level config cache.  Intended for use in tests only."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def get_run_mode() -> "RunMode":
    """Return :class:`RunMode` from env var ``SPECTRAQUANT_RUN_MODE``.

    Defaults to :attr:`RunMode.NORMAL`.

    Raises:
        InvalidRunModeError: If the env value is not a valid run mode.
    """
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import InvalidRunModeError

    raw = os.getenv("SPECTRAQUANT_RUN_MODE", "normal").lower()
    try:
        return RunMode(raw)
    except ValueError:
        valid = [m.value for m in RunMode]
        raise InvalidRunModeError(
            f"Invalid SPECTRAQUANT_RUN_MODE={raw!r}. Valid values: {valid}"
        ) from None


def get_run_mode_from_config(cfg: dict[str, Any]) -> "RunMode":
    """Return :class:`RunMode` from *cfg*, falling back to env then default.

    Lookup order:
    1. ``cfg["run"]["mode"]`` (base.yaml value).
    2. ``SPECTRAQUANT_RUN_MODE`` environment variable.
    3. :attr:`RunMode.NORMAL`.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        Resolved :class:`~spectraquant_v3.core.enums.RunMode`.

    Raises:
        InvalidRunModeError: If the resolved value is not a valid mode.
    """
    from spectraquant_v3.core.enums import RunMode
    from spectraquant_v3.core.errors import InvalidRunModeError

    raw = (
        cfg.get("run", {}).get("mode")
        or os.getenv("SPECTRAQUANT_RUN_MODE", "normal")
    ).lower()
    try:
        return RunMode(raw)
    except ValueError:
        valid = [m.value for m in RunMode]
        raise InvalidRunModeError(
            f"Invalid run mode {raw!r} (from config or env). Valid values: {valid}"
        ) from None


# ---------------------------------------------------------------------------
# Required keys that every valid base config must contain
# ---------------------------------------------------------------------------

_REQUIRED_BASE_KEYS: tuple[str, ...] = ("run", "cache", "qa", "execution", "portfolio")


def validate_config(cfg: dict[str, Any]) -> None:
    """Assert that *cfg* contains all required top-level keys.

    This is a lightweight structural check, not a full JSON-schema validation.
    Call after loading so problems are caught early before they cause
    confusing KeyError failures deep in the pipeline.

    Args:
        cfg: Merged configuration dictionary.

    Raises:
        ConfigValidationError: If a required key is absent or has wrong type.
    """
    from spectraquant_v3.core.errors import ConfigValidationError

    missing = [k for k in _REQUIRED_BASE_KEYS if k not in cfg]
    if missing:
        raise ConfigValidationError(
            f"Config is missing required top-level keys: {missing}. "
            "Check config/v3/base.yaml."
        )

    # Type spot-checks
    if not isinstance(cfg["run"], dict):
        raise ConfigValidationError("config['run'] must be a mapping.")
    if not isinstance(cfg["cache"], dict):
        raise ConfigValidationError("config['cache'] must be a mapping.")
    if not isinstance(cfg["qa"], dict):
        raise ConfigValidationError("config['qa'] must be a mapping.")


def validate_equity_config(cfg: dict[str, Any]) -> None:
    """Assert that *cfg* contains the ``equities`` section required by equity pipelines.

    Call this at the top of :func:`~spectraquant_v3.pipeline.equity_pipeline.run_equity_pipeline`
    so that a config built from the wrong YAML file fails fast with an actionable
    error instead of running the full universe stage with an empty ticker list.

    Args:
        cfg: Merged configuration dictionary.

    Raises:
        ConfigValidationError: If the ``equities`` key is absent or not a mapping.
    """
    from spectraquant_v3.core.errors import ConfigValidationError

    if "equities" not in cfg:
        raise ConfigValidationError(
            "Equity pipeline config is missing required 'equities' section. "
            "Use get_equity_config() or include an 'equities' key in your config dict."
        )
    if not isinstance(cfg["equities"], dict):
        raise ConfigValidationError(
            f"config['equities'] must be a mapping, got {type(cfg['equities']).__name__}."
        )


def validate_crypto_config(cfg: dict[str, Any]) -> None:
    """Assert that *cfg* contains the ``crypto`` section required by crypto pipelines.

    Call this at the top of :func:`~spectraquant_v3.pipeline.crypto_pipeline.run_crypto_pipeline`
    so that a config built from the wrong YAML file fails fast with an actionable
    error instead of running the full universe stage with an empty symbol list.

    Args:
        cfg: Merged configuration dictionary.

    Raises:
        ConfigValidationError: If the ``crypto`` key is absent or not a mapping.
    """
    from spectraquant_v3.core.errors import ConfigValidationError

    if "crypto" not in cfg:
        raise ConfigValidationError(
            "Crypto pipeline config is missing required 'crypto' section. "
            "Use get_crypto_config() or include a 'crypto' key in your config dict."
        )
    if not isinstance(cfg["crypto"], dict):
        raise ConfigValidationError(
            f"config['crypto'] must be a mapping, got {type(cfg['crypto']).__name__}."
        )


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into a copy of *base*."""
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
