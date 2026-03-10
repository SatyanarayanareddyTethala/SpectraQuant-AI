"""Factor registry with versioning for reproducible datasets."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List


@dataclass(frozen=True)
class FactorSpec:
    name: str
    version: str
    lookback: int
    input_columns: List[str]
    output_column: str


_REGISTRY: Dict[str, FactorSpec] = {}


def register_factor(spec: FactorSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"Factor already registered: {spec.name}")
    _REGISTRY[spec.name] = spec


def register_default_factors() -> None:
    defaults = [
        FactorSpec("return_1d", "1.0", 1, ["close"], "ret_1d"),
        FactorSpec("return_5d", "1.0", 5, ["close"], "ret_5d"),
        FactorSpec("sma_5", "1.0", 5, ["close"], "sma_5"),
        FactorSpec("vol_5", "1.0", 5, ["close"], "vol_5"),
        FactorSpec("rsi_14", "1.0", 14, ["close"], "rsi_14"),
    ]
    for spec in defaults:
        if spec.name not in _REGISTRY:
            register_factor(spec)


def get_factor_metadata() -> List[Dict]:
    return [spec.__dict__ for spec in _REGISTRY.values()]


def get_factor_set_hash() -> str:
    digest = hashlib.sha256()
    for spec in sorted(_REGISTRY.values(), key=lambda s: s.name):
        digest.update(f"{spec.name}:{spec.version}".encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "FactorSpec",
    "register_factor",
    "register_default_factors",
    "get_factor_metadata",
    "get_factor_set_hash",
]
