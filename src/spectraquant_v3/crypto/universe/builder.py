from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant_v3.core.enums import AssetClass
from spectraquant_v3.core.errors import EmptyUniverseError
from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry


@dataclass
class UniverseEntry:
    canonical_symbol: str
    asset_class: str = AssetClass.CRYPTO.value
    included: bool = False
    reason: str = ""
    market_cap_usd: float = 0.0
    volume_24h_usd: float = 0.0
    age_days: int = 0
    source: str = ""


@dataclass
class UniverseArtifact:
    run_id: str
    as_of: str
    asset_class: str = AssetClass.CRYPTO.value
    universe_mode: str = "static"
    included_symbols: list[str] = field(default_factory=list)
    excluded_symbols: list[str] = field(default_factory=list)
    entries: list[UniverseEntry] = field(default_factory=list)

    def write(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out / f"universe_crypto_{ts}_{self.run_id}.json"
        payload = {
            "run_id": self.run_id,
            "as_of": self.as_of,
            "asset_class": self.asset_class,
            "universe_mode": self.universe_mode,
            "included_count": len(self.included_symbols),
            "excluded_count": len(self.excluded_symbols),
            "included_symbols": sorted(self.included_symbols),
            "excluded_symbols": sorted(self.excluded_symbols),
            "entries": [asdict(e) for e in self.entries],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path


@dataclass
class CryptoQualityGate:
    min_market_cap_usd: float = 0.0
    min_24h_volume_usd: float = 0.0
    min_age_days: int = 0
    require_tradable_mapping: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CryptoQualityGate":
        qg = cfg.get("crypto", {}).get("quality_gate", {})
        return cls(
            min_market_cap_usd=float(qg.get("min_market_cap_usd", 0.0)),
            min_24h_volume_usd=float(qg.get("min_24h_volume_usd", 0.0)),
            min_age_days=int(qg.get("min_age_days", 0)),
            require_tradable_mapping=bool(qg.get("require_tradable_mapping", True)),
        )


class CryptoUniverseBuilder:
    def __init__(self, cfg: dict[str, Any], registry: CryptoSymbolRegistry, run_id: str = "unknown") -> None:
        self._cfg = cfg
        self._registry = registry
        self._run_id = run_id
        self._quality_gate = CryptoQualityGate.from_config(cfg)

    def build(self, market_data: dict[str, dict[str, Any]] | None = None) -> UniverseArtifact:
        from spectraquant_v3.crypto.universe.universe_engine import UniverseEngine

        cfg = dict(self._cfg)
        cfg_crypto = dict(cfg.get("crypto", {}))
        filters = dict(cfg_crypto.get("universe_filters", {}))
        if "require_tradable_mapping" not in filters:
            filters["require_tradable_mapping"] = self._quality_gate.require_tradable_mapping
        cfg_crypto["universe_filters"] = filters
        cfg["crypto"] = cfg_crypto

        engine = UniverseEngine(cfg, self._registry)
        rows = engine.evaluate(market_data=market_data)

        included = sorted([r.canonical_symbol for r in rows if r.included])
        excluded = sorted([r.canonical_symbol for r in rows if not r.included])
        if not included:
            raise EmptyUniverseError(
                f"EMPTY_CRYPTO_UNIVERSE: 0 of {len(rows)} candidates passed filters in run '{self._run_id}'."
            )
        return UniverseArtifact(
            run_id=self._run_id,
            as_of=datetime.now(timezone.utc).isoformat(),
            universe_mode=cfg_crypto.get("universe_mode", "static"),
            included_symbols=included,
            excluded_symbols=excluded,
            entries=rows,
        )
