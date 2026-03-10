from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectraquant_v3.core.errors import AssetClassLeakError, ConfigValidationError
from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry, _is_equity_symbol
from spectraquant_v3.crypto.universe.builder import UniverseEntry


@dataclass
class UniverseFilterConfig:
    exclude_stablecoins: bool = True
    exclude_wrapped_assets: bool = True
    min_market_cap_usd: float = 0.0
    min_daily_volume_usd: float = 0.0
    min_listing_age_days: int = 0
    require_exchange_coverage: bool = True
    required_exchanges: list[str] = field(default_factory=list)
    require_tradable_mapping: bool = True
    top_market_cap_enabled: bool = False
    top_market_cap_limit: int = 0
    collect_all_fail_reasons: bool = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "UniverseFilterConfig":
        crypto = cfg.get("crypto", {})
        filters = crypto.get("universe_filters", {})
        quality = crypto.get("quality_gate", {})
        top_market_cap = filters.get("top_market_cap", {})

        required_exchanges = filters.get("required_exchanges", crypto.get("exchanges", []))
        limit = int(top_market_cap.get("limit", crypto.get("universe_top_n", 0) or 0))

        return cls(
            exclude_stablecoins=bool(filters.get("exclude_stablecoins", True)),
            exclude_wrapped_assets=bool(filters.get("exclude_wrapped_assets", True)),
            min_market_cap_usd=float(filters.get("min_market_cap_usd", quality.get("min_market_cap_usd", 0.0))),
            min_daily_volume_usd=float(filters.get("min_daily_volume_usd", quality.get("min_24h_volume_usd", 0.0))),
            min_listing_age_days=int(filters.get("min_listing_age_days", quality.get("min_age_days", 0))),
            require_exchange_coverage=bool(filters.get("require_exchange_coverage", True)),
            required_exchanges=[str(x).lower() for x in required_exchanges],
            require_tradable_mapping=bool(filters.get("require_tradable_mapping", quality.get("require_tradable_mapping", True))),
            top_market_cap_enabled=bool(top_market_cap.get("enabled", False)),
            top_market_cap_limit=limit,
            collect_all_fail_reasons=bool(filters.get("collect_all_fail_reasons", False)),
        )


@dataclass
class UniverseEvaluationRow(UniverseEntry):
    fail_reasons: list[str] = field(default_factory=list)


class UniverseEngine:
    def __init__(self, cfg: dict[str, Any], registry: CryptoSymbolRegistry) -> None:
        self._cfg = cfg
        self._registry = registry
        self._filters = UniverseFilterConfig.from_config(cfg)

    def evaluate(
        self,
        market_data: dict[str, dict[str, Any]] | None = None,
        provider_candidates: list[str] | None = None,
    ) -> list[UniverseEvaluationRow]:
        self._validate_config()
        metadata = market_data or {}
        candidates = self._load_candidates(provider_candidates, metadata)
        rows: list[UniverseEvaluationRow] = []

        for sym in candidates:
            if _is_equity_symbol(sym):
                raise AssetClassLeakError(f"Equity symbol '{sym}' found in crypto universe candidates.")
            md = metadata.get(sym, {})
            row = UniverseEvaluationRow(
                canonical_symbol=sym,
                market_cap_usd=float(md.get("market_cap_usd", 0.0)),
                volume_24h_usd=float(md.get("volume_24h_usd", md.get("daily_volume_usd", 0.0))),
                age_days=int(md.get("age_days", md.get("listing_age_days", 0))),
                source=md.get("source", self._cfg.get("crypto", {}).get("universe_mode", "static")),
            )
            row.fail_reasons = self._row_fail_reasons(row, md)
            row.included = not row.fail_reasons
            row.reason = row.fail_reasons[0] if row.fail_reasons else "passed_all_gates"
            if not self._filters.collect_all_fail_reasons and row.fail_reasons:
                row.fail_reasons = [row.fail_reasons[0]]
            rows.append(row)

        if self._filters.top_market_cap_enabled and self._filters.top_market_cap_limit > 0:
            rows = self._apply_top_market_cap(rows)
        return rows

    def _load_candidates(
        self,
        provider_candidates: list[str] | None,
        market_data: dict[str, dict[str, Any]],
    ) -> list[str]:
        crypto_cfg = self._cfg.get("crypto", {})
        mode = str(crypto_cfg.get("universe_mode", "static"))
        configured = [s.upper() for s in crypto_cfg.get("symbols", [])]
        provider = [s.upper() for s in (provider_candidates or [])]

        if mode == "static":
            return sorted(set(configured) | set(provider))

        all_syms = self._registry.all_symbols()
        ranked = sorted(
            all_syms,
            key=lambda s: float(market_data.get(s, {}).get("market_cap_usd", 0.0)),
            reverse=True,
        )
        if mode == "dataset_topN":
            n = int(crypto_cfg.get("universe_top_n", 20))
            return ranked[:n]
        if mode == "hybrid":
            n = int(crypto_cfg.get("universe_top_n", 20))
            return sorted(set(configured) | set(ranked[:n]) | set(provider))
        if configured or provider:
            return sorted(set(configured) | set(provider))
        return all_syms

    def _row_fail_reasons(self, row: UniverseEvaluationRow, md: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if self._filters.require_tradable_mapping and not self._registry.contains(row.canonical_symbol):
            reasons.append("missing_registry_mapping")

        if self._filters.exclude_stablecoins and bool(md.get("is_stablecoin", False)):
            reasons.append("stablecoin_excluded")

        if self._filters.exclude_wrapped_assets and bool(md.get("is_wrapped", False)):
            reasons.append("wrapped_asset_excluded")

        if self._filters.min_market_cap_usd > 0 and row.market_cap_usd > 0 and row.market_cap_usd < self._filters.min_market_cap_usd:
            reasons.append("below_min_market_cap")

        if self._filters.min_daily_volume_usd > 0 and row.volume_24h_usd > 0 and row.volume_24h_usd < self._filters.min_daily_volume_usd:
            reasons.append("below_min_daily_volume")

        if self._filters.min_listing_age_days > 0 and row.age_days > 0 and row.age_days < self._filters.min_listing_age_days:
            reasons.append("below_min_listing_age")

        if self._filters.require_exchange_coverage and "exchanges" in md:
            available = {str(x).lower() for x in md.get("exchanges", [])}
            missing = [ex for ex in self._filters.required_exchanges if ex not in available]
            if missing:
                reasons.append(f"missing_exchange_coverage:{','.join(missing)}")

        return reasons

    def _apply_top_market_cap(self, rows: list[UniverseEvaluationRow]) -> list[UniverseEvaluationRow]:
        included = [r for r in rows if r.included]
        keep = {
            r.canonical_symbol
            for r in sorted(included, key=lambda x: x.market_cap_usd, reverse=True)[: self._filters.top_market_cap_limit]
        }
        for row in rows:
            if row.included and row.canonical_symbol not in keep:
                row.included = False
                reason = "outside_top_market_cap"
                row.reason = row.reason if row.reason != "passed_all_gates" else reason
                if reason not in row.fail_reasons:
                    row.fail_reasons.append(reason)
        return rows

    def _validate_config(self) -> None:
        if self._filters.top_market_cap_enabled and self._filters.top_market_cap_limit <= 0:
            raise ConfigValidationError("crypto.universe_filters.top_market_cap.limit must be > 0 when enabled")
