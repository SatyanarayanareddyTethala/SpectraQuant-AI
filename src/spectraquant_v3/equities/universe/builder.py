"""Equity universe builder for SpectraQuant-AI-V3.

Resolves the set of equity symbols that are eligible for a given run,
applies quality gates, and writes a universe artifact (JSON) that records
the inclusion/exclusion decision and reason for every candidate symbol.

This module must never import from ``spectraquant_v3.crypto``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectraquant_v3.core.enums import AssetClass
from spectraquant_v3.core.errors import AssetClassLeakError, EmptyUniverseError
from spectraquant_v3.equities.symbols.registry import EquitySymbolRegistry


# ---------------------------------------------------------------------------
# Universe entry dataclass
# ---------------------------------------------------------------------------

@dataclass
class UniverseEntry:
    """One candidate symbol evaluated during universe construction."""

    canonical_symbol: str
    asset_class: str = AssetClass.EQUITY.value
    included: bool = False
    reason: str = ""
    last_close: float = 0.0
    avg_volume: float = 0.0
    history_days: int = 0
    source: str = ""


@dataclass
class UniverseArtifact:
    """Full universe artifact written as JSON after each run."""

    run_id: str
    as_of: str
    asset_class: str = AssetClass.EQUITY.value
    included_symbols: list[str] = field(default_factory=list)
    excluded_symbols: list[str] = field(default_factory=list)
    entries: list[UniverseEntry] = field(default_factory=list)

    def write(self, output_dir: str | Path) -> Path:
        """Persist the artifact as JSON.

        Args:
            output_dir: Directory to write to (created if absent).

        Returns:
            Path to the written JSON file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out / f"universe_equities_{ts}_{self.run_id}.json"
        payload = {
            "run_id": self.run_id,
            "as_of": self.as_of,
            "asset_class": self.asset_class,
            "included_count": len(self.included_symbols),
            "excluded_count": len(self.excluded_symbols),
            "included_symbols": sorted(self.included_symbols),
            "excluded_symbols": sorted(self.excluded_symbols),
            "entries": [asdict(e) for e in self.entries],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

@dataclass
class EquityQualityGate:
    """Quality thresholds applied to each candidate equity symbol."""

    min_price: float = 10.0
    min_avg_volume: float = 100_000.0
    min_history_days: int = 60

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EquityQualityGate":
        """Build from ``cfg["equities"]["quality_gate"]``."""
        qg = cfg.get("equities", {}).get("quality_gate", {})
        return cls(
            min_price=float(qg.get("min_price", 10.0)),
            min_avg_volume=float(qg.get("min_avg_volume", 100_000.0)),
            min_history_days=int(qg.get("min_history_days", 60)),
        )

    def check(self, entry: UniverseEntry) -> tuple[bool, str]:
        """Apply all quality gates to *entry*.

        Returns:
            (passed, reason) where *reason* is empty string when passed.
        """
        if (
            self.min_price > 0
            and entry.last_close > 0
            and entry.last_close < self.min_price
        ):
            return (
                False,
                f"last_close={entry.last_close:.2f} < min_price={self.min_price:.2f}",
            )
        if (
            self.min_avg_volume > 0
            and entry.avg_volume > 0
            and entry.avg_volume < self.min_avg_volume
        ):
            return (
                False,
                f"avg_volume={entry.avg_volume:.0f} "
                f"< min_avg_volume={self.min_avg_volume:.0f}",
            )
        if (
            self.min_history_days > 0
            and entry.history_days > 0
            and entry.history_days < self.min_history_days
        ):
            return (
                False,
                f"history_days={entry.history_days} "
                f"< min_history_days={self.min_history_days}",
            )
        return True, ""


# ---------------------------------------------------------------------------
# Universe builder
# ---------------------------------------------------------------------------

class EquityUniverseBuilder:
    """Resolves the equity trading universe for a single pipeline run.

    Args:
        cfg:      Merged equity configuration dict.
        registry: Pre-populated :class:`EquitySymbolRegistry`.
        run_id:   Parent run identifier for artifact naming.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        registry: EquitySymbolRegistry,
        run_id: str = "unknown",
    ) -> None:
        self._cfg = cfg
        self._registry = registry
        self._run_id = run_id
        self._quality_gate = EquityQualityGate.from_config(cfg)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(
        self,
        price_data: dict[str, dict[str, Any]] | None = None,
    ) -> UniverseArtifact:
        """Build the universe and return a :class:`UniverseArtifact`.

        Args:
            price_data: Optional dict keyed by canonical symbol containing
                        summary metrics (``last_close``, ``avg_volume``,
                        ``history_days``).  When *None*, quality gates that
                        require this data are skipped (symbols pass by default).

        Returns:
            :class:`UniverseArtifact` with full inclusion/exclusion log.

        Raises:
            EmptyUniverseError: If zero symbols pass quality gates.
            AssetClassLeakError: If any candidate symbol is a crypto pair.
        """
        candidates = self._resolve_candidates()
        price_data = price_data or {}

        entries: list[UniverseEntry] = []
        included: list[str] = []
        excluded: list[str] = []

        # Excludes list from config
        exclude_set: set[str] = {
            s.upper()
            for s in self._cfg.get("equities", {})
            .get("universe", {})
            .get("exclude", [])
        }

        for sym in candidates:
            # Asset-class leak guard
            from spectraquant_v3.equities.symbols.registry import _is_crypto_symbol
            if _is_crypto_symbol(sym):
                raise AssetClassLeakError(
                    f"Crypto symbol '{sym}' found in equity universe candidates."
                )

            pd_data = price_data.get(sym, {})
            entry = UniverseEntry(
                canonical_symbol=sym,
                source="config",
                last_close=float(pd_data.get("last_close", 0.0)),
                avg_volume=float(pd_data.get("avg_volume", 0.0)),
                history_days=int(pd_data.get("history_days", 0)),
            )

            # Explicit exclusion list
            if sym.upper() in exclude_set:
                entry.included = False
                entry.reason = "explicitly_excluded_in_config"
                entries.append(entry)
                excluded.append(sym)
                continue

            # Registry check
            if not self._registry.contains(sym):
                entry.included = False
                entry.reason = f"Symbol '{sym}' not in symbol registry"
                entries.append(entry)
                excluded.append(sym)
                continue

            passed, reason = self._quality_gate.check(entry)
            entry.included = passed
            entry.reason = reason if not passed else "passed_all_gates"
            entries.append(entry)
            if passed:
                included.append(sym)
            else:
                excluded.append(sym)

        if not included:
            raise EmptyUniverseError(
                f"EMPTY_EQUITY_UNIVERSE: 0 of {len(candidates)} candidates passed "
                f"quality gates in run '{self._run_id}'. "
                f"Excluded: {excluded}"
            )

        return UniverseArtifact(
            run_id=self._run_id,
            as_of=datetime.now(timezone.utc).isoformat(),
            included_symbols=sorted(included),
            excluded_symbols=sorted(excluded),
            entries=entries,
        )

    # ------------------------------------------------------------------
    # Candidate resolution
    # ------------------------------------------------------------------

    def _resolve_candidates(self) -> list[str]:
        """Return the candidate symbols from config."""
        universe_cfg = self._cfg.get("equities", {}).get("universe", {})
        tickers: list[str] = universe_cfg.get("tickers", [])
        if tickers:
            return tickers
        # Fall back to registry
        return self._registry.all_symbols()
