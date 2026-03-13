"""Hybrid universe loader for SpectraQuant-AI-V3.

Loads a configurable hybrid universe CSV (equities + crypto + FX), validates
its schema, deduplicates symbols, and returns structured per-asset-class lists.

Supported CSV columns:
    asset_class, symbol, exchange, sector, liquidity_score, volatility_score, notes

Usage::

    from spectraquant_v3.core.universe_loader import load_universe

    universe = load_universe("data/universe/hybrid_universe.csv")
    # Returns:
    # {
    #   "equities": [UniverseAsset(...), ...],
    #   "crypto":   [UniverseAsset(...), ...],
    #   "forex":    [UniverseAsset(...), ...],
    # }
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectraquant_v3.core.errors import UniverseValidationError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Columns that must be present in the universe CSV (others are optional).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "asset_class",
    "symbol",
    "exchange",
    "sector",
    "liquidity_score",
    "volatility_score",
)

#: Maximum total assets allowed in a single universe file.
MAX_UNIVERSE_SIZE: int = 150

#: Recognised asset_class values (lowercase).
VALID_ASSET_CLASSES: frozenset[str] = frozenset({"equity", "crypto", "forex"})

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseAsset:
    """Typed representation of a single universe row."""

    asset_class: str
    symbol: str
    exchange: str
    sector: str
    liquidity_score: str
    volatility_score: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation of this asset."""
        return {
            "asset_class": self.asset_class,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "sector": self.sector,
            "liquidity_score": self.liquidity_score,
            "volatility_score": self.volatility_score,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_universe(path: str | Path) -> dict[str, list[UniverseAsset]]:
    """Load and validate the hybrid universe CSV.

    Steps performed:
    1. Read the CSV from *path* (must exist).
    2. Validate that all :data:`REQUIRED_COLUMNS` are present.
    3. Strip whitespace from all string values.
    4. Reject rows with an empty ``symbol`` field.
    5. Reject rows with an unrecognised ``asset_class`` value.
    6. Deduplicate symbols (first occurrence wins; duplicates are dropped).
    7. Reject universe files with more than :data:`MAX_UNIVERSE_SIZE` assets.
    8. Return a dict keyed by ``"equities"``, ``"crypto"``, and ``"forex"``.

    Args:
        path: Filesystem path to the hybrid universe CSV file.

    Returns:
        Dictionary with keys ``"equities"``, ``"crypto"``, and ``"forex"``,
        each mapping to a list of :class:`UniverseAsset` instances.

    Raises:
        FileNotFoundError: If *path* does not exist.
        UniverseValidationError: If the CSV fails schema or content validation.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {csv_path}. "
            "Create data/universe/hybrid_universe.csv or update universe.file in base.yaml."
        )

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        raw_rows = list(reader)
        fieldnames = reader.fieldnames  # capture while file is still open

    # ------------------------------------------------------------------
    # 1. Schema validation: required columns
    # ------------------------------------------------------------------
    if fieldnames is None:
        raise UniverseValidationError(
            f"Universe CSV at {csv_path} appears to be empty or has no header row."
        )

    actual_columns = {col.strip() for col in fieldnames}
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in actual_columns]
    if missing_columns:
        raise UniverseValidationError(
            f"Universe CSV is missing required columns: {missing_columns}. "
            f"Found columns: {sorted(actual_columns)}."
        )

    # ------------------------------------------------------------------
    # 2. Row-level validation and deduplication
    # ------------------------------------------------------------------
    seen_symbols: dict[str, int] = {}  # symbol -> first row index (1-based)
    duplicate_symbols: list[str] = []
    bad_asset_classes: list[tuple[int, str]] = []
    empty_symbol_rows: list[int] = []
    assets: list[UniverseAsset] = []

    for row_idx, raw in enumerate(raw_rows, start=2):  # row 1 is the header
        # Strip whitespace from all values
        row = {k.strip(): v.strip() for k, v in raw.items() if k is not None}

        symbol = row.get("symbol", "")
        if not symbol:
            empty_symbol_rows.append(row_idx)
            continue

        asset_class = row.get("asset_class", "").lower()
        if asset_class not in VALID_ASSET_CLASSES:
            bad_asset_classes.append((row_idx, row.get("asset_class", "")))
            continue

        if symbol in seen_symbols:
            duplicate_symbols.append(symbol)
            continue  # first occurrence wins; drop duplicates

        seen_symbols[symbol] = row_idx
        assets.append(
            UniverseAsset(
                asset_class=asset_class,
                symbol=symbol,
                exchange=row.get("exchange", ""),
                sector=row.get("sector", ""),
                liquidity_score=row.get("liquidity_score", ""),
                volatility_score=row.get("volatility_score", ""),
                notes=row.get("notes", ""),
            )
        )

    # Collect all validation errors and raise together
    errors: list[str] = []
    if empty_symbol_rows:
        errors.append(
            f"Rows with empty symbol field (row numbers): {empty_symbol_rows}."
        )
    if bad_asset_classes:
        errors.append(
            f"Rows with invalid asset_class (row, value): {bad_asset_classes}. "
            f"Valid values: {sorted(VALID_ASSET_CLASSES)}."
        )
    if errors:
        raise UniverseValidationError(
            "Universe CSV failed row-level validation:\n" + "\n".join(errors)
        )

    # ------------------------------------------------------------------
    # 3. Size guard
    # ------------------------------------------------------------------
    if len(assets) > MAX_UNIVERSE_SIZE:
        raise UniverseValidationError(
            f"Universe contains {len(assets)} assets after deduplication, "
            f"which exceeds the maximum of {MAX_UNIVERSE_SIZE}. "
            "Reduce the universe size or increase MAX_UNIVERSE_SIZE."
        )

    # ------------------------------------------------------------------
    # 4. Partition by asset class
    # ------------------------------------------------------------------
    equities: list[UniverseAsset] = []
    crypto: list[UniverseAsset] = []
    forex: list[UniverseAsset] = []

    for asset in assets:
        if asset.asset_class == "equity":
            equities.append(asset)
        elif asset.asset_class == "crypto":
            crypto.append(asset)
        elif asset.asset_class == "forex":
            forex.append(asset)

    return {
        "equities": equities,
        "crypto": crypto,
        "forex": forex,
        "_duplicates_dropped": duplicate_symbols,  # type: ignore[dict-item]
    }


def get_symbols_by_class(
    universe: dict[str, list[UniverseAsset]],
    asset_class: str,
) -> list[str]:
    """Extract symbol strings for a given asset class from a loaded universe dict.

    Args:
        universe: Return value of :func:`load_universe`.
        asset_class: One of ``"equities"``, ``"crypto"``, or ``"forex"``.

    Returns:
        List of symbol strings, preserving order from the CSV.

    Raises:
        KeyError: If *asset_class* is not a valid key in *universe*.
    """
    return [asset.symbol for asset in universe[asset_class]]


def inject_universe_into_config(
    cfg: dict[str, Any],
    universe_path: str | Path,
) -> tuple[dict[str, Any], dict[str, list[UniverseAsset]]]:
    """Load the hybrid universe and return a patched copy of *cfg* with its symbols.

    This is the integration shim used by the equity and crypto CLI ``run``
    commands.  When a universe file is configured, the symbols it defines
    for each asset class override the default symbol lists in the returned
    config copy.  The original *cfg* dict is never mutated.

    Specifically:
    - Equity symbols replace ``patched_cfg["equities"]["universe"]["tickers"]``.
    - Crypto symbols replace ``patched_cfg["crypto"]["symbols"]``.

    Args:
        cfg:            Merged configuration dictionary (not mutated).
        universe_path:  Path to the hybrid universe CSV.

    Returns:
        A ``(patched_cfg, universe)`` tuple where *patched_cfg* is a deep copy
        of *cfg* with the universe symbols injected, and *universe* is the full
        parsed universe dict from :func:`load_universe`.

    Raises:
        FileNotFoundError: If *universe_path* does not exist.
        UniverseValidationError: If the CSV fails validation.
    """
    import copy

    universe = load_universe(universe_path)
    patched = copy.deepcopy(cfg)

    equity_symbols = get_symbols_by_class(universe, "equities")
    crypto_symbols = get_symbols_by_class(universe, "crypto")

    if equity_symbols:
        patched.setdefault("equities", {}).setdefault("universe", {})
        patched["equities"]["universe"]["tickers"] = equity_symbols

    if crypto_symbols:
        patched.setdefault("crypto", {})
        patched["crypto"]["symbols"] = crypto_symbols

    return patched, universe
