"""Build and filter a tradeable crypto universe from static CSV data."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CSV = Path(__file__).with_name("crypto_universe.csv")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CryptoAsset:
    """Single crypto asset with metadata used for universe filtering."""

    symbol: str
    name: str
    sector: str
    market_cap_rank: int
    avg_daily_volume: float = 0.0
    is_stablecoin: bool = False
    exchange_count: int = 0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_crypto_universe(
    csv_path: Path | str | None = None,
    min_volume: float = 0.0,
    min_market_cap_rank: int = 500,
    exclude_stablecoins: bool = True,
    max_assets: int | None = None,
) -> list[CryptoAsset]:
    """Load the CSV universe file and return assets that pass all filters.

    Parameters
    ----------
    csv_path:
        Path to the CSV file. Defaults to the bundled ``crypto_universe.csv``.
    min_volume:
        Minimum average daily volume (USD) to include an asset.
    min_market_cap_rank:
        Only include assets whose ``market_cap_rank`` is at most this value
        (lower rank number == larger market cap).
    exclude_stablecoins:
        When *True*, assets flagged as stablecoins are removed.
    max_assets:
        If set, cap the returned list at this many assets (after filtering
        and ranking by liquidity).
    """
    csv_path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV
    logger.info("Loading crypto universe from %s", csv_path)

    # NOTE: avg_daily_volume and exchange_count are not present in the CSV;
    # they default to zero and are meant to be enriched later via
    # fetch_universe_metadata() once a live data source is integrated.
    assets: list[CryptoAsset] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            asset = CryptoAsset(
                symbol=row["symbol"].strip().upper(),
                name=row["name"].strip(),
                sector=row["sector"].strip(),
                market_cap_rank=int(row["market_cap_rank"]),
                is_stablecoin=row["is_stablecoin"].strip().lower() == "true",
            )
            assets.append(asset)

    logger.info("Parsed %d assets from CSV", len(assets))

    # -- apply filters -------------------------------------------------------
    if exclude_stablecoins:
        assets = [a for a in assets if not a.is_stablecoin]

    assets = [a for a in assets if a.market_cap_rank <= min_market_cap_rank]
    assets = [a for a in assets if a.avg_daily_volume >= min_volume]

    assets = rank_by_liquidity(assets)

    if max_assets is not None:
        assets = assets[:max_assets]

    logger.info("Universe built with %d assets after filtering", len(assets))
    return assets


def fetch_universe_metadata(symbols: list[str]) -> pd.DataFrame:
    """Return a DataFrame with volume and market-cap data for *symbols*.

    .. note::
        This is a placeholder that returns synthetic data.  Replace with a
        live API call (e.g. CoinGecko, CoinMarketCap) in production.

    TODO: Integrate live market-data API for real-time volume/mcap figures.
    """
    logger.warning(
        "fetch_universe_metadata is using placeholder data for %d symbols",
        len(symbols),
    )
    records = [
        {
            "symbol": s,
            "avg_daily_volume": 0.0,
            "market_cap_usd": 0.0,
        }
        for s in symbols
    ]
    return pd.DataFrame.from_records(records)


def rank_by_liquidity(assets: list[CryptoAsset]) -> list[CryptoAsset]:
    """Sort assets by descending volume then ascending market-cap rank."""
    return sorted(
        assets,
        key=lambda a: (-a.avg_daily_volume, a.market_cap_rank),
    )
