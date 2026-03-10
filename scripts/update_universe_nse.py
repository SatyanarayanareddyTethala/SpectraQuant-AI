#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraquant.core.universe import update_nse_universe


def main() -> None:
    df, dedup_removed = update_nse_universe(
        raw_path=Path("data/universe/raw/EQUITY_L.csv"),
        output_path=Path("data/universe/universe_nse.csv"),
    )
    print(f"count={len(df)}")
    print("first_10_rows=")
    print(df.head(10).to_string(index=False))
    print(f"duplicates_removed={dedup_removed > 0} (removed={dedup_removed})")


if __name__ == "__main__":
    main()
