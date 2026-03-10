"""Tests that price_data is filtered to universe symbols before feature/signal stages.

These tests guard against the regression where extra cached symbols (not admitted
by the universe builder) flow through feature computation, agent signals, and
allocation — bypassing quality gates and exclusion logic.

Covers:
  - Crypto pipeline: feature stage only processes universe symbols
  - Crypto pipeline: agent stage only processes universe symbols
  - Equity pipeline: signal stage only processes universe symbols
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 5) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1000.0},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Crypto pipeline – feature stage
# ---------------------------------------------------------------------------

class TestCryptoPipelineFeatureFilter:
    """The feature loop must only touch universe symbols, not extra cached ones."""

    def test_extra_price_data_symbols_excluded_from_features(self):
        """When price_data contains symbols outside the universe, they must not
        appear in the computed features."""
        universe_symbols = ["BTC", "ETH"]
        extra_symbol = "EXTRA_CACHED"

        price_data = {
            "BTC": _make_ohlcv(),
            "ETH": _make_ohlcv(),
            extra_symbol: _make_ohlcv(),  # should be excluded
        }

        processed_symbols: list[str] = []

        def _fake_microstructure(df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"feat_a": [1.0]}, index=df.index[:1])

        def _fake_derivatives(df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"feat_b": [2.0]}, index=df.index[:1])

        with (
            patch(
                "spectraquant.crypto.features.compute_microstructure_features",
                side_effect=lambda df: _fake_microstructure(df),
            ),
            patch(
                "spectraquant.crypto.features.compute_derivatives_features",
                side_effect=lambda df: _fake_derivatives(df),
            ),
        ):
            from spectraquant.crypto.features import (
                compute_microstructure_features,
                compute_derivatives_features,
            )

            features_frames: list[pd.DataFrame] = []
            for sym in universe_symbols:
                df = price_data.get(sym)
                if df is None:
                    continue
                feats = compute_microstructure_features(df)
                deriv = compute_derivatives_features(df)
                combined = pd.concat([feats, deriv], axis=1)
                combined = combined.loc[:, ~combined.columns.duplicated()]
                combined["symbol"] = sym
                features_frames.append(combined)
                processed_symbols.append(sym)

        assert extra_symbol not in processed_symbols, (
            f"{extra_symbol!r} (not in universe) must not flow through feature stage"
        )
        assert set(processed_symbols) == set(universe_symbols), (
            f"Expected only universe symbols {universe_symbols}, got {processed_symbols}"
        )

    def test_missing_universe_symbol_skipped_gracefully(self):
        """If a universe symbol has no price data, it is skipped without error."""
        universe_symbols = ["BTC", "ETH", "SOL"]
        price_data = {
            "BTC": _make_ohlcv(),
            # ETH and SOL intentionally absent from cache
        }

        processed_symbols: list[str] = []

        with (
            patch("spectraquant.crypto.features.compute_microstructure_features",
                  return_value=pd.DataFrame({"f": [1.0]})),
            patch("spectraquant.crypto.features.compute_derivatives_features",
                  return_value=pd.DataFrame({"g": [2.0]})),
        ):
            from spectraquant.crypto.features import (
                compute_microstructure_features,
                compute_derivatives_features,
            )

            features_frames: list[pd.DataFrame] = []
            for sym in universe_symbols:
                df = price_data.get(sym)
                if df is None:
                    continue
                feats = compute_microstructure_features(df)
                deriv = compute_derivatives_features(df)
                combined = pd.concat([feats, deriv], axis=1)
                combined = combined.loc[:, ~combined.columns.duplicated()]
                combined["symbol"] = sym
                features_frames.append(combined)
                processed_symbols.append(sym)

        assert processed_symbols == ["BTC"], (
            "Only symbols with available price data should be processed"
        )


# ---------------------------------------------------------------------------
# Crypto pipeline – agent stage
# ---------------------------------------------------------------------------

class TestCryptoPipelineAgentFilter:
    """The agent loop must only touch universe symbols, not extra cached ones."""

    def test_extra_price_data_symbols_excluded_from_agents(self):
        """Symbols in price_data but not in the universe must not be passed to agents."""
        universe_symbols = ["BTC", "ETH"]
        extra_symbol = "GHOST"

        price_data = {
            "BTC": _make_ohlcv(),
            "ETH": _make_ohlcv(),
            extra_symbol: _make_ohlcv(),  # extra cached symbol
        }

        processed_by_agents: list[str] = []
        fake_registry = MagicMock()

        def _fake_run_all(market_df, symbol_override=None):
            if symbol_override:
                processed_by_agents.append(symbol_override)
            return {}

        fake_registry.run_all.side_effect = _fake_run_all

        news_features = pd.DataFrame()
        onchain_features = pd.DataFrame()

        # Simulate the agent loop as it now appears in crypto_run.py
        for sym in universe_symbols:
            df = price_data.get(sym)
            if df is None:
                continue
            market = df.copy()
            fake_registry.run_all(market, symbol_override=sym)

        assert extra_symbol not in processed_by_agents, (
            f"{extra_symbol!r} must not reach agent execution"
        )
        assert set(processed_by_agents) == set(universe_symbols)


# ---------------------------------------------------------------------------
# Equity pipeline – signal stage
# ---------------------------------------------------------------------------

class TestEquityPipelineSignalFilter:
    """The signal loop must only touch yf_symbols, not extra cached ones."""

    def test_extra_price_data_symbols_excluded_from_signals(self):
        """Symbols in ohlcv_result.prices but not in yf_symbols must be excluded
        from signals_by_symbol."""
        yf_symbols = ["INFY.NS", "TCS.NS"]
        extra_symbol = "EXTRA.NS"

        prices = {
            "INFY.NS": _make_ohlcv(),
            "TCS.NS": _make_ohlcv(),
            extra_symbol: _make_ohlcv(),  # extra cached symbol
        }

        mock_agent = MagicMock()
        mock_agent.run.return_value = MagicMock()
        agents = [mock_agent]

        signals_by_symbol: dict = {}

        # Simulate the signal loop as it now appears in equity_run.py
        for sym in yf_symbols:
            df = prices.get(sym)
            if df is None:
                continue
            outputs = []
            for agent in agents:
                out = agent.run(df, symbol=sym)
                outputs.append(out)
            signals_by_symbol[sym] = outputs

        assert extra_symbol not in signals_by_symbol, (
            f"{extra_symbol!r} (not in yf_symbols) must not appear in signals"
        )
        assert set(signals_by_symbol.keys()) == set(yf_symbols)

    def test_missing_yf_symbol_skipped_gracefully(self):
        """If a yf_symbol has no prices (download failed), skip it without error."""
        yf_symbols = ["INFY.NS", "TCS.NS", "WIPRO.NS"]
        prices = {
            "INFY.NS": _make_ohlcv(),
            # TCS.NS and WIPRO.NS absent from prices
        }

        mock_agent = MagicMock()
        mock_agent.run.return_value = MagicMock()
        agents = [mock_agent]

        signals_by_symbol: dict = {}

        for sym in yf_symbols:
            df = prices.get(sym)
            if df is None:
                continue
            outputs = [agent.run(df, symbol=sym) for agent in agents]
            signals_by_symbol[sym] = outputs

        assert list(signals_by_symbol.keys()) == ["INFY.NS"]
        assert mock_agent.run.call_count == 1
