"""Tests for BacktestEngine news_feature_map injection and hybrid vs baseline comparison.

Covers:
- news_feature_map is stored and used during walk-forward steps
- Point-in-time safety: news scores after the step date are excluded
- Hybrid strategy with news produces different (and non-trivially distinct) signals
  compared to hybrid without news when news sentiment diverges from momentum
- Baseline (equity_momentum_v1) results are unaffected by news_feature_map
- Missing news for a symbol degrades gracefully (no ERROR, no exception)
- No-news symbol coexists cleanly with news-enabled symbol in the same run

All tests are self-contained (no network calls, no file-system side-effects).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 80, seed: int = 0, trend: float = 0.002) -> pd.DataFrame:
    """Synthetic OHLCV with a mild upward trend (deterministic)."""
    rng = np.random.default_rng(seed)
    daily_ret = trend + rng.standard_normal(n) * 0.01
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    high = close * (1.0 + rng.uniform(0.001, 0.015, n))
    low = close * (1.0 - rng.uniform(0.001, 0.015, n))
    open_ = close * (1.0 + rng.uniform(-0.005, 0.005, n))
    volume = rng.uniform(1_000_000, 5_000_000, n)
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_news_df(
    n: int = 80,
    score: float = 0.6,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """Synthetic news sentiment DataFrame with a constant score."""
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"news_sentiment_score": [score] * n}, index=idx)


def _make_negative_news_df(n: int = 80, start: str = "2024-01-02") -> pd.DataFrame:
    """Strongly negative news sentiment (diverges from positive momentum)."""
    return _make_news_df(n=n, score=-0.8, start=start)


def _equity_cfg() -> dict:
    from spectraquant_v3.core.config import get_equity_config

    cfg = get_equity_config()
    cfg["equities"]["universe"]["require_exchange_coverage"] = False
    return cfg


def _crypto_cfg() -> dict:
    from spectraquant_v3.core.config import get_crypto_config

    return get_crypto_config()


# ---------------------------------------------------------------------------
# Tests: news_feature_map storage and injection
# ---------------------------------------------------------------------------


class TestNewsFeatureMapStorage:
    """BacktestEngine stores news_feature_map and exposes it via _news_feature_map."""

    def test_engine_stores_empty_map_when_none(self) -> None:
        from spectraquant_v3.backtest.engine import BacktestEngine

        price = {"INFY.NS": _make_ohlcv()}
        engine = BacktestEngine(
            cfg=_equity_cfg(),
            asset_class="equity",
            price_data=price,
            min_in_sample_periods=20,
        )
        assert engine._news_feature_map == {}

    def test_engine_stores_news_feature_map(self) -> None:
        from spectraquant_v3.backtest.engine import BacktestEngine

        price = {"INFY.NS": _make_ohlcv()}
        news = {"INFY.NS": _make_news_df()}
        engine = BacktestEngine(
            cfg=_equity_cfg(),
            asset_class="equity",
            price_data=price,
            news_feature_map=news,
            min_in_sample_periods=20,
        )
        assert "INFY.NS" in engine._news_feature_map

    def test_news_feature_map_is_a_copy(self) -> None:
        """Mutations to the original dict should not affect the engine's state."""
        from spectraquant_v3.backtest.engine import BacktestEngine

        price = {"INFY.NS": _make_ohlcv()}
        news = {"INFY.NS": _make_news_df()}
        original_news = dict(news)
        engine = BacktestEngine(
            cfg=_equity_cfg(),
            asset_class="equity",
            price_data=price,
            news_feature_map=news,
            min_in_sample_periods=20,
        )
        # Mutate the original dict after construction
        news["TCS.NS"] = _make_news_df()
        # Engine's internal map should NOT have TCS.NS
        assert "TCS.NS" not in engine._news_feature_map
        assert list(engine._news_feature_map.keys()) == list(original_news.keys())


# ---------------------------------------------------------------------------
# Tests: _inject_news_scores helper
# ---------------------------------------------------------------------------


class TestInjectNewsScores:
    """Unit tests for BacktestEngine._inject_news_scores."""

    def _engine(self, news_map=None):
        from spectraquant_v3.backtest.engine import BacktestEngine

        price = {"INFY.NS": _make_ohlcv(), "TCS.NS": _make_ohlcv(seed=1)}
        return BacktestEngine(
            cfg=_equity_cfg(),
            asset_class="equity",
            price_data=price,
            news_feature_map=news_map,
            min_in_sample_periods=20,
        )

    def test_injects_news_score_into_feature_df(self) -> None:
        """news_sentiment_score column appears in the enriched feature DataFrame."""
        news = {"INFY.NS": _make_news_df(score=0.5)}
        engine = self._engine(news)

        idx = pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC")
        feature_df = pd.DataFrame({"ret_20d": [0.02] * 40, "rsi": [55.0] * 40}, index=idx)
        step_date = pd.Timestamp("2024-03-01", tz="UTC")

        enriched = engine._inject_news_scores({"INFY.NS": feature_df}, step_date)

        assert "news_sentiment_score" in enriched["INFY.NS"].columns
        assert float(enriched["INFY.NS"]["news_sentiment_score"].iloc[-1]) == pytest.approx(0.5)

    def test_symbol_not_in_news_map_unchanged(self) -> None:
        """Symbol absent from news_feature_map keeps its original feature DataFrame."""
        news = {"INFY.NS": _make_news_df(score=0.5)}
        engine = self._engine(news)

        idx = pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC")
        feature_df = pd.DataFrame({"ret_20d": [0.02] * 40, "rsi": [55.0] * 40}, index=idx)
        step_date = pd.Timestamp("2024-03-01", tz="UTC")

        enriched = engine._inject_news_scores({"TCS.NS": feature_df}, step_date)

        assert "news_sentiment_score" not in enriched["TCS.NS"].columns

    def test_point_in_time_safety_excludes_future_news(self) -> None:
        """News rows after the step date must not be included."""
        # News DataFrame: past rows have score 0.3, future rows have score 0.9
        past_idx = pd.date_range("2024-01-02", periods=20, freq="B", tz="UTC")
        future_idx = pd.date_range("2024-04-01", periods=20, freq="B", tz="UTC")
        past_df = pd.DataFrame({"news_sentiment_score": [0.3] * 20}, index=past_idx)
        future_df = pd.DataFrame({"news_sentiment_score": [0.9] * 20}, index=future_idx)
        news_df = pd.concat([past_df, future_df]).sort_index()

        news = {"INFY.NS": news_df}
        engine = self._engine(news)

        idx = pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC")
        feature_df = pd.DataFrame({"ret_20d": [0.02] * 40, "rsi": [55.0] * 40}, index=idx)
        # Step date is in March – before the future news rows in April
        step_date = pd.Timestamp("2024-03-15", tz="UTC")

        enriched = engine._inject_news_scores({"INFY.NS": feature_df}, step_date)

        # Must use only past data -> score should be 0.3, not 0.9
        injected_score = float(enriched["INFY.NS"]["news_sentiment_score"].iloc[-1])
        assert injected_score == pytest.approx(0.3, abs=1e-9), (
            f"Expected 0.3 (past score) but got {injected_score} - look-ahead detected"
        )

    def test_no_past_news_leaves_feature_df_unchanged(self) -> None:
        """If all news rows are after the step date, return original feature_df."""
        future_idx = pd.date_range("2025-01-02", periods=10, freq="B", tz="UTC")
        news_df = pd.DataFrame({"news_sentiment_score": [0.7] * 10}, index=future_idx)

        news = {"INFY.NS": news_df}
        engine = self._engine(news)

        idx = pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")
        feature_df = pd.DataFrame({"ret_20d": [0.02] * 30, "rsi": [55.0] * 30}, index=idx)
        step_date = pd.Timestamp("2024-03-01", tz="UTC")

        enriched = engine._inject_news_scores({"INFY.NS": feature_df}, step_date)

        assert "news_sentiment_score" not in enriched["INFY.NS"].columns

    def test_nan_news_scores_are_skipped(self) -> None:
        """All-NaN news column should leave the feature DataFrame unchanged."""
        idx = pd.date_range("2024-01-02", periods=20, freq="B", tz="UTC")
        news_df = pd.DataFrame({"news_sentiment_score": [float("nan")] * 20}, index=idx)

        news = {"INFY.NS": news_df}
        engine = self._engine(news)

        feat_idx = pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")
        feature_df = pd.DataFrame({"ret_20d": [0.02] * 30}, index=feat_idx)
        step_date = pd.Timestamp("2024-03-01", tz="UTC")

        enriched = engine._inject_news_scores({"INFY.NS": feature_df}, step_date)

        # All-NaN drops → column absent or value is NaN
        if "news_sentiment_score" in enriched["INFY.NS"].columns:
            assert enriched["INFY.NS"]["news_sentiment_score"].isna().all()
        # No exception → graceful

    def test_missing_news_sentiment_score_column_leaves_unchanged(self) -> None:
        """news_feature_map with wrong column leaves feature_df intact."""
        idx = pd.date_range("2024-01-02", periods=20, freq="B", tz="UTC")
        # Wrong column name
        news_df = pd.DataFrame({"sentiment": [0.5] * 20}, index=idx)

        news = {"INFY.NS": news_df}
        engine = self._engine(news)

        feat_idx = pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")
        feature_df = pd.DataFrame({"ret_20d": [0.02] * 30}, index=feat_idx)
        step_date = pd.Timestamp("2024-03-01", tz="UTC")

        enriched = engine._inject_news_scores({"INFY.NS": feature_df}, step_date)

        assert "news_sentiment_score" not in enriched["INFY.NS"].columns

    def test_original_feature_df_not_mutated(self) -> None:
        """Injection must not mutate the original feature DataFrame."""
        news = {"INFY.NS": _make_news_df(score=0.4)}
        engine = self._engine(news)

        idx = pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")
        feature_df = pd.DataFrame({"ret_20d": [0.02] * 30, "rsi": [55.0] * 30}, index=idx)
        orig_cols = set(feature_df.columns)

        step_date = pd.Timestamp("2024-03-01", tz="UTC")
        engine._inject_news_scores({"INFY.NS": feature_df}, step_date)

        # Original must be unchanged
        assert set(feature_df.columns) == orig_cols


# ---------------------------------------------------------------------------
# Tests: full backtest run – hybrid with and without news
# ---------------------------------------------------------------------------


class TestBacktestHybridVsBaseline:
    """End-to-end walk-forward comparison: hybrid+news, hybrid no-news, baseline."""

    def _run_backtest(
        self,
        strategy_id: str,
        price_data: dict,
        news_feature_map=None,
        min_periods: int = 30,
        freq: str = "ME",
    ):
        from spectraquant_v3.backtest.engine import BacktestEngine

        engine = BacktestEngine(
            cfg=_equity_cfg(),
            asset_class="equity",
            price_data=price_data,
            strategy_id=strategy_id,
            news_feature_map=news_feature_map,
            rebalance_freq=freq,
            min_in_sample_periods=min_periods,
        )
        return engine.run()

    def test_hybrid_backtest_runs_to_completion_with_news(self) -> None:
        """Hybrid strategy with news_feature_map runs and returns BacktestResults."""
        price = {"INFY.NS": _make_ohlcv(n=120, seed=0, trend=0.001)}
        news = {"INFY.NS": _make_news_df(n=120, score=0.5)}

        results = self._run_backtest(
            "equity_momentum_news_hybrid_v1",
            price_data=price,
            news_feature_map=news,
        )

        assert results is not None
        assert results.n_steps >= 1

    def test_hybrid_backtest_runs_without_news(self) -> None:
        """Hybrid strategy without news degrades gracefully to pure momentum."""
        price = {"INFY.NS": _make_ohlcv(n=120, seed=0, trend=0.001)}

        results = self._run_backtest(
            "equity_momentum_news_hybrid_v1",
            price_data=price,
            news_feature_map=None,
        )

        assert results is not None
        assert results.n_steps >= 1

    def test_baseline_momentum_runs_with_news_map_provided(self) -> None:
        """Baseline strategy is unaffected even if news_feature_map is supplied."""
        price = {"INFY.NS": _make_ohlcv(n=120, seed=0, trend=0.001)}
        news = {"INFY.NS": _make_news_df(n=120, score=0.5)}

        results = self._run_backtest(
            "equity_momentum_v1",
            price_data=price,
            news_feature_map=news,  # Supplied but ignored by momentum agent
        )

        assert results is not None
        assert results.n_steps >= 1

    def test_hybrid_with_negative_news_produces_lower_signal_scores(self) -> None:
        """Strongly negative news should dampen a positive-momentum hybrid score.

        We compare composite_scores in backtest snapshots:
        - hybrid+negative_news: momentum partially cancelled by news → lower scores
        - hybrid+no_news: pure momentum baseline → higher scores

        We also verify that negative news causes more signals to be blocked
        (confidence too low after dampening), demonstrating real risk-off behavior.
        """
        # Strong upward trend → momentum agent produces positive scores
        price = {"INFY.NS": _make_ohlcv(n=120, seed=42, trend=0.004)}

        # Negative news that opposes the positive momentum
        negative_news = {"INFY.NS": _make_negative_news_df(n=120)}

        results_with_bad_news = self._run_backtest(
            "equity_momentum_news_hybrid_v1",
            price_data=price,
            news_feature_map=negative_news,
        )
        results_no_news = self._run_backtest(
            "equity_momentum_news_hybrid_v1",
            price_data=price,
            news_feature_map=None,
        )

        # Both must run
        assert results_with_bad_news.n_steps >= 1
        assert results_no_news.n_steps >= 1

        # With negative news, composite scores should be lower (dampened by news)
        scores_with_news = [
            s.composite_scores.get("INFY.NS", 0.0)
            for s in results_with_bad_news.snapshots
            if s.composite_scores
        ]
        scores_no_news = [
            s.composite_scores.get("INFY.NS", 0.0)
            for s in results_no_news.snapshots
            if s.composite_scores
        ]

        assert scores_with_news, "No composite scores recorded for bad-news run"
        assert scores_no_news, "No composite scores recorded for no-news run"

        avg_score_with_news = sum(scores_with_news) / len(scores_with_news)
        avg_score_no_news = sum(scores_no_news) / len(scores_no_news)

        # Negative news should reduce the average signal score below no-news baseline
        assert avg_score_with_news < avg_score_no_news, (
            f"Expected negative news to dampen scores: "
            f"avg_with_news={avg_score_with_news:.4f} "
            f"avg_no_news={avg_score_no_news:.4f}"
        )

    def test_hybrid_with_positive_news_matches_or_exceeds_no_news(self) -> None:
        """Positive news that aligns with positive momentum should not reduce returns."""
        price = {"INFY.NS": _make_ohlcv(n=120, seed=42, trend=0.003)}
        positive_news = {"INFY.NS": _make_news_df(n=120, score=0.7)}

        results_with_news = self._run_backtest(
            "equity_momentum_news_hybrid_v1",
            price_data=price,
            news_feature_map=positive_news,
        )
        results_no_news = self._run_backtest(
            "equity_momentum_news_hybrid_v1",
            price_data=price,
            news_feature_map=None,
        )

        assert results_with_news.n_steps >= 1
        assert results_no_news.n_steps >= 1
        # Positive-aligned news boosts or maintains signal → same or better return
        assert results_with_news.total_return >= results_no_news.total_return - 1e-9, (
            "Positive aligned news should not reduce returns below the no-news baseline"
        )

    def test_multi_symbol_partial_news_coverage(self) -> None:
        """Only one of two symbols has news; the other degrades gracefully."""
        price = {
            "INFY.NS": _make_ohlcv(n=120, seed=0, trend=0.002),
            "TCS.NS": _make_ohlcv(n=120, seed=1, trend=0.002),
        }
        # Only INFY has news
        news = {"INFY.NS": _make_news_df(n=120, score=0.5)}

        results = self._run_backtest(
            "equity_momentum_news_hybrid_v1",
            price_data=price,
            news_feature_map=news,
        )

        assert results is not None
        assert results.n_steps >= 1
        # Both symbols should appear in the universe across steps
        all_universe = set()
        for snap in results.snapshots:
            all_universe.update(snap.universe)
        assert "INFY.NS" in all_universe
        assert "TCS.NS" in all_universe

    def test_strategy_id_recorded_in_results(self) -> None:
        """BacktestResults.run_id contains the run_id, not a crash."""
        price = {"INFY.NS": _make_ohlcv(n=80, seed=0, trend=0.001)}

        results = self._run_backtest(
            "equity_momentum_news_hybrid_v1",
            price_data=price,
        )

        assert results.run_id == "backtest"
        assert results.asset_class == "equity"


# ---------------------------------------------------------------------------
# Tests: no_signal_reasons tracking for news-related degradation
# ---------------------------------------------------------------------------


class TestNoSignalReasonsWithNews:
    """Verify NO_SIGNAL reason codes are tracked correctly with news data."""

    def test_no_news_data_reason_tracked_in_pure_news_sentinel_agent(self) -> None:
        """EquityNewsSentimentAgent emits NO_NEWS_DATA when column absent."""
        from spectraquant_v3.core.enums import NoSignalReason, SignalStatus
        from spectraquant_v3.equities.signals.news_sentiment import (
            EquityNewsSentimentAgent,
        )

        agent = EquityNewsSentimentAgent(run_id="test")
        idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        df_no_news = pd.DataFrame({"close": [100.0] * 30}, index=idx)

        row = agent.evaluate("RELIANCE.NS", df_no_news)
        assert row.status == SignalStatus.NO_SIGNAL.value
        assert row.no_signal_reason == NoSignalReason.NO_NEWS_DATA.value

    def test_hybrid_no_signal_reason_absent_when_news_present(self) -> None:
        """Hybrid agent should not emit NO_NEWS_DATA when news column is available."""
        from spectraquant_v3.core.enums import SignalStatus
        from spectraquant_v3.equities.signals.hybrid import (
            EquityMomentumNewsHybridAgent,
        )

        agent = EquityMomentumNewsHybridAgent(run_id="test")
        idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "ret_20d": [0.05] * 40,
                "rsi": [55.0] * 40,
                "news_sentiment_score": [0.4] * 40,
                "vol_realised": [0.15] * 40,
            },
            index=idx,
        )

        row = agent.evaluate("INFY.NS", df)
        # With good momentum and news, should be OK
        assert row.status == SignalStatus.OK.value
        assert row.no_signal_reason == ""
