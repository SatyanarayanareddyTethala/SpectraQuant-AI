"""Unit tests for the new intelligence/pricing/governance modules.

Covers:
  - model_selector: determinism and correct output per input combination
  - predictions: new explainability columns present in output schema
  - event_classifier: basic keyword-based classification
  - entity_linker: ticker linkage
  - target_engine: bull/base/bear scenarios
  - downside_engine: VaR/CVaR computation
  - governance/prediction_log: record writing and reading
  - news/schema: EnrichedArticle TypedDict present with expected keys
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# ModelSelector – determinism and correctness
# ---------------------------------------------------------------------------

class TestModelSelector:
    """src/spectraquant/intelligence/model_selector.py"""

    def _make_selector(self):
        from spectraquant.intelligence.model_selector import ModelSelector
        return ModelSelector()

    def test_same_inputs_produce_same_output(self):
        """ModelSelector must be deterministic given the same inputs."""
        selector = self._make_selector()
        ctx = {"magnitude": 0.5, "competitor_shock": 0.1, "event_type": "earnings",
               "uncertainty": 0.3}
        result1 = selector.select(ctx, regime="TRENDING", vol_state=0.015)
        result2 = selector.select(ctx, regime="TRENDING", vol_state=0.015)
        assert result1 == result2

    def test_panic_regime_returns_volatility_model(self):
        from spectraquant.intelligence.model_selector import AnalysisModel, ModelSelector
        selector = ModelSelector()
        model = selector.select({}, regime="PANIC", vol_state=0.01)
        assert model == AnalysisModel.VOLATILITY

    def test_high_vol_state_returns_volatility_model(self):
        from spectraquant.intelligence.model_selector import AnalysisModel, ModelSelector
        selector = ModelSelector()
        model = selector.select({}, regime="TRENDING", vol_state=0.40)
        assert model == AnalysisModel.VOLATILITY

    def test_strong_news_with_low_uncertainty_returns_event_drift(self):
        from spectraquant.intelligence.model_selector import AnalysisModel, ModelSelector
        selector = ModelSelector()
        ctx = {"magnitude": 0.8, "uncertainty": 0.2, "event_type": "earnings",
               "competitor_shock": 0.0}
        model = selector.select(ctx, regime="TRENDING", vol_state=0.01)
        assert model == AnalysisModel.EVENT_DRIFT

    def test_no_news_trending_returns_momentum(self):
        from spectraquant.intelligence.model_selector import AnalysisModel, ModelSelector
        selector = ModelSelector()
        model = selector.select(None, regime="TRENDING", vol_state=0.01)
        assert model == AnalysisModel.MOMENTUM

    def test_no_news_choppy_returns_mean_reversion(self):
        from spectraquant.intelligence.model_selector import AnalysisModel, ModelSelector
        selector = ModelSelector()
        model = selector.select(None, regime="CHOPPY", vol_state=0.01)
        assert model == AnalysisModel.MEAN_REVERSION

    def test_peer_shock_no_direct_news_returns_peer_relative(self):
        from spectraquant.intelligence.model_selector import AnalysisModel, ModelSelector
        selector = ModelSelector()
        ctx = {"magnitude": 0.05, "competitor_shock": 0.50, "event_type": "",
               "uncertainty": 0.5}
        model = selector.select(ctx, regime="TRENDING", vol_state=0.01)
        assert model == AnalysisModel.PEER_RELATIVE

    def test_all_analysis_models_are_valid_enum_values(self):
        from spectraquant.intelligence.model_selector import AnalysisModel
        expected = {"EVENT_DRIFT", "MOMENTUM", "MEAN_REVERSION", "VOLATILITY",
                    "PEER_RELATIVE", "NO_TRADE"}
        assert {m.value for m in AnalysisModel} == expected

    def test_determinism_across_multiple_regimes(self):
        """Same inputs → same output for every regime label."""
        from spectraquant.intelligence.model_selector import ModelSelector
        selector = ModelSelector()
        regimes = ["TRENDING", "CHOPPY", "RISK_ON", "RISK_OFF", "EVENT_DRIVEN"]
        ctx = {"magnitude": 0.1, "competitor_shock": 0.1, "event_type": "macro",
               "uncertainty": 0.7}
        for regime in regimes:
            r1 = selector.select(ctx, regime=regime, vol_state=0.02)
            r2 = selector.select(ctx, regime=regime, vol_state=0.02)
            assert r1 == r2, f"Non-determinism detected for regime={regime}"


# ---------------------------------------------------------------------------
# Prediction frame – schema completeness
# ---------------------------------------------------------------------------

REQUIRED_EXPLAINABILITY_COLS = [
    "reason", "event_type", "analysis_model",
    "expected_move_pct", "target_price", "stop_price",
    "confidence", "risk_score", "news_refs",
]

_TICKERS = ["A.NS", "B.NS", "C.NS"]
_METRICS = {
    t: {"mean_return": 0.005, "volatility": 0.018, "momentum_daily": 0.003, "rsi": 55.0}
    for t in _TICKERS
}
_FACTOR_SCORES = {t: 0.3 for t in _TICKERS}
_DATES = {t: pd.Timestamp("2024-06-01") for t in _TICKERS}


class TestPredictionFrameSchema:
    """Schema completeness tests for build_prediction_frame."""

    def _build(self, **kwargs):
        from spectraquant.core.predictions import build_prediction_frame
        defaults = dict(
            tickers=_TICKERS,
            metrics_by_ticker=_METRICS,
            factor_scores=_FACTOR_SCORES,
            horizon="20d",
            horizon_days=20.0,
            model_version="test-v1",
            factor_set_version="test-fs",
            regime="TRENDING",
            prediction_dates=_DATES,
        )
        defaults.update(kwargs)
        return build_prediction_frame(**defaults)

    def test_explainability_columns_present_without_news(self):
        """All new explainability columns must be present even without news."""
        df = self._build()
        for col in REQUIRED_EXPLAINABILITY_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_explainability_columns_present_with_news(self):
        """All explainability columns present when news context is supplied."""
        news_ctx = {
            "A.NS": {
                "magnitude": 0.6, "sentiment": "positive", "event_type": "earnings",
                "source_rank": 0.9, "recency": 0.8, "uncertainty": 0.2,
                "top_headlines": "A.NS beats Q4 estimates",
                "competitor_shock": 0.1,
                "news_refs": ["https://example.com/a"],
            }
        }
        df = self._build(
            news_context_by_ticker=news_ctx,
            price_by_ticker={"A.NS": 1000.0, "B.NS": 500.0, "C.NS": 200.0},
        )
        for col in REQUIRED_EXPLAINABILITY_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_news_refs_is_list_type(self):
        """news_refs column must contain list values."""
        df = self._build()
        for val in df["news_refs"]:
            assert isinstance(val, list), f"Expected list, got {type(val)}"

    def test_target_price_nonzero_when_price_supplied(self):
        """target_price should be non-zero when last_price is provided."""
        df = self._build(price_by_ticker={t: 1000.0 for t in _TICKERS})
        assert (df["target_price"] != 0).any()

    def test_confidence_in_range(self):
        """confidence must be in [0.05, 0.95]."""
        df = self._build()
        assert (df["confidence"] >= 0.05).all()
        assert (df["confidence"] <= 0.95).all()

    def test_risk_score_in_range(self):
        """risk_score must be in [0, 1]."""
        df = self._build()
        assert (df["risk_score"] >= 0).all()
        assert (df["risk_score"] <= 1).all()

    def test_analysis_model_column_nonempty_strings(self):
        """analysis_model must be non-empty string for all rows."""
        df = self._build()
        assert df["analysis_model"].ne("").all()

    def test_backward_compat_no_new_required_args(self):
        """build_prediction_frame still works with original argument set."""
        from spectraquant.core.predictions import build_prediction_frame
        df = build_prediction_frame(
            tickers=_TICKERS,
            metrics_by_ticker=_METRICS,
            factor_scores=_FACTOR_SCORES,
            horizon="20d",
            horizon_days=20.0,
            model_version="v0",
            factor_set_version="fs0",
            regime="neutral",
            prediction_dates=_DATES,
        )
        # Original columns still present
        assert "expected_return_annual" in df.columns
        assert "predicted_return" in df.columns
        assert "probability" in df.columns


# ---------------------------------------------------------------------------
# EventClassifier – basic keyword classification
# ---------------------------------------------------------------------------

class TestEventClassifier:
    def test_earnings_beat_classified_as_earnings_positive(self):
        from spectraquant.news.event_classifier import EventClassifier
        clf = EventClassifier()
        result = clf.classify({"title": "Infosys earnings beat analyst estimates Q4", "content": ""})
        assert result.event_type == "earnings"
        assert result.sentiment == "positive"

    def test_earnings_miss_classified_as_earnings_negative(self):
        from spectraquant.news.event_classifier import EventClassifier
        clf = EventClassifier()
        result = clf.classify({"title": "TCS quarterly loss: earnings miss consensus", "content": ""})
        assert result.event_type == "earnings"
        assert result.sentiment == "negative"

    def test_empty_article_returns_unknown(self):
        from spectraquant.news.event_classifier import EventClassifier
        clf = EventClassifier()
        result = clf.classify({"title": "", "content": ""})
        assert result.event_type == "unknown"

    def test_magnitude_is_in_valid_range(self):
        from spectraquant.news.event_classifier import EventClassifier
        clf = EventClassifier()
        result = clf.classify({"title": "SEBI fine imposed on broker", "content": ""})
        assert 0.0 <= result.magnitude <= 1.0

    def test_uncertainty_is_in_valid_range(self):
        from spectraquant.news.event_classifier import EventClassifier
        clf = EventClassifier()
        result = clf.classify({"title": "Reliance RBI policy macro rate hike earnings", "content": ""})
        assert 0.0 <= result.uncertainty <= 1.0

    def test_score_source_known_source(self):
        from spectraquant.news.event_classifier import EventClassifier
        rank = EventClassifier.score_source("Reuters")
        assert rank >= 0.9

    def test_score_source_unknown_returns_default(self):
        from spectraquant.news.event_classifier import EventClassifier
        rank = EventClassifier.score_source("SomeRandomBlog")
        assert 0.0 < rank < 1.0

    def test_regulatory_fine_classified_as_regulatory_negative(self):
        from spectraquant.news.event_classifier import EventClassifier
        clf = EventClassifier()
        result = clf.classify({"title": "SEBI fine imposed on HDFC Bank", "content": ""})
        assert result.event_type == "regulatory"

    def test_classify_is_deterministic(self):
        """Same article must produce the same result on every call."""
        from spectraquant.news.event_classifier import EventClassifier
        clf = EventClassifier()
        article = {"title": "Infosys Q4 earnings beat consensus", "content": "Strong results"}
        r1 = clf.classify(article)
        r2 = clf.classify(article)
        assert r1.event_type == r2.event_type
        assert r1.sentiment == r2.sentiment
        assert r1.magnitude == r2.magnitude


# ---------------------------------------------------------------------------
# EntityLinker – ticker linkage
# ---------------------------------------------------------------------------

class TestEntityLinker:
    def _mapping(self):
        return {
            "tickers": ["TCS.NS", "INFY.NS", "RELIANCE.NS"],
            "ticker_to_company": {
                "TCS.NS": "Tata Consultancy Services",
                "INFY.NS": "Infosys",
                "RELIANCE.NS": "Reliance Industries",
            },
            "aliases": {"tata consulting": "TCS.NS"},
        }

    def test_ticker_token_match(self):
        from spectraquant.news.entity_linker import EntityLinker
        linker = EntityLinker(self._mapping())
        result = linker.link({"title": "TCS reports strong Q4 results", "content": ""})
        assert "TCS.NS" in result.tickers

    def test_company_name_match(self):
        from spectraquant.news.entity_linker import EntityLinker
        linker = EntityLinker(self._mapping())
        result = linker.link({"title": "Infosys wins cloud deal", "content": ""})
        assert "INFY.NS" in result.tickers

    def test_alias_match(self):
        from spectraquant.news.entity_linker import EntityLinker
        linker = EntityLinker(self._mapping())
        result = linker.link({"title": "Tata consulting wins government deal", "content": ""})
        assert "TCS.NS" in result.tickers

    def test_no_match_returns_empty(self):
        from spectraquant.news.entity_linker import EntityLinker
        linker = EntityLinker(self._mapping())
        result = linker.link({"title": "Global market update", "content": ""})
        assert result.tickers == []
        assert result.competitors == []

    def test_competitors_resolved_for_matched_ticker(self):
        from spectraquant.news.entity_linker import EntityLinker
        linker = EntityLinker(self._mapping())
        result = linker.link({"title": "TCS earnings beat estimates", "content": ""})
        # TCS is in IT sector; competitors should include INFY, WIPRO, etc.
        assert isinstance(result.competitors, list)


# ---------------------------------------------------------------------------
# TargetEngine – price scenarios
# ---------------------------------------------------------------------------

class TestTargetEngine:
    def test_base_target_correct(self):
        from spectraquant.pricing.target_engine import TargetEngine
        engine = TargetEngine()
        scenarios = engine.build(last_price=1000.0, expected_move=0.05, atr=20.0)
        assert abs(scenarios.base_target - 1050.0) < 0.5

    def test_stop_below_last_price(self):
        from spectraquant.pricing.target_engine import TargetEngine
        engine = TargetEngine()
        scenarios = engine.build(last_price=1000.0, expected_move=0.05, atr=20.0)
        assert scenarios.stop_price < 1000.0

    def test_bull_above_base(self):
        from spectraquant.pricing.target_engine import TargetEngine
        engine = TargetEngine()
        scenarios = engine.build(last_price=1000.0, expected_move=0.05, atr=20.0)
        assert scenarios.bull_target >= scenarios.base_target

    def test_risk_reward_positive_for_upside(self):
        from spectraquant.pricing.target_engine import TargetEngine
        engine = TargetEngine()
        scenarios = engine.build(last_price=1000.0, expected_move=0.05, atr=20.0)
        assert scenarios.risk_reward > 0

    def test_invalid_price_raises(self):
        from spectraquant.pricing.target_engine import TargetEngine
        engine = TargetEngine()
        with pytest.raises(ValueError):
            engine.build(last_price=0.0, expected_move=0.05)

    def test_expected_move_pct_correct(self):
        from spectraquant.pricing.target_engine import TargetEngine
        engine = TargetEngine()
        scenarios = engine.build(last_price=1000.0, expected_move=0.042, atr=20.0)
        assert abs(scenarios.expected_move_pct - 4.2) < 0.01

    def test_build_from_analogs(self):
        from spectraquant.pricing.target_engine import TargetEngine
        engine = TargetEngine()
        returns = [0.02, -0.01, 0.04, 0.06, -0.02, 0.03]
        scenarios = engine.build_from_analogs(1000.0, returns, atr=20.0)
        assert scenarios.base_target > 0


# ---------------------------------------------------------------------------
# DownsideEngine – VaR / CVaR
# ---------------------------------------------------------------------------

class TestDownsideEngine:
    def test_var_95_below_expected_downside(self):
        from spectraquant.pricing.downside_engine import DownsideEngine
        engine = DownsideEngine()
        returns = [-0.03, 0.01, -0.05, 0.02, -0.08, 0.01, -0.12, -0.02, 0.03, 0.01]
        risk = engine.estimate(1000.0, analog_returns=returns)
        # VaR_95 should be <= expected_downside (more extreme)
        assert risk.var_95_pct <= risk.expected_downside_pct

    def test_cvar_95_lte_var_95(self):
        from spectraquant.pricing.downside_engine import DownsideEngine
        engine = DownsideEngine()
        returns = [-0.03, 0.01, -0.05, 0.02, -0.08, 0.01, -0.12, -0.02, 0.03, 0.01]
        risk = engine.estimate(1000.0, analog_returns=returns)
        assert risk.cvar_95_pct <= risk.var_95_pct

    def test_risk_score_in_range(self):
        from spectraquant.pricing.downside_engine import DownsideEngine
        engine = DownsideEngine()
        returns = [-0.03, 0.01, -0.05]
        risk = engine.estimate(1000.0, analog_returns=returns)
        assert 0.0 <= risk.risk_score <= 1.0

    def test_crash_probability_in_range(self):
        from spectraquant.pricing.downside_engine import DownsideEngine
        engine = DownsideEngine()
        returns = [-0.03, -0.15, 0.01, -0.02, -0.11]
        risk = engine.estimate(1000.0, analog_returns=returns)
        assert 0.0 <= risk.crash_probability <= 1.0

    def test_invalid_price_raises(self):
        from spectraquant.pricing.downside_engine import DownsideEngine
        engine = DownsideEngine()
        with pytest.raises(ValueError):
            engine.estimate(0.0)

    def test_implied_vol_fallback_when_no_analogs(self):
        from spectraquant.pricing.downside_engine import DownsideEngine
        engine = DownsideEngine()
        risk = engine.estimate(1000.0, implied_vol=0.25)
        assert risk.var_95_pct < 0


# ---------------------------------------------------------------------------
# GovernanceLogger – prediction record writing / reading
# ---------------------------------------------------------------------------

class TestGovernanceLogger:
    def test_write_and_read_roundtrip(self, tmp_path):
        from spectraquant.governance.prediction_log import GovernanceLogger
        logger = GovernanceLogger(log_dir=tmp_path)
        record = {
            "ticker": "RELIANCE.NS",
            "action": "BUY",
            "reason": "Earnings beat",
            "event_type": "earnings",
            "analysis_model": "EVENT_DRIFT",
            "expected_move_pct": 4.2,
            "target_price": 3025.0,
            "stop_price": 2870.0,
            "confidence": 0.74,
            "risk_score": 0.25,
            "news_refs": ["https://example.com/article/1"],
        }
        logger.write(record)
        records = logger.read_all()
        assert len(records) == 1
        assert records[0]["ticker"] == "RELIANCE.NS"

    def test_log_file_created(self, tmp_path):
        from spectraquant.governance.prediction_log import GovernanceLogger
        logger = GovernanceLogger(log_dir=tmp_path)
        logger.write({"ticker": "X.NS", "action": "HOLD"})
        assert logger.log_path.exists()

    def test_count_increments(self, tmp_path):
        from spectraquant.governance.prediction_log import GovernanceLogger
        logger = GovernanceLogger(log_dir=tmp_path)
        logger.write({"ticker": "A.NS", "action": "BUY"})
        logger.write({"ticker": "B.NS", "action": "SELL"})
        assert logger.count() == 2

    def test_missing_mandatory_keys_still_writes(self, tmp_path):
        """Incomplete records are written (not dropped) even if they fail validation."""
        from spectraquant.governance.prediction_log import GovernanceLogger
        logger = GovernanceLogger(log_dir=tmp_path, validate=True)
        logger.write({"ticker": "X.NS"})  # missing many mandatory keys
        assert logger.count() == 1

    def test_prediction_record_missing_keys(self):
        from spectraquant.governance.prediction_log import PredictionRecord, MANDATORY_KEYS
        rec = PredictionRecord({"ticker": "X.NS", "action": "BUY"})
        missing = rec.missing_keys()
        assert len(missing) > 0
        assert "reason" in missing

    def test_prediction_record_complete(self):
        from spectraquant.governance.prediction_log import PredictionRecord, MANDATORY_KEYS
        rec = PredictionRecord({k: "val" for k in MANDATORY_KEYS})
        assert rec.is_complete()

    def test_write_batch(self, tmp_path):
        from spectraquant.governance.prediction_log import GovernanceLogger
        logger = GovernanceLogger(log_dir=tmp_path)
        records = [{"ticker": f"{i}.NS", "action": "BUY"} for i in range(5)]
        logger.write_batch(records)
        assert logger.count() == 5

    def test_asof_utc_auto_injected(self, tmp_path):
        """asof_utc is injected if not present in the record."""
        from spectraquant.governance.prediction_log import GovernanceLogger
        logger = GovernanceLogger(log_dir=tmp_path)
        logger.write({"ticker": "X.NS"})
        records = logger.read_all()
        assert "asof_utc" in records[0]


# ---------------------------------------------------------------------------
# News schema – EnrichedArticle
# ---------------------------------------------------------------------------

class TestEnrichedArticle:
    def test_enriched_article_has_expected_keys(self):
        """EnrichedArticle must declare the new optional fields."""
        from spectraquant.news.schema import EnrichedArticle
        hints = EnrichedArticle.__annotations__
        for key in ("event_type", "entities", "competitors", "magnitude_score", "source_rank"):
            assert key in hints, f"EnrichedArticle missing field: {key}"

    def test_canonical_article_unchanged(self):
        """CanonicalArticle must still have its original six fields."""
        from spectraquant.news.schema import CanonicalArticle
        hints = CanonicalArticle.__annotations__
        for key in ("title", "description", "content", "source_name",
                    "published_at_utc", "url"):
            assert key in hints, f"CanonicalArticle missing field: {key}"


# ---------------------------------------------------------------------------
# No-silent-fallback contract (news-first mode with no candidates)
# ---------------------------------------------------------------------------

class TestNoSilentFallback:
    def test_news_first_enabled_no_candidates_raises(self, tmp_path):
        """When news-first mode is enabled and there are no candidates,
        NewsUniverseEmptyError must be raised (not a silent fallback)."""
        from spectraquant.cli.main import _resolve_download_tickers, NewsUniverseEmptyError
        import os

        cfg = {
            "news_universe": {"enabled": True},
            "universe": {"tickers": ["X.NS"]},
        }
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(NewsUniverseEmptyError):
                _resolve_download_tickers(cfg, from_news=True)
        finally:
            os.chdir(old_cwd)
