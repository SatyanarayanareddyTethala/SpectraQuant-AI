from __future__ import annotations

import math

import pandas as pd
from typer.testing import CliRunner

from spectraquant_v3.backtest.engine import BacktestEngine
from spectraquant_v3.backtest.results import RebalanceSnapshot
from spectraquant_v3.cli.main import app
from spectraquant_v3.core.enums import AssetClass
from spectraquant_v3.core.schema import SymbolRecord
from spectraquant_v3.crypto.features.engine import compute_features
from spectraquant_v3.crypto.news.news_fetcher import CoinDeskRSSAdapter, CryptoPanicAdapter, fetch_articles
from spectraquant_v3.crypto.news.news_normalizer import normalize_article_payload, validate_article_schema
from spectraquant_v3.crypto.news.news_store import NewsStore
from spectraquant_v3.crypto.signals.cross_sectional_momentum import CryptoCrossSectionalMomentumAgent
from spectraquant_v3.crypto.universe.universe_engine import UniverseEngine
from spectraquant_v3.crypto.symbols.registry import CryptoSymbolRegistry
from spectraquant_v3.experiments.experiment_manager import ExperimentManager
from spectraquant_v3.experiments.run_tracker import RunTracker
from spectraquant_v3.strategies.allocators.rank_vol_target_allocator import RankVolTargetAllocator


class _FakeCryptoPanicProvider:
    def _get(self, _path: str, params: dict):
        if params["page"] == 1:
            return {
                "results": [
                    {"id": "cp_2", "title": "ETH ETF update", "url": "https://cp/2", "published_at": "2024-01-02T00:00:00Z", "currencies": [{"code": "ETH"}]},
                    {"id": "cp_1", "title": "BTC listing rumor", "url": "https://cp/1", "published_at": "2024-01-01T00:00:00Z", "currencies": [{"code": "BTC"}]},
                ],
                "next": "more",
            }
        return {"results": [], "next": None}


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


class _FakeRSSSession:
    def __init__(self, xml: str):
        self.xml = xml

    def get(self, _url: str, timeout: int = 20):
        assert timeout == 20
        return _FakeResponse(self.xml)


def _feature_row(ret_5: float, ret_20: float, ret_60: float, ret_120: float) -> pd.DataFrame:
    return pd.DataFrame({"ret_5d": [ret_5], "ret_20d": [ret_20], "ret_60d": [ret_60], "ret_120d": [ret_120]})


def _ohlcv_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=idx,
    )


def _crypto_cfg() -> dict:
    return {
        "crypto": {
            "symbols": ["BTC", "ETH", "USDT", "WBTC", "DOGE"],
            "universe_mode": "static",
            "exchanges": ["binance", "coinbase"],
            "universe_filters": {
                "exclude_stablecoins": True,
                "exclude_wrapped_assets": True,
                "min_market_cap_usd": 500,
                "min_daily_volume_usd": 100,
                "min_listing_age_days": 30,
                "require_exchange_coverage": True,
                "required_exchanges": ["binance", "coinbase"],
                "top_market_cap": {"enabled": True, "limit": 1},
                "collect_all_fail_reasons": True,
            },
        }
    }


def _registry() -> CryptoSymbolRegistry:
    reg = CryptoSymbolRegistry()
    for sym in ["BTC", "ETH", "USDT", "WBTC", "DOGE"]:
        reg.register(SymbolRecord(sym, AssetClass.CRYPTO, provider_symbol=f"{sym}/USDT"))
    return reg


def test_universe_engine_filter_paths_and_top_market_cap_reasons():
    engine = UniverseEngine(_crypto_cfg(), _registry())
    rows = engine.evaluate(
        market_data={
            "BTC": {"market_cap_usd": 1000, "volume_24h_usd": 300, "age_days": 100, "exchanges": ["binance", "coinbase"]},
            "ETH": {"market_cap_usd": 900, "volume_24h_usd": 200, "age_days": 90, "exchanges": ["binance", "coinbase"]},
            "USDT": {"market_cap_usd": 1200, "volume_24h_usd": 500, "age_days": 1000, "is_stablecoin": True, "exchanges": ["binance", "coinbase"]},
            "WBTC": {"market_cap_usd": 800, "volume_24h_usd": 200, "age_days": 180, "is_wrapped": True, "exchanges": ["binance", "coinbase"]},
            "DOGE": {"market_cap_usd": 400, "volume_24h_usd": 50, "age_days": 10, "exchanges": ["binance"]},
        }
    )

    by_symbol = {r.canonical_symbol: r for r in rows}
    assert by_symbol["BTC"].included is True
    assert by_symbol["ETH"].included is False and "outside_top_market_cap" in by_symbol["ETH"].fail_reasons
    assert "stablecoin_excluded" in by_symbol["USDT"].fail_reasons
    assert "wrapped_asset_excluded" in by_symbol["WBTC"].fail_reasons
    assert "below_min_market_cap" in by_symbol["DOGE"].fail_reasons
    assert "below_min_daily_volume" in by_symbol["DOGE"].fail_reasons
    assert "below_min_listing_age" in by_symbol["DOGE"].fail_reasons
    assert "missing_exchange_coverage:coinbase" in by_symbol["DOGE"].fail_reasons


def test_news_ingestion_normalization_dedupe_and_schema(tmp_path):
    cp_articles = fetch_articles(CryptoPanicAdapter(_FakeCryptoPanicProvider()), max_pages=2)
    rss_xml = """
    <rss><channel>
      <item><guid>rss_2</guid><title>SOL fork details</title><link>https://rss/2</link><pubDate>2024-01-03T00:00:00Z</pubDate></item>
      <item><guid>rss_1</guid><title>ADA upgrade timeline</title><link>https://rss/1</link><pubDate>2024-01-01T00:00:00Z</pubDate></item>
    </channel></rss>
    """
    rss_articles = fetch_articles(CoinDeskRSSAdapter(session=_FakeRSSSession(rss_xml)))

    merged = [
        normalize_article_payload(row, source_name="cryptopanic", sentiment_score=0.1)
        for row in cp_articles
    ] + [
        normalize_article_payload(row, source_name="coindesk", sentiment_score=-0.2)
        for row in rss_articles
    ]

    for article in merged:
        validate_article_schema(article)

    store = NewsStore(tmp_path / "store")
    duplicate_first = merged + [dict(merged[0])]
    jsonl_path = store.write_jsonl("BTC", duplicate_first)
    assert len(jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == len(merged)


def test_feature_merge_time_alignment_no_lookahead_and_missing_news_fallback():
    ohlcv = _ohlcv_frame()
    news = pd.DataFrame(
        {
            "sentiment": [0.9, -0.5],
            "symbol": ["BTC", "BTC"],
        },
        index=pd.to_datetime(["2024-01-01T00:30:00Z", "2024-01-01T02:30:00Z"]),
    )
    features = compute_features(ohlcv, symbol="BTC", news_df=news, news_merge_tolerance="1d")

    # 01:00 bar sees only 00:30 news. 02:00 bar must not see 02:30 future news.
    assert features.loc[pd.Timestamp("2024-01-01T01:00:00"), "news_sentiment_1h"] == 0.9
    assert pd.isna(features.loc[pd.Timestamp("2024-01-01T02:00:00"), "news_sentiment_1h"])

    features_no_news = compute_features(ohlcv, symbol="BTC", news_df=None)
    assert features_no_news["news_sentiment_24h"].isna().all()


def test_cross_sectional_ranking_zscore_topn_and_stable_tiebreak():
    agent = CryptoCrossSectionalMomentumAgent(run_id="r1", top_n=2)
    feature_map = {
        "ETH": _feature_row(0.02, 0.06, 0.08, 0.1),
        "BTC": _feature_row(0.03, 0.06, 0.08, 0.1),
        "ADA": _feature_row(0.03, 0.06, 0.08, 0.1),
    }
    metrics = agent.evaluate_cross_section(feature_map)

    # BTC/ADA tie on returns -> alphabetical tie-break.
    assert metrics["ADA"]["rank"] == 1
    assert metrics["BTC"]["rank"] == 2
    assert metrics["ETH"]["rank"] == 3

    z = pd.Series([metrics[s]["normalized_score"] for s in ["ADA", "BTC", "ETH"]])
    assert abs(z.mean()) < 1e-12


def test_allocator_rank_proportional_vol_target_and_weight_caps():
    allocator = RankVolTargetAllocator(target_vol=0.10, max_weight=0.25, min_tradable_weight=0.12)
    final, diag = allocator.allocate(
        {
            "A": {"rank": 1, "confidence": 1.0, "vol": 0.20},
            "B": {"rank": 2, "confidence": 0.6, "vol": 0.40},
            "C": {"rank": 3, "confidence": 0.1, "vol": 0.30},
        }
    )
    assert diag["stage_base"]["A"] > diag["stage_base"]["B"] > diag["stage_base"]["C"]
    assert "C" in diag["dropped_symbols"]
    assert all(abs(w) <= 0.25 + 1e-12 for w in final.values())


def test_backtest_turnover_cost_deduction_and_metrics_integrity():
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    price = pd.DataFrame({"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1.01, 1.02, 1.03], "volume": [1, 1, 1, 1]}, index=idx)
    cfg = {"portfolio": {"allocator": "equal_weight"}, "crypto": {"signals": {}}, "backtest": {"target_vol": 0.2}}
    engine = BacktestEngine(cfg=cfg, asset_class="crypto", price_data={"BTC": price}, min_in_sample_periods=2, commission=10, slippage=0, spread=0)

    turnover, cost = engine._compute_turnover_and_cost(prev_weights={"BTC": 0.0}, target_weights={"BTC": 0.5})
    assert turnover == 0.5
    assert cost == 0.0005

    snapshots = [
        RebalanceSnapshot(date="2024-01-03T00:00:00+00:00", universe=["BTC"], signals_ok=1, signals_nosig=0, policy_passed=1, policy_blocked=0, allocations={"BTC": 0.5}, portfolio_value=1.01, step_return=0.01, gross_return=0.011, net_return=0.01, turnover=0.5, positions_count=1, exposure=0.5),
        RebalanceSnapshot(date="2024-01-04T00:00:00+00:00", universe=["BTC"], signals_ok=1, signals_nosig=0, policy_passed=1, policy_blocked=0, allocations={"BTC": 0.5}, portfolio_value=1.02, step_return=0.01, gross_return=0.01, net_return=0.01, turnover=0.1, positions_count=1, exposure=0.5),
    ]
    results = engine._compile_results(snapshots, [0.01, 0.01])
    summary = results.to_dict()
    assert summary["turnover"] == 0.3
    assert summary["avg_positions"] == 1.0
    assert "exposure" in summary and math.isfinite(summary["exposure"])


def test_experiment_tracking_persists_metadata_and_compare_fields(tmp_path):
    store_dir = tmp_path / "experiments"
    tracker = RunTracker("exp_det", "crypto_cross_sectional_momentum_v1", dataset_version="fixture_v1", config={"a": 1})
    tracker.record_metrics({"sharpe": 1.2, "turnover": 0.25, "cagr": 0.18})
    tracker.record_artefact("report", "reports/backtest.json")

    from spectraquant_v3.experiments.result_store import ResultStore

    paths = tracker.save(ResultStore(store_dir))
    assert set(paths) == {"config", "metrics"}

    manager = ExperimentManager(store_dir)
    rows = manager.compare_experiments(["exp_det"])
    assert rows[0]["dataset_version"] == "fixture_v1"
    assert rows[0]["config_hash"]
    assert rows[0]["sharpe"] == 1.2
    assert rows[0]["turnover"] == 0.25


def test_cli_integration_exact_command_path_for_final_goal(monkeypatch, tmp_path):
    runner = CliRunner()
    captured: dict[str, str] = {}

    class DummyResult:
        def summary(self):
            return "ok"

        def write(self, _output_dir: str):
            path = tmp_path / "bt.json"
            path.write_text("{}", encoding="utf-8")
            return path

        def summary_dict(self):
            return {"sharpe": 1.0, "annualised_return": 0.1, "annualised_volatility": 0.2, "max_drawdown": -0.1, "turnover": 0.2, "win_rate": 0.6}

    class DummyEngine:
        def __init__(self, **kwargs):
            captured["strategy"] = kwargs["strategy_id"]

        def run(self):
            return DummyResult()

    class DummyCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def read_parquet(self, _sym):
            return pd.DataFrame({"close": [1, 2], "open": [1, 2], "high": [1, 2], "low": [1, 2], "volume": [1, 2]}, index=pd.date_range("2024-01-01", periods=2, tz="UTC"))

    monkeypatch.setattr("spectraquant_v3.backtest.engine.BacktestEngine", DummyEngine)
    monkeypatch.setattr("spectraquant_v3.core.cache.CacheManager", DummyCache)
    monkeypatch.setattr("spectraquant_v3.core.config.get_crypto_config", lambda *_: {"crypto": {"symbols": ["BTC"], "signals": {}}, "portfolio": {"allocator": "equal_weight"}})
    monkeypatch.setattr("spectraquant_v3.experiments.experiment_manager.ExperimentManager.run_experiment", lambda *args, **kwargs: {"ok": True})

    result = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "--asset-class",
            "crypto",
            "--strategy",
            "crypto_cross_sectional_momentum_v1",
            "--symbols",
            "BTC",
            "--run-id",
            "bt_det",
        ],
    )
    assert result.exit_code == 0
    assert captured["strategy"] == "crypto_cross_sectional_momentum_v1"
