import pandas as pd
from spectraquant.alpha.meta_policy import compute_expert_weights, detect_regime


def test_perf_weighted_caps_and_sum():
    cfg = {"experts": {"list": ["trend", "value"]}, "meta_policy": {"weight_floor": 0.05, "weight_cap": 0.6, "min_trades_for_trust": 20}}
    w = compute_expert_weights("perf_weighted", {"label": "neutral"}, {"trend": {"perf_score": 0.2, "trades": 30}, "value": {"perf_score": 0.1, "trades": 30}}, cfg)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(0.05 <= x <= 0.6 for x in w.values())


def test_regime_detector_stable_labels():
    close = list(range(1, 120))
    reg = detect_regime(pd.DataFrame({"close": close}), {"meta_policy": {"regime": {"trend_fast": 20, "trend_slow": 50, "high_vol_threshold": 0.25}}})
    assert reg["label"] in {"risk_on", "risk_off", "neutral"}
