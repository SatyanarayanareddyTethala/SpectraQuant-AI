from __future__ import annotations

import pandas as pd
import pytest

from spectraquant.qa.quality_gates import (
    QualityGateError,
    run_quality_gates_price_frame,
    get_canonical_price_column,
    detect_split_like_ratio,
)


def _price_frame(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"date": dates, "close": closes})


def _price_frame_with_adj(closes: list[float], adj_closes: list[float]) -> pd.DataFrame:
    """Create a price frame with both close and adj_close columns."""
    if len(closes) != len(adj_closes):
        raise ValueError(f"closes and adj_closes must have same length, got {len(closes)} and {len(adj_closes)}")
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"date": dates, "close": closes, "adj_close": adj_closes})


def test_extreme_return_threshold_allows_30pct_drop() -> None:
    df = _price_frame([100.0, 70.0, 71.0])
    cfg = {"qa": {"max_abs_daily_return": 0.8, "stale_tolerance_minutes": 10000000}}
    run_quality_gates_price_frame(df, ticker="TEST.NS", exchange="NSE", interval="1d", cfg=cfg)


def test_extreme_return_threshold_rejects_500pct_jump() -> None:
    df = _price_frame([100.0, 600.0, 610.0])
    cfg = {"qa": {"max_abs_daily_return": 0.8, "stale_tolerance_minutes": 10000000}}
    with pytest.raises(QualityGateError) as excinfo:
        run_quality_gates_price_frame(df, ticker="TEST.NS", exchange="NSE", interval="1d", cfg=cfg)
    assert any(issue.code == "extreme_return" for issue in excinfo.value.issues)


def test_canonical_column_selection_prefers_adj_close() -> None:
    """Test that canonical column selection prefers adj_close when available."""
    df = _price_frame_with_adj([100.0, 277.25, 278.0], [100.0, 101.0, 101.5])
    cfg = {"qa": {"price_return_source": "auto"}}
    col_name, series = get_canonical_price_column(df, cfg)
    assert col_name == "adj_close"
    assert len(series) == 3


def test_canonical_column_selection_uses_close_when_adj_not_available() -> None:
    """Test that canonical column selection falls back to close when adj_close not available."""
    df = _price_frame([100.0, 101.0, 102.0])
    cfg = {"qa": {"price_return_source": "auto"}}
    col_name, series = get_canonical_price_column(df, cfg)
    assert col_name == "close"
    assert len(series) == 3


def test_canonical_column_selection_force_adj_close() -> None:
    """Test that price_return_source='adj_close' forces adj_close usage."""
    df = _price_frame_with_adj([100.0, 277.25, 278.0], [100.0, 101.0, 101.5])
    cfg = {"qa": {"price_return_source": "adj_close"}}
    col_name, series = get_canonical_price_column(df, cfg)
    assert col_name == "adj_close"


def test_canonical_column_selection_force_adj_close_fails_when_not_available() -> None:
    """Test that forcing adj_close fails when it's not available."""
    df = _price_frame([100.0, 101.0, 102.0])
    cfg = {"qa": {"price_return_source": "adj_close"}}
    with pytest.raises(ValueError, match="adj_close requested but not available"):
        get_canonical_price_column(df, cfg)


def test_corporate_action_mismatch_detected_as_warn() -> None:
    """Test that corporate action mismatches (extreme close, normal adj_close) are detected as WARN."""
    # Simulate AHLUCONT.NS scenario: close jumps ~165% but adj_close is normal
    df = _price_frame_with_adj(
        [104.55, 277.25, 278.0],  # Close with corporate action jump
        [104.55, 105.0, 105.5],  # Adj_close is normal (no jump)
    )
    cfg = {
        "qa": {
            "max_abs_daily_return": 0.8,
            "stale_tolerance_minutes": 10000000,
            "price_return_source": "auto",  # Will use adj_close
        }
    }
    # Should NOT raise because adj_close (canonical) is normal
    issues = run_quality_gates_price_frame(df, ticker="AHLUCONT.NS", exchange="NSE", interval="1d", cfg=cfg)
    # No FAIL issues, only WARN if any
    assert not any(issue.severity == "FAIL" for issue in issues)


def test_extreme_return_fail_when_both_series_extreme() -> None:
    """Test that extreme returns FAIL when both close and adj_close are extreme."""
    # Both close and adj_close have extreme jumps
    df = _price_frame_with_adj(
        [100.0, 600.0, 610.0],  # Close with extreme jump
        [100.0, 580.0, 590.0],  # Adj_close also extreme
    )
    cfg = {
        "qa": {
            "max_abs_daily_return": 0.8,
            "stale_tolerance_minutes": 10000000,
            "price_return_source": "auto",
        }
    }
    with pytest.raises(QualityGateError) as excinfo:
        run_quality_gates_price_frame(df, ticker="TEST.NS", exchange="NSE", interval="1d", cfg=cfg)
    assert any(issue.code == "extreme_return" for issue in excinfo.value.issues)


def test_lenient_mode_downgrades_extreme_return_to_warn() -> None:
    """Test that lenient mode downgrades extreme returns to WARN."""
    df = _price_frame([100.0, 600.0, 610.0])
    cfg = {
        "qa": {
            "max_abs_daily_return": 0.8,
            "stale_tolerance_minutes": 10000000,
            "mode": "lenient",
        }
    }
    issues = run_quality_gates_price_frame(df, ticker="TEST.NS", exchange="NSE", interval="1d", cfg=cfg)
    # Should have extreme_return issue but as WARN, not FAIL
    extreme_issues = [i for i in issues if i.code == "extreme_return"]
    assert len(extreme_issues) > 0
    assert all(i.severity == "WARN" for i in extreme_issues)


def test_strict_mode_fails_on_extreme_return() -> None:
    """Test that strict mode (default) fails on extreme returns."""
    df = _price_frame([100.0, 600.0, 610.0])
    cfg = {
        "qa": {
            "max_abs_daily_return": 0.8,
            "stale_tolerance_minutes": 10000000,
            "mode": "strict",
        }
    }
    with pytest.raises(QualityGateError) as excinfo:
        run_quality_gates_price_frame(df, ticker="TEST.NS", exchange="NSE", interval="1d", cfg=cfg)
    assert any(issue.code == "extreme_return" and issue.severity == "FAIL" for issue in excinfo.value.issues)


def test_split_like_helper_matches_composite_factor() -> None:
    matched = detect_split_like_ratio(2.651, [2, 3, 4, 5, 10, 20], 0.03)
    assert matched is not None


def test_split_like_extreme_both_close_and_adj_warns_in_strict_mode() -> None:
    """AHLUCONT-like case: both close and adj_close jump by ~2.65x due to unadjusted source data."""
    df = _price_frame_with_adj([104.11, 276.09, 277.00], [104.11, 276.09, 277.00])
    cfg = {
        "qa": {
            "mode": "strict",
            "price_return_source": "auto",
            "max_abs_daily_return": 0.8,
            "split_like_enabled": True,
            "split_like_factors": [2, 3, 4, 5, 10, 20],
            "split_like_tolerance": 0.03,
            "stale_tolerance_minutes": 10000000,
        }
    }

    issues = run_quality_gates_price_frame(df, ticker="AHLUCONT.NS", exchange="NSE", interval="1d", cfg=cfg)

    split_like = [issue for issue in issues if issue.code == "corporate_action_split_like"]
    assert split_like
    assert all(issue.severity == "WARN" for issue in split_like)
    assert not any(issue.severity == "FAIL" for issue in issues)


def test_non_split_like_extreme_still_fails_in_strict_mode() -> None:
    """Extreme move not near a split-like ratio must remain FAIL in strict mode."""
    df = _price_frame_with_adj([100.0, 222.0, 223.0], [100.0, 222.0, 223.0])
    cfg = {
        "qa": {
            "mode": "strict",
            "price_return_source": "auto",
            "max_abs_daily_return": 0.8,
            "split_like_enabled": True,
            "split_like_factors": [2, 3, 4, 5, 10, 20],
            "split_like_tolerance": 0.03,
            "stale_tolerance_minutes": 10000000,
        }
    }

    with pytest.raises(QualityGateError) as excinfo:
        run_quality_gates_price_frame(df, ticker="TEST.NS", exchange="NSE", interval="1d", cfg=cfg)

    assert any(issue.code == "extreme_return" and issue.severity == "FAIL" for issue in excinfo.value.issues)


def test_flatline_strict_warn_when_illiquid_or_no_volume() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 101.0, 102.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            "volume": [1000, 1200, 900, 1100, 0, 0, 0, 0],
        }
    )
    cfg = {"qa": {"flatline_window": 5, "mode": "strict", "stale_tolerance_minutes": 10000000, "min_price_rows": 1}}

    issues = run_quality_gates_price_frame(df, ticker="FEL.NS", exchange="NSE", interval="1d", cfg=cfg)

    flatline = [i for i in issues if i.code == "flatline_window"]
    assert flatline
    assert all(i.severity == "WARN" for i in flatline)
    assert all(i.details.get("remediation_hint") is not None for i in flatline)
    assert all(i.details.get("column_used") == "close" for i in flatline)


def test_flatline_strict_fail_when_volume_positive() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 101.0, 102.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            "volume": [1000, 1200, 900, 1100, 600, 650, 700, 750],
        }
    )
    cfg = {"qa": {"flatline_window": 5, "mode": "strict", "stale_tolerance_minutes": 10000000, "min_price_rows": 1}}

    with pytest.raises(QualityGateError) as excinfo:
        run_quality_gates_price_frame(df, ticker="FEL.NS", exchange="NSE", interval="1d", cfg=cfg)

    assert any(issue.code == "flatline_window" and issue.severity == "FAIL" for issue in excinfo.value.issues)


def test_flatline_lenient_is_warn() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 101.0, 102.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            "volume": [1000, 1200, 900, 1100, 600, 650, 700, 750],
        }
    )
    cfg = {"qa": {"flatline_window": 5, "mode": "lenient", "stale_tolerance_minutes": 10000000, "min_price_rows": 1}}

    issues = run_quality_gates_price_frame(df, ticker="FEL.NS", exchange="NSE", interval="1d", cfg=cfg)

    flatline = [i for i in issues if i.code == "flatline_window"]
    assert flatline
    assert all(i.severity == "WARN" for i in flatline)


def test_flatline_fel_ns_regression_adj_close_window_warns() -> None:
    dates = pd.to_datetime([
        "2026-02-10",
        "2026-02-11",
        "2026-02-12",
        "2026-02-13",
        "2026-02-16",
        "2026-02-17",
        "2026-02-18",
    ], utc=True)
    df = pd.DataFrame(
        {
            "date": dates,
            "close": [0.46, 0.45, 0.44, 0.44, 0.44, 0.44, 0.44],
            "adj_close": [0.46, 0.45, 0.44, 0.44, 0.44, 0.44, 0.44],
            "volume": [150000, 120000, 0, 0, 0, 0, 0],
        }
    )
    cfg = {
        "qa": {
            "flatline_window": 5,
            "mode": "strict",
            "price_return_source": "auto",
            "stale_tolerance_minutes": 10000000,
            "min_price_rows": 1,
        }
    }

    issues = run_quality_gates_price_frame(df, ticker="FEL.NS", exchange="NSE", interval="1d", cfg=cfg)

    flatline = [i for i in issues if i.code == "flatline_window"]
    assert flatline
    assert all(i.severity == "WARN" for i in flatline)
    assert all(i.details.get("column_used") == "adj_close" for i in flatline)
