"""Quality gates for data and artifact validation."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from spectraquant.core.time import ensure_datetime_column, is_intraday_horizon
from spectraquant.data.normalize import normalize_price_frame
from spectraquant.core.trading_time import expected_latest_time, is_stale, latest_valid_bar_time
from spectraquant.qa.mode import GateMode, resolve_gate_mode
from spectraquant.core.diagnostics import record_quality_gates
from spectraquant.core.predictions import ANNUAL_RETURN_MAX

logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports/qa")


@dataclass
class Issue:
    stage: str
    severity: str
    code: str
    message: str
    details: Dict[str, str]


class QualityGateError(ValueError):
    def __init__(self, issues: List[Issue]) -> None:
        super().__init__(f"Quality gates failed with {len([i for i in issues if i.severity == 'FAIL'])} failures.")
        self.issues = issues


def _generate_remediation_hints(failures: List[Issue]) -> List[str]:
    """Generate remediation hints for specific failure types."""
    hints = []
    for issue in failures:
        if issue.code == "extreme_return":
            ticker = issue.details.get("ticker", "TICKER")
            hints.append(
                f"• {ticker}: Extreme return detected. "
                f"Consider: (1) Check for data errors in source CSV, "
                f"(2) Use 'qa.mode: lenient' in config.yaml, "
                f"(3) Verify adjusted prices are being used (qa.price_return_source: auto)"
            )
        elif issue.code == "corporate_action_mismatch":
            ticker = issue.details.get("ticker", "TICKER")
            hints.append(
                f"• {ticker}: Corporate action detected in raw close but not adj_close. "
                f"This is typically safe. Set 'qa.price_return_source: auto' to use adjusted prices."
            )
        elif issue.code == "flatline_window":
            ticker = issue.details.get("ticker", "TICKER")
            hints.append(
                f"• {ticker}: Recent prices are flat with non-zero volume. "
                f"Likely bad vendor data. Verify latest bars or switch source for this ticker."
            )
    return hints


def _fail_if_needed(issues: List[Issue], mode: GateMode) -> None:
    if mode.force_pass:
        return
    failures = [i for i in issues if i.severity == "FAIL"]
    if failures:
        failure_payload = [asdict(issue) for issue in failures]
        impacted_tickers = sorted({issue.details.get("ticker") for issue in failures if issue.details.get("ticker")})

        logger.error(
            "Quality gate failures: %s",
            json.dumps(
                {
                    "failure_count": len(failures),
                    "impacted_ticker_count": len(impacted_tickers),
                    "impacted_tickers": impacted_tickers,
                    "failures": failure_payload,
                },
                default=str,
            ),
        )

        remediation_hints = _generate_remediation_hints(failures)
        if remediation_hints:
            logger.info("Remediation hints:")
            for hint in remediation_hints[:10]:
                logger.info(hint)

        record_quality_gates(failure_payload)
        raise QualityGateError(issues)


def _append_issue(issues: List[Issue], stage: str, severity: str, code: str, message: str, **details) -> None:
    issues.append(Issue(stage=stage, severity=severity, code=code, message=message, details=details))


def get_canonical_price_column(df: pd.DataFrame, cfg: Dict) -> tuple[str, pd.Series]:
    """Select the canonical price column for QA checks based on config and availability.
    
    Args:
        df: DataFrame with price data
        cfg: Configuration dictionary containing qa.price_return_source setting
    
    Returns:
        tuple of (column_name, series) where column_name is the selected column
        and series is the numeric price data.
        
    Raises:
        ValueError: If the requested price column is not available in the dataframe.
            - When price_return_source='adj_close' but adj_close column not found
            - When price_return_source='close' but close column not found
            - When no suitable price column (close or adj_close) exists
        
    Priority order (configurable via qa.price_return_source):
    - "auto" (default): adj_close -> close
    - "adj_close": adj_close only
    - "close": close only
    """
    qa_cfg = cfg.get("qa", {})
    source = str(qa_cfg.get("price_return_source", "auto")).lower()
    
    # Normalize column names for case-insensitive matching
    col_map = {col.lower(): col for col in df.columns}
    
    if source == "adj_close":
        # Force adj_close usage
        for candidate in ("adj_close", "adj close"):
            if candidate in col_map:
                col = col_map[candidate]
                series = pd.to_numeric(df[col], errors="coerce")
                return col, series
        raise ValueError("adj_close requested but not available in dataframe")
    
    elif source == "close":
        # Force close usage
        if "close" in col_map:
            col = col_map["close"]
            series = pd.to_numeric(df[col], errors="coerce")
            return col, series
        raise ValueError("close requested but not available in dataframe")
    
    else:  # "auto" or any other value defaults to auto
        # Prefer adj_close if available, otherwise fall back to close
        for candidate in ("adj_close", "adj close"):
            if candidate in col_map:
                col = col_map[candidate]
                series = pd.to_numeric(df[col], errors="coerce")
                logger.debug("Using adjusted close (%s) for QA return calculations", col)
                return col, series
        
        # Fall back to close
        if "close" in col_map:
            col = col_map["close"]
            series = pd.to_numeric(df[col], errors="coerce")
            logger.debug("Using raw close for QA return calculations (adj_close not available)")
            return col, series
        
        raise ValueError("No suitable price column (close or adj_close) found in dataframe")


def detect_split_like_ratio(
    ratio: float,
    factors: list[float],
    tolerance: float,
) -> float | None:
    """Return the matched split factor when ratio resembles a split-like move."""
    if not np.isfinite(ratio) or ratio <= 0:
        return None

    candidate_factors: set[float] = {float(factor) for factor in factors if factor > 0}

    # Some vendors publish split+bonus actions as one combined jump.
    # Add limited composite candidates like 8/3 (2x split plus 1:3 bonus).
    if any(np.isclose(factor, 2.0) for factor in candidate_factors):
        for denominator in sorted(candidate_factors):
            if denominator <= 0 or np.log2(denominator).is_integer():
                continue
            for numerator in (2.0, 4.0, 8.0, 16.0):
                composite = numerator / denominator
                if 1.0 < composite <= 25.0:
                    candidate_factors.add(composite)

    for factor in sorted(candidate_factors):
        if np.isclose(ratio, factor, rtol=tolerance) or np.isclose(ratio, 1.0 / factor, rtol=tolerance):
            return float(factor)
    return None



def run_quality_gates_price_frame(
    df: pd.DataFrame,
    ticker: str,
    exchange: str,
    interval: str,
    cfg: Dict,
) -> List[Issue]:
    issues: List[Issue] = []
    mode = resolve_gate_mode(cfg)
    stage = "price_frame"
    df = normalize_price_frame(df)
    date_index = df.index
    if date_index.duplicated().any():
        _append_issue(issues, stage, "FAIL", "duplicate_dates", "Duplicate dates detected", ticker=ticker)
    if not date_index.is_monotonic_increasing:
        _append_issue(issues, stage, "FAIL", "non_monotonic", "Dates not monotonic", ticker=ticker)
    if (date_index.normalize() == pd.Timestamp("1970-01-01", tz="UTC")).any():
        _append_issue(issues, stage, "FAIL", "epoch_date", "Epoch date detected", ticker=ticker)

    qa_cfg = cfg.get("qa", {})
    max_missing_pct = float(qa_cfg.get("max_missing_pct", 0.2))
    min_rows = int(qa_cfg.get("min_price_rows", 30))
    if len(df) < min_rows:
        _append_issue(
            issues,
            stage,
            "WARN",
            "sparse_history",
            "Sparse price history",
            ticker=ticker,
            rows=str(len(df)),
        )
    critical_cols = [c for c in ("close", "open", "high", "low") if c in df.columns]
    for col in critical_cols:
        missing_pct = df[col].isna().mean()
        if missing_pct > max_missing_pct:
            _append_issue(
                issues,
                stage,
                "FAIL",
                "missingness",
                f"Missingness too high for {col}",
                ticker=ticker,
                missing_pct=str(missing_pct),
            )

    close = pd.to_numeric(df.get("close"), errors="coerce")
    if close.isna().any():
        _append_issue(issues, stage, "FAIL", "nan_close", "Close contains NaNs", ticker=ticker)
    if (close <= 0).any():
        _append_issue(issues, stage, "FAIL", "non_positive_price", "Non-positive prices", ticker=ticker)

    flatline_window = int(qa_cfg.get("flatline_window", 5))
    if flatline_window > 1 and close.nunique() <= 1:
        _append_issue(issues, stage, "FAIL", "flatline", "Flatline prices", ticker=ticker)

    canonical_col = "close"
    canonical_series = close
    try:
        canonical_col, canonical_series = get_canonical_price_column(df, cfg)
    except ValueError as exc:
        logger.warning("Cannot determine canonical price series for %s: %s", ticker, exc)

    if flatline_window > 1:
        tail_prices = pd.to_numeric(canonical_series.tail(flatline_window), errors="coerce").dropna()
        if len(tail_prices) >= flatline_window and tail_prices.nunique() <= 1:
            tail_dates = df.index[-flatline_window:]
            volume_tail = pd.Series(dtype=float)
            if "volume" in df.columns:
                volume_tail = pd.to_numeric(df["volume"].tail(flatline_window), errors="coerce")
            non_zero_volume = bool(not volume_tail.empty and volume_tail.notna().all() and (volume_tail > 0).all())

            recent_trading_normal = True
            if len(tail_dates) >= 2:
                max_gap_days = max((tail_dates[i] - tail_dates[i - 1]).days for i in range(1, len(tail_dates)))
                recent_trading_normal = max_gap_days <= 7

            severity = "WARN"
            if mode.mode == "strict" and non_zero_volume and recent_trading_normal:
                severity = "FAIL"

            details = {
                "ticker": ticker,
                "window_size": str(flatline_window),
                "last_dates": str([str(d) for d in tail_dates.tolist()]),
                "last_prices": str([float(p) for p in tail_prices.tolist()]),
                "unique_count": str(int(tail_prices.nunique())),
                "column_used": canonical_col,
                "remediation_hint": "likely illiquid/suspended/stale; verify exchange activity and vendor bars",
                "recent_trading_normal": str(recent_trading_normal),
            }
            if not volume_tail.empty:
                details["last_volumes"] = str(volume_tail.tolist())

            _append_issue(
                issues,
                stage,
                severity,
                "flatline_window",
                "Flatline in recent window",
                **details,
            )

    # Corporate-action-aware extreme return check
    max_abs_daily_return = float(qa_cfg.get("max_abs_daily_return", 0.8))
    split_like_enabled = bool(qa_cfg.get("split_like_enabled", True))
    split_like_tolerance = float(qa_cfg.get("split_like_tolerance", 0.03))
    split_like_factors = [float(f) for f in qa_cfg.get("split_like_factors", [2, 3, 4, 5, 10, 20])]
    
    # Get the canonical price column for QA checks (prefer adj_close)
    try:
        canonical_returns = canonical_series.pct_change().abs()
        extreme_indices = canonical_returns > max_abs_daily_return
        
        if extreme_indices.any():
            # Found extreme returns in canonical series
            # Check if this might be a corporate action issue
            col_map_lower = {col.lower(): col for col in df.columns}
            has_adj_close = any(c in col_map_lower for c in ("adj_close", "adj close"))
            
            # If we used close but adj_close exists, check if adj_close is normal
            if canonical_col.lower() == "close" and has_adj_close:
                # Get adj_close series
                for adj_candidate in ("adj_close", "adj close"):
                    if adj_candidate in col_map_lower:
                        adj_col = col_map_lower[adj_candidate]
                        adj_series = pd.to_numeric(df[adj_col], errors="coerce")
                        adj_returns = adj_series.pct_change().abs()
                        adj_extreme = adj_returns > max_abs_daily_return
                        
                        # If adj_close is normal where close is extreme, it's likely a corporate action
                        if not adj_extreme[extreme_indices].any():
                            # Corporate action mismatch detected - always WARN
                            extreme_dates = df.index[extreme_indices].tolist()
                            _append_issue(
                                issues,
                                stage,
                                "WARN",
                                "corporate_action_mismatch",
                                "Extreme return detected in raw close but not in adj_close (likely corporate action)",
                                ticker=ticker,
                                dates=str(extreme_dates[:5]),  # Show first 5 dates
                                column_checked="close",
                            )
                            break
                        else:
                            # Both are extreme - this is a real issue unless split-like behavior is detected
                            extreme_dates = df.index[extreme_indices].tolist()
                            prev_prices = canonical_series.shift(1)[extreme_indices].tolist()
                            current_prices = canonical_series[extreme_indices].tolist()
                            returns_vals = canonical_returns[extreme_indices].tolist()

                            severity = "WARN" if mode.mode == "lenient" else "FAIL"
                            issue_code = "extreme_return"
                            issue_message = "Extreme price jump detected in both close and adj_close"
                            issue_details = {
                                "ticker": ticker,
                                "dates": str(extreme_dates[:5]),
                                "prev_prices": str(prev_prices[:5]),
                                "prices": str(current_prices[:5]),
                                "returns": str(returns_vals[:5]),
                                "column_used": canonical_col,
                            }

                            if split_like_enabled and mode.mode != "lenient":
                                ratios = (canonical_series / canonical_series.shift(1))[extreme_indices]
                                split_matches = []
                                for idx, ratio in ratios.items():
                                    matched_factor = detect_split_like_ratio(
                                        ratio=float(ratio),
                                        factors=split_like_factors,
                                        tolerance=split_like_tolerance,
                                    )
                                    if matched_factor is not None:
                                        split_matches.append(
                                            {
                                                "date": str(idx),
                                                "prev_price": float(canonical_series.shift(1).loc[idx]),
                                                "price": float(canonical_series.loc[idx]),
                                                "return": float(canonical_returns.loc[idx]),
                                                "ratio": float(ratio),
                                                "matched_factor": matched_factor,
                                            }
                                        )

                                if split_matches:
                                    severity = "WARN"
                                    issue_code = "corporate_action_split_like"
                                    issue_message = (
                                        "Extreme return appears split-like in canonical price series "
                                        "(possible missing corporate action adjustment in source data)"
                                    )
                                    issue_details["split_like_matches"] = str(split_matches[:5])
                                    issue_details["split_like_tolerance"] = str(split_like_tolerance)
                                    issue_details["split_like_factors"] = str(split_like_factors)

                            _append_issue(issues, stage, severity, issue_code, issue_message, **issue_details)
                        break
            else:
                # Used adj_close or no adj_close available - report as extreme
                extreme_dates = df.index[extreme_indices].tolist()
                prev_prices = canonical_series.shift(1)[extreme_indices].tolist()
                current_prices = canonical_series[extreme_indices].tolist()
                returns_vals = canonical_returns[extreme_indices].tolist()

                severity = "WARN" if mode.mode == "lenient" else "FAIL"
                issue_code = "extreme_return"
                issue_message = "Extreme price jump detected"
                issue_details = {
                    "ticker": ticker,
                    "dates": str(extreme_dates[:5]),
                    "prev_prices": str(prev_prices[:5]),
                    "prices": str(current_prices[:5]),
                    "returns": str(returns_vals[:5]),
                    "column_used": canonical_col,
                }

                if split_like_enabled and mode.mode != "lenient":
                    ratios = (canonical_series / canonical_series.shift(1))[extreme_indices]
                    split_matches = []
                    for idx, ratio in ratios.items():
                        matched_factor = detect_split_like_ratio(
                            ratio=float(ratio),
                            factors=split_like_factors,
                            tolerance=split_like_tolerance,
                        )
                        if matched_factor is not None:
                            split_matches.append(
                                {
                                    "date": str(idx),
                                    "prev_price": float(canonical_series.shift(1).loc[idx]),
                                    "price": float(canonical_series.loc[idx]),
                                    "return": float(canonical_returns.loc[idx]),
                                    "ratio": float(ratio),
                                    "matched_factor": matched_factor,
                                }
                            )

                    if split_matches:
                        severity = "WARN"
                        issue_code = "corporate_action_split_like"
                        issue_message = (
                            "Extreme return appears split-like in canonical price series "
                            "(possible missing corporate action adjustment in source data)"
                        )
                        issue_details["split_like_matches"] = str(split_matches[:5])
                        issue_details["split_like_tolerance"] = str(split_like_tolerance)
                        issue_details["split_like_factors"] = str(split_like_factors)

                _append_issue(issues, stage, severity, issue_code, issue_message, **issue_details)
    except ValueError as exc:
        # No suitable price column found
        logger.warning("Cannot perform extreme return check for %s: %s", ticker, exc)

    min_volume = float(qa_cfg.get("min_volume", 0))
    if "volume" in df.columns and min_volume > 0:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        if (vol < min_volume).any():
            _append_issue(issues, stage, "WARN", "low_volume", "Low volume detected", ticker=ticker)

    if interval.lower().endswith("m"):
        stale_tolerance = int(qa_cfg.get("stale_tolerance_minutes", 30))
        now_utc = pd.Timestamp.utcnow()
        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")
        else:
            now_utc = now_utc.tz_convert("UTC")
        latest_ts = latest_valid_bar_time(exchange, df, interval)
        if is_stale(exchange, interval, latest_ts, now_utc, stale_tolerance):
            expected = expected_latest_time(exchange, interval, now_utc)
            _append_issue(
                issues,
                stage,
                "FAIL",
                "stale_data",
                "Latest bar is stale",
                ticker=ticker,
                latest=str(latest_ts),
                expected=str(expected),
            )

    _fail_if_needed(issues, mode)
    return issues


def run_quality_gates_dataset(df: pd.DataFrame, cfg: Dict) -> List[Issue]:
    issues: List[Issue] = []
    mode = resolve_gate_mode(cfg)
    stage = "dataset"
    df = ensure_datetime_column(df, "date")
    if df.isna().any().any():
        _append_issue(issues, stage, "FAIL", "nan_values", "Dataset contains NaNs")
    numeric = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        _append_issue(issues, stage, "FAIL", "non_finite", "Dataset contains non-finite values")
    _fail_if_needed(issues, mode)
    return issues


def run_quality_gates_predictions(df: pd.DataFrame, cfg: Dict) -> List[Issue]:
    issues: List[Issue] = []
    stage = "predictions"
    mode = resolve_gate_mode(cfg)
    test_mode = mode.test_mode
    df = ensure_datetime_column(df, "date")
    if (df["score"] < 0).any() or (df["score"] > 100).any():
        _append_issue(issues, stage, "FAIL", "score_range", "Scores out of range")
    if df["score"].isna().any():
        _append_issue(issues, stage, "FAIL", "score_nan", "Scores contain NaNs")
    if "expected_return_horizon" in df.columns:
        if "predicted_return" in df.columns:
            same_share = (df["predicted_return"] == df["expected_return_horizon"]).mean()
            if same_share > 0.8:
                _append_issue(
                    issues,
                    stage,
                    "WARN",
                    "predicted_return_placeholder_disabled",
                    "predicted_return identical to expected_return_horizon",
                    share=f"{same_share:.2%}",
                )
        return_series = df["expected_return_horizon"]
    elif "expected_return" in df.columns:
        return_series = df["expected_return"]
    elif "predicted_return" in df.columns:
        return_series = df["predicted_return"]
    else:
        return_series = df.get("predicted_return_1d", pd.Series(dtype=float))
    if not return_series.empty:
        rounded = return_series.round(6)
        top_share = rounded.value_counts(normalize=True).iloc[0]
        if top_share >= 0.8:
            _append_issue(
                issues,
                stage,
                "WARN",
                "flat_predicted_return",
                "Predicted returns appear degenerate across tickers/horizons",
                top_share=f"{top_share:.2%}",
            )
    if "expected_return_annual" in df.columns:
        annual_returns = pd.to_numeric(df["expected_return_annual"], errors="coerce").dropna()
        if not annual_returns.empty:
            annual_count = int(len(annual_returns))
            annual_severity = "FAIL" if annual_count >= 3 else "WARN"
            if test_mode and annual_severity == "FAIL":
                annual_severity = "WARN"
            clipped_share = float((annual_returns.abs() >= float(ANNUAL_RETURN_MAX)).mean())
            if clipped_share > 0.5:
                _append_issue(
                    issues,
                    stage,
                    annual_severity,
                    "expected_return_clipped",
                    "More than half of expected annual returns hit clip bounds",
                    share=f"{clipped_share:.2%}",
                )
            annual_std = float(annual_returns.std(ddof=0))
            if annual_std < 1e-4:
                _append_issue(
                    issues,
                    stage,
                    annual_severity,
                    "expected_return_annual_flat",
                    "Expected annual returns have near-zero dispersion",
                    stddev=f"{annual_std:.2e}",
                )
    if "horizon" in df.columns and "ticker" in df.columns:
        return_col = (
            "expected_return_horizon"
            if "expected_return_horizon" in df.columns
            else "expected_return"
            if "expected_return" in df.columns
            else "predicted_return"
        )
        qa_cfg = cfg.get("qa", {})
        min_std_daily = float(qa_cfg.get("min_expected_return_std_daily", 1e-6))
        min_std_intraday = float(qa_cfg.get("min_expected_return_std_intraday", 1e-9))
        for horizon, group in df.groupby("horizon"):
            tickers = group["ticker"].nunique()
            if tickers <= 1:
                continue
            horizon_severity = "FAIL" if tickers >= 3 else "WARN"
            if test_mode and horizon_severity == "FAIL":
                horizon_severity = "WARN"
            expected_return_severity = "WARN" if mode.test_mode else horizon_severity
            horizon_returns = pd.to_numeric(group[return_col], errors="coerce").dropna()
            if not horizon_returns.empty:
                stddev = float(horizon_returns.std(ddof=0))
                threshold = min_std_intraday if is_intraday_horizon(str(horizon)) else min_std_daily
                if stddev < threshold:
                    _append_issue(
                        issues,
                        stage,
                        expected_return_severity,
                        "degenerate_expected_return",
                        "Expected returns are constant across tickers for horizon",
                        horizon=str(horizon),
                        stddev=f"{stddev:.2e}",
                    )
                if horizon_returns.nunique() == 1:
                    _append_issue(
                        issues,
                        stage,
                        expected_return_severity,
                        "expected_return_identical",
                        "Expected returns are identical across tickers for horizon",
                        horizon=str(horizon),
                    )
            if "probability" in group.columns:
                probs = pd.to_numeric(group["probability"], errors="coerce").dropna()
                if not probs.empty:
                    prob_std = float(probs.std(ddof=0))
                    if prob_std < 1e-6:
                        _append_issue(
                            issues,
                            stage,
                            horizon_severity,
                            "degenerate_probability",
                            "Probabilities are identical across tickers for horizon",
                            horizon=str(horizon),
                            stddev=f"{prob_std:.2e}",
                        )
    if "expected_return" in df.columns and "ticker" in df.columns:
        overall_returns = pd.to_numeric(df["expected_return"], errors="coerce").dropna()
        if df["ticker"].nunique() > 1 and not overall_returns.empty:
            overall_std = float(overall_returns.std(ddof=0))
            if overall_std < 1e-6:
                severity = "WARN" if test_mode else "FAIL"
                _append_issue(
                    issues,
                    stage,
                    severity,
                    "expected_return_constant",
                    "Expected returns have near-zero dispersion across tickers",
                    stddev=f"{overall_std:.2e}",
                )
    if "probability" in df.columns:
        probs = pd.to_numeric(df["probability"], errors="coerce").dropna()
        if len(probs) > 1:
            prob_std = float(probs.std(ddof=0))
            if prob_std < 1e-6:
                severity = "WARN" if test_mode else "FAIL"
                _append_issue(
                    issues,
                    stage,
                    severity,
                    "probability_constant",
                    "Probabilities are identical across predictions; model outputs collapsed",
                    stddev=f"{prob_std:.2e}",
                )
    if {"target_price", "last_close"}.issubset(df.columns):
        target_price = pd.to_numeric(df["target_price"], errors="coerce")
        last_close = pd.to_numeric(df["last_close"], errors="coerce")
        if len(target_price) > 0 and np.isclose(target_price, last_close, atol=1e-8).all():
            severity = "WARN" if test_mode else "FAIL"
            _append_issue(
                issues,
                stage,
                severity,
                "target_price_flat",
                "Target prices match last close for all rows; expected returns are zeroed",
            )
    if "probability" in df.columns and "score" in df.columns:
        if np.isclose(df["probability"], df["score"]).all():
            _append_issue(
                issues,
                stage,
                "WARN",
                "probability_equals_score",
                "Probability matches score for all rows",
            )
    _fail_if_needed(issues, mode)
    return issues


def run_quality_gates_signals(df: pd.DataFrame, cfg: Dict) -> List[Issue]:
    issues: List[Issue] = []
    stage = "signals"
    mode = resolve_gate_mode(cfg)
    test_mode = mode.test_mode
    df = ensure_datetime_column(df, "date")
    if df["signal"].isna().any():
        _append_issue(issues, stage, "FAIL", "signal_nan", "Signals contain NaNs")
    allowed = {"BUY", "HOLD", "SELL"}
    if not df["signal"].astype(str).str.upper().isin(allowed).all():
        _append_issue(issues, stage, "FAIL", "signal_invalid", "Invalid signal values")
    signals_cfg = cfg.get("signals", {})
    min_buy = int(signals_cfg.get("min_buy_signals", 0))
    if test_mode:
        min_buy = int(signals_cfg.get("min_buy_signals_test", 0))
    if min_buy > 0:
        buys = df["signal"].astype(str).str.upper().eq("BUY").sum()
        if buys < min_buy:
            severity = "WARN" if test_mode else "FAIL"
            _append_issue(
                issues,
                stage,
                severity,
                "insufficient_buy_signals",
                "Too few BUY signals generated",
                count=str(int(buys)),
                minimum=str(min_buy),
            )
    _fail_if_needed(issues, mode)
    return issues


def write_quality_report(issues: List[Issue]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"quality_report_{timestamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "total": len(issues),
            "fail": len([i for i in issues if i.severity == "FAIL"]),
            "warn": len([i for i in issues if i.severity == "WARN"]),
        },
        "issues": [asdict(issue) for issue in issues],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
