"""Portfolio rationale builder for decision-grade explainability."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def _latest_date(df: pd.DataFrame, date_col: str = "date") -> pd.Timestamp | None:
    if date_col not in df.columns:
        return None
    dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max()


def _infer_selected(df: pd.DataFrame) -> pd.Series:
    if "selected" in df.columns:
        return df["selected"].astype(bool)
    if "weight" in df.columns:
        return df["weight"].fillna(0).astype(float) > 0
    if "signal" in df.columns:
        return df["signal"].astype(str).str.upper().eq("BUY")
    if "rank" in df.columns:
        return df["rank"].astype(float) <= min(10, len(df))
    return pd.Series([True] * len(df), index=df.index)


def _score_column(df: pd.DataFrame) -> str | None:
    for col in ("ensemble_score", "score", "probability", "confidence"):
        if col in df.columns:
            return col
    return None


def _summarize_contributions(contrib_df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    if contrib_df.empty:
        return [], []
    contrib_df = contrib_df.sort_values("contribution", ascending=False)
    positive = contrib_df[contrib_df["contribution"] > 0].head(3)
    negative = contrib_df[contrib_df["contribution"] < 0].tail(3)
    pos_list = [
        {"feature": row["feature"], "contribution": float(row["contribution"])}
        for _, row in positive.iterrows()
    ]
    neg_list = [
        {"feature": row["feature"], "contribution": float(row["contribution"])}
        for _, row in negative.iterrows()
    ]
    return pos_list, neg_list


def build_portfolio_rationale(
    portfolio_df: Any,
    feature_contributions: Any,
    regime_labels: Any,
) -> dict:
    portfolio_df = _ensure_dataframe(portfolio_df)
    if portfolio_df.empty:
        raise ValueError("portfolio_df is empty; cannot build rationale")
    if "ticker" not in portfolio_df.columns:
        raise ValueError("portfolio_df must include a ticker column")

    feature_df = _ensure_dataframe(feature_contributions)
    if not feature_df.empty and not {"ticker", "feature", "contribution"}.issubset(feature_df.columns):
        raise ValueError("feature_contributions must include ticker, feature, contribution columns")

    regimes = pd.Series(regime_labels)
    regimes.index = pd.to_datetime(regimes.index, utc=True, errors="coerce")
    regimes = regimes.dropna()

    portfolio_df = portfolio_df.copy()
    if "date" in portfolio_df.columns:
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"], utc=True, errors="coerce")
        portfolio_df = portfolio_df.dropna(subset=["date"])

    latest_dt = _latest_date(portfolio_df, "date")
    if latest_dt is not None:
        latest_regime = regimes.loc[regimes.index <= latest_dt].tail(1)
        regime_context = latest_regime.iloc[0] if not latest_regime.empty else "UNKNOWN"
    else:
        regime_context = "UNKNOWN"

    selected_mask = _infer_selected(portfolio_df)
    selected_df = portfolio_df[selected_mask]
    rejected_df = portfolio_df[~selected_mask]
    if selected_df.empty:
        raise ValueError("No selected positions found in portfolio_df")

    score_col = _score_column(portfolio_df)
    if score_col:
        selected_scores = selected_df[score_col].astype(float)
        cutoff = selected_scores.min() if not selected_scores.empty else 0.0
    else:
        cutoff = 0.0

    rationale_entries: list[dict[str, Any]] = []
    for _, row in selected_df.iterrows():
        ticker = row["ticker"]
        ticker_features = feature_df[feature_df["ticker"] == ticker]
        pos_feats, neg_feats = _summarize_contributions(ticker_features)

        confidence = float(row[score_col]) if score_col and pd.notna(row[score_col]) else None
        reasons = []
        if score_col and confidence is not None:
            reasons.append(f"Score {confidence:.2f} exceeded cutoff {cutoff:.2f} using {score_col}.")
        if pos_feats:
            reasons.append(
                "Positive drivers: " + ", ".join(f"{f['feature']}" for f in pos_feats) + "."
            )
        if neg_feats:
            reasons.append(
                "Offsets observed from: " + ", ".join(f"{f['feature']}" for f in neg_feats) + "."
            )

        similar_rejections = []
        if score_col and not rejected_df.empty:
            rejected_scores = rejected_df.copy()
            rejected_scores[score_col] = pd.to_numeric(rejected_scores[score_col], errors="coerce")
            rejected_scores = rejected_scores.dropna(subset=[score_col])
            rejected_scores["distance"] = (cutoff - rejected_scores[score_col]).abs()
            closest = rejected_scores.sort_values("distance").head(3)
            for _, rej in closest.iterrows():
                reason = f"Score {rej[score_col]:.2f} below cutoff {cutoff:.2f}."
                similar_rejections.append({"ticker": rej["ticker"], "reason": reason})

        rationale_entries.append(
            {
                "ticker": ticker,
                "date": row.get("date"),
                "model_confidence": confidence,
                "regime": regime_context,
                "top_contributors_positive": pos_feats,
                "top_contributors_negative": neg_feats,
                "selection_rationale": reasons,
                "similar_rejections": similar_rejections,
            }
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regime_context": regime_context,
        "positions": rationale_entries,
    }

    output_dir = Path("reports/explain")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"portfolio_rationale_{run_id}.json"
    output_path.write_text(json.dumps(report, indent=2, default=str))
    report["output_path"] = str(output_path)
    return report


__all__ = ["build_portfolio_rationale"]
