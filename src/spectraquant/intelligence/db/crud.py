"""CRUD operations for all Intelligence database models."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from spectraquant.intelligence.db.models import (
    Alert,
    FailureLabel,
    FeaturesDaily,
    Fill,
    LearningRun,
    ModelRegistry,
    NewsEnriched,
    NewsRaw,
    PlanTrade,
    PremarketPlan,
    TradeOutcome,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PremarketPlan
# ---------------------------------------------------------------------------

def create_plan(session: Session, plan_date: datetime, plan_json: str,
                simulation: bool = True, rank_model_id: Optional[int] = None) -> PremarketPlan:
    plan = PremarketPlan(
        plan_date=plan_date,
        plan_json=plan_json,
        simulation=simulation,
        rank_model_id=rank_model_id,
    )
    session.add(plan)
    session.flush()
    logger.debug("Created plan %d for %s", plan.plan_id, plan_date)
    return plan


def get_plan_by_date(session: Session, plan_date: datetime) -> Optional[PremarketPlan]:
    return session.query(PremarketPlan).filter(
        PremarketPlan.plan_date == plan_date
    ).first()


def get_latest_plan(session: Session) -> Optional[PremarketPlan]:
    return session.query(PremarketPlan).order_by(
        PremarketPlan.plan_date.desc()
    ).first()


# ---------------------------------------------------------------------------
# PlanTrade
# ---------------------------------------------------------------------------

def create_plan_trade(session: Session, plan_id: int, symbol: str,
                      direction: str = "long", entry: Optional[float] = None,
                      stop_loss: Optional[float] = None, target: Optional[float] = None,
                      risk_per_share: Optional[float] = None, shares: Optional[float] = None,
                      trigger_json: Optional[str] = None) -> PlanTrade:
    trade = PlanTrade(
        plan_id=plan_id, symbol=symbol, direction=direction,
        entry=entry, stop_loss=stop_loss, target=target,
        risk_per_share=risk_per_share, shares=shares,
        trigger_json=trigger_json,
    )
    session.add(trade)
    session.flush()
    return trade


def get_plan_trades(session: Session, plan_id: int) -> List[PlanTrade]:
    return session.query(PlanTrade).filter(PlanTrade.plan_id == plan_id).all()


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

def create_alert(session: Session, category: str, dedupe_key: str,
                 plan_id: Optional[int] = None, symbol: Optional[str] = None,
                 payload: Optional[str] = None) -> Optional[Alert]:
    """Create an alert only if its *dedupe_key* is unique."""
    existing = session.query(Alert).filter(Alert.dedupe_key == dedupe_key).first()
    if existing is not None:
        logger.debug("Alert already exists for key=%s", dedupe_key)
        return None
    alert = Alert(
        plan_id=plan_id, symbol=symbol, category=category,
        dedupe_key=dedupe_key, payload=payload,
    )
    session.add(alert)
    session.flush()
    return alert


def get_pending_alerts(session: Session) -> List[Alert]:
    return session.query(Alert).filter(Alert.status == "pending").order_by(
        Alert.created_at
    ).all()


def update_alert_status(session: Session, alert_id: int, status: str) -> None:
    alert = session.query(Alert).filter(Alert.alert_id == alert_id).first()
    if alert:
        alert.status = status
        alert.sent_at = datetime.now(tz=timezone.utc)
        session.flush()


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

def create_fill(session: Session, symbol: str, side: str, qty: float,
                price: float, commission: float = 0.0,
                plan_id: Optional[int] = None) -> Fill:
    fill = Fill(
        plan_id=plan_id, symbol=symbol, side=side,
        qty=qty, price=price, commission=commission,
    )
    session.add(fill)
    session.flush()
    return fill


def get_fills_by_plan(session: Session, plan_id: int) -> List[Fill]:
    return session.query(Fill).filter(Fill.plan_id == plan_id).order_by(Fill.ts).all()


# ---------------------------------------------------------------------------
# TradeOutcome
# ---------------------------------------------------------------------------

def create_trade_outcome(session: Session, trade_id: int, symbol: str,
                         pnl: Optional[float] = None, mae: Optional[float] = None,
                         mfe: Optional[float] = None, holding_seconds: Optional[int] = None,
                         exit_reason: Optional[str] = None) -> TradeOutcome:
    outcome = TradeOutcome(
        trade_id=trade_id, symbol=symbol, pnl=pnl,
        mae=mae, mfe=mfe, holding_seconds=holding_seconds,
        exit_reason=exit_reason,
    )
    session.add(outcome)
    session.flush()
    return outcome


def get_outcomes_by_symbol(session: Session, symbol: str) -> List[TradeOutcome]:
    return session.query(TradeOutcome).filter(
        TradeOutcome.symbol == symbol
    ).order_by(TradeOutcome.created_at.desc()).all()


# ---------------------------------------------------------------------------
# FailureLabel
# ---------------------------------------------------------------------------

def create_failure_label(session: Session, outcome_id: int, label: str,
                         notes: Optional[str] = None) -> FailureLabel:
    fl = FailureLabel(outcome_id=outcome_id, label=label, notes=notes)
    session.add(fl)
    session.flush()
    return fl


# ---------------------------------------------------------------------------
# LearningRun
# ---------------------------------------------------------------------------

def create_learning_run(session: Session) -> LearningRun:
    run = LearningRun()
    session.add(run)
    session.flush()
    return run


def update_learning_run(session: Session, run_id: int, status: str,
                        metrics_json: Optional[str] = None) -> None:
    run = session.query(LearningRun).filter(LearningRun.run_id == run_id).first()
    if run:
        run.status = status
        run.finished_at = datetime.now(tz=timezone.utc)
        if metrics_json is not None:
            run.metrics_json = metrics_json
        session.flush()


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def create_news_raw(session: Session, headline: str, source: Optional[str] = None,
                    url: Optional[str] = None, content_hash: Optional[str] = None,
                    published_at: Optional[datetime] = None) -> Optional[NewsRaw]:
    if content_hash:
        existing = session.query(NewsRaw).filter(
            NewsRaw.content_hash == content_hash
        ).first()
        if existing:
            return None
    nr = NewsRaw(
        headline=headline, source=source, url=url,
        content_hash=content_hash, published_at=published_at,
    )
    session.add(nr)
    session.flush()
    return nr


def create_news_enriched(session: Session, news_id: int,
                         risk_score: Optional[float] = None,
                         risk_tags: Optional[str] = None,
                         symbols: Optional[str] = None) -> NewsEnriched:
    ne = NewsEnriched(
        news_id=news_id, risk_score=risk_score,
        risk_tags=risk_tags, symbols=symbols,
    )
    session.add(ne)
    session.flush()
    return ne


# ---------------------------------------------------------------------------
# FeaturesDaily
# ---------------------------------------------------------------------------

def upsert_features_daily(session: Session, symbol: str, date: datetime,
                          features_json: str) -> FeaturesDaily:
    existing = session.query(FeaturesDaily).filter(
        FeaturesDaily.symbol == symbol, FeaturesDaily.date == date
    ).first()
    if existing:
        existing.features_json = features_json
        session.flush()
        return existing
    fd = FeaturesDaily(symbol=symbol, date=date, features_json=features_json)
    session.add(fd)
    session.flush()
    return fd


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

def create_model(session: Session, model_name: str, version: Optional[str] = None,
                 artifact_path: Optional[str] = None,
                 metrics_json: Optional[str] = None) -> ModelRegistry:
    mr = ModelRegistry(
        model_name=model_name, version=version,
        artifact_path=artifact_path, metrics_json=metrics_json,
    )
    session.add(mr)
    session.flush()
    return mr


def get_active_model(session: Session, model_name: str) -> Optional[ModelRegistry]:
    return session.query(ModelRegistry).filter(
        ModelRegistry.model_name == model_name,
        ModelRegistry.status == "active",
    ).order_by(ModelRegistry.created_at.desc()).first()


def update_model_status(session: Session, model_id: int, status: str) -> None:
    mr = session.query(ModelRegistry).filter(
        ModelRegistry.model_id == model_id
    ).first()
    if mr:
        mr.status = status
        session.flush()
