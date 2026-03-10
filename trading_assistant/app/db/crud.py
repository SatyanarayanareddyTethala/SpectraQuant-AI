"""
CRUD operations for database models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from . import models


# News operations
def create_news_raw(
    db: Session,
    ts_published: datetime,
    source: str,
    title: str,
    content_hash: str,
    body: Optional[str] = None,
    url: Optional[str] = None,
    symbols: Optional[List[str]] = None
) -> models.NewsRaw:
    """Create a raw news article"""
    news = models.NewsRaw(
        ts_published=ts_published,
        source=source,
        url=url,
        title=title,
        body=body,
        symbols=symbols,
        hash=content_hash
    )
    db.add(news)
    db.commit()
    db.refresh(news)
    return news


def get_news_by_hash(db: Session, content_hash: str) -> Optional[models.NewsRaw]:
    """Check if news article already exists by hash"""
    return db.query(models.NewsRaw).filter(models.NewsRaw.hash == content_hash).first()


def create_news_enriched(
    db: Session,
    news_id: int,
    embedding: Optional[bytes] = None,
    risk_tags: Optional[List[str]] = None,
    risk_score: Optional[float] = None,
    summary: Optional[str] = None
) -> models.NewsEnriched:
    """Create enriched news data"""
    enriched = models.NewsEnriched(
        news_id=news_id,
        embedding=embedding,
        risk_tags=risk_tags,
        risk_score=risk_score,
        summary_3bul=summary
    )
    db.add(enriched)
    db.commit()
    db.refresh(enriched)
    return enriched


# Model registry operations
def create_model(
    db: Session,
    model_type: str,
    data_window: Dict[str, Any],
    metrics: Dict[str, Any],
    model_path: Optional[str] = None,
    status: str = 'candidate'
) -> models.ModelRegistry:
    """Register a new model"""
    model = models.ModelRegistry(
        created_at=datetime.utcnow(),
        model_type=model_type,
        data_window=data_window,
        metrics_json=metrics,
        status=status,
        model_path=model_path
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def get_active_model(db: Session, model_type: str) -> Optional[models.ModelRegistry]:
    """Get the currently active model of a given type"""
    return db.query(models.ModelRegistry).filter(
        and_(
            models.ModelRegistry.model_type == model_type,
            models.ModelRegistry.status == 'active'
        )
    ).order_by(desc(models.ModelRegistry.created_at)).first()


def update_model_status(db: Session, model_id: int, status: str):
    """Update model status"""
    model = db.query(models.ModelRegistry).filter(models.ModelRegistry.model_id == model_id).first()
    if model:
        model.status = status
        db.commit()


# Premarket plan operations
def create_premarket_plan(
    db: Session,
    plan_date: datetime,
    plan_json: Dict[str, Any],
    model_id_rank: Optional[int] = None,
    model_id_fail: Optional[int] = None
) -> models.PremarketPlan:
    """Create a premarket plan"""
    plan = models.PremarketPlan(
        plan_date=plan_date,
        generated_at=datetime.utcnow(),
        model_id_rank=model_id_rank,
        model_id_fail=model_id_fail,
        plan_json=plan_json
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def get_plan_by_date(db: Session, plan_date: datetime) -> Optional[models.PremarketPlan]:
    """Get premarket plan for a specific date"""
    return db.query(models.PremarketPlan).filter(
        models.PremarketPlan.plan_date == plan_date
    ).first()


def get_latest_plan(db: Session) -> Optional[models.PremarketPlan]:
    """Get the most recent premarket plan"""
    return db.query(models.PremarketPlan).order_by(
        desc(models.PremarketPlan.plan_date)
    ).first()


# Plan trades operations
def create_plan_trade(db: Session, plan_id: int, trade_data: Dict[str, Any]) -> models.PlanTrade:
    """Create a trade in a premarket plan"""
    trade = models.PlanTrade(
        plan_id=plan_id,
        **trade_data
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


def get_plan_trades(db: Session, plan_id: int) -> List[models.PlanTrade]:
    """Get all trades for a plan"""
    return db.query(models.PlanTrade).filter(
        models.PlanTrade.plan_id == plan_id
    ).order_by(models.PlanTrade.rank).all()


# Alert operations
def create_alert(
    db: Session,
    alert_type: str,
    dedupe_key: str,
    payload: Dict[str, Any],
    email_to: List[str],
    plan_id: Optional[int] = None,
    symbol: Optional[str] = None
) -> Optional[models.Alert]:
    """
    Create an alert with deduplication.
    Returns None if duplicate key exists.
    """
    # Check if dedupe key already exists
    existing = db.query(models.Alert).filter(
        models.Alert.dedupe_key == dedupe_key
    ).first()
    
    if existing:
        return None
    
    alert = models.Alert(
        plan_id=plan_id,
        symbol=symbol,
        alert_type=alert_type,
        created_at=datetime.utcnow(),
        dedupe_key=dedupe_key,
        payload_json=payload,
        email_to=email_to,
        email_status='pending'
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def update_alert_status(
    db: Session,
    alert_id: int,
    status: str,
    sent_at: Optional[datetime] = None
):
    """Update alert email status"""
    alert = db.query(models.Alert).filter(models.Alert.alert_id == alert_id).first()
    if alert:
        alert.email_status = status
        if sent_at:
            alert.sent_at = sent_at
        db.commit()


def get_pending_alerts(db: Session) -> List[models.Alert]:
    """Get all pending alerts"""
    return db.query(models.Alert).filter(
        models.Alert.email_status == 'pending'
    ).all()


# Fill operations
def create_fill(
    db: Session,
    symbol: str,
    ts_fill: datetime,
    action: str,
    qty: int,
    price: float,
    fees: float,
    slippage_bps: float,
    plan_id: Optional[int] = None,
    venue: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
) -> models.Fill:
    """Record a trade fill"""
    fill = models.Fill(
        plan_id=plan_id,
        symbol=symbol,
        ts_fill=ts_fill,
        action=action,
        qty=qty,
        price=price,
        fees=fees,
        slippage_bps=slippage_bps,
        venue=venue,
        meta_json=meta
    )
    db.add(fill)
    db.commit()
    db.refresh(fill)
    return fill


# Trade outcome operations
def create_trade_outcome(
    db: Session,
    symbol: str,
    entry_ts: datetime,
    entry_price: float,
    plan_id: Optional[int] = None,
    outcome_data: Optional[Dict[str, Any]] = None
) -> models.TradeOutcome:
    """Create a trade outcome record"""
    outcome = models.TradeOutcome(
        plan_id=plan_id,
        symbol=symbol,
        entry_ts=entry_ts,
        entry_price=entry_price,
        **(outcome_data or {})
    )
    db.add(outcome)
    db.commit()
    db.refresh(outcome)
    return outcome


def update_trade_outcome(
    db: Session,
    trade_id: int,
    exit_ts: datetime,
    exit_price: float,
    pnl_net: float,
    return_net: float,
    mae: float,
    mfe: float,
    holding_mins: int,
    cost_total: float,
    outcome_json: Optional[Dict[str, Any]] = None
):
    """Update trade outcome with exit data"""
    outcome = db.query(models.TradeOutcome).filter(
        models.TradeOutcome.trade_id == trade_id
    ).first()
    if outcome:
        outcome.exit_ts = exit_ts
        outcome.exit_price = exit_price
        outcome.pnl_net = pnl_net
        outcome.return_net = return_net
        outcome.mae = mae
        outcome.mfe = mfe
        outcome.holding_mins = holding_mins
        outcome.cost_total = cost_total
        outcome.outcome_json = outcome_json
        db.commit()


def create_failure_label(
    db: Session,
    trade_id: int,
    label: str,
    severity: str,
    details: Optional[Dict[str, Any]] = None
) -> models.FailureLabel:
    """Add a failure label to a trade outcome"""
    failure = models.FailureLabel(
        trade_id=trade_id,
        label=label,
        severity=severity,
        details_json=details
    )
    db.add(failure)
    db.commit()
    db.refresh(failure)
    return failure


# Learning run operations
def create_learning_run(
    db: Session,
    data_range: Dict[str, Any],
    decision: str,
    drift_flags: Optional[Dict[str, Any]] = None,
    candidate_models: Optional[Dict[str, Any]] = None,
    promoted_model_id: Optional[int] = None,
    notes: Optional[str] = None
) -> models.LearningRun:
    """Create a learning run record"""
    run = models.LearningRun(
        started_at=datetime.utcnow(),
        data_range=data_range,
        drift_flags=drift_flags,
        candidate_models=candidate_models,
        promoted_model_id=promoted_model_id,
        decision=decision,
        notes=notes
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def update_learning_run(
    db: Session,
    run_id: int,
    finished_at: datetime,
    promoted_model_id: Optional[int] = None,
    decision: Optional[str] = None
):
    """Update learning run with completion data"""
    run = db.query(models.LearningRun).filter(
        models.LearningRun.run_id == run_id
    ).first()
    if run:
        run.finished_at = finished_at
        if promoted_model_id is not None:
            run.promoted_model_id = promoted_model_id
        if decision is not None:
            run.decision = decision
        db.commit()


# Market data operations
def bulk_insert_bars_5m(db: Session, bars_data: List[Dict[str, Any]]):
    """Bulk insert 5-minute bars"""
    bars = [models.Bars5M(**bar) for bar in bars_data]
    db.bulk_save_objects(bars)
    db.commit()


def bulk_insert_eod(db: Session, eod_data: List[Dict[str, Any]]):
    """Bulk insert EOD bars"""
    bars = [models.EOD(**bar) for bar in eod_data]
    db.bulk_save_objects(bars)
    db.commit()


def get_latest_bar_date(db: Session, symbol: str) -> Optional[datetime]:
    """Get the latest date for which we have EOD data"""
    result = db.query(models.EOD.date).filter(
        models.EOD.symbol == symbol
    ).order_by(desc(models.EOD.date)).first()
    return result[0] if result else None
