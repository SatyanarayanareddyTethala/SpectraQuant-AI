"""
SQLAlchemy database models for trading assistant.
Implements exact schema from requirements.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    ForeignKey, Index, UniqueConstraint, Text, JSON, LargeBinary
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, BYTEA
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Bars5M(Base):
    """5-minute intraday bars"""
    __tablename__ = "bars_5m"
    
    ts = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_bars_5m_symbol', 'symbol'),
        Index('idx_bars_5m_ts', 'ts'),
    )


class EOD(Base):
    """End-of-day bars"""
    __tablename__ = "eod"
    
    date = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_eod_symbol', 'symbol'),
        Index('idx_eod_date', 'date'),
    )


class NewsRaw(Base):
    """Raw news articles"""
    __tablename__ = "news_raw"
    
    news_id = Column(Integer, primary_key=True, autoincrement=True)
    ts_published = Column(DateTime, nullable=False)
    source = Column(String(100), nullable=False)
    url = Column(Text, nullable=True)
    title = Column(Text, nullable=False)
    body = Column(Text, nullable=True)
    symbols = Column(ARRAY(String), nullable=True)
    hash = Column(String(64), unique=True, nullable=False)
    
    # Relationship to enriched data
    enriched = relationship("NewsEnriched", back_populates="raw", uselist=False)
    
    __table_args__ = (
        Index('idx_news_raw_ts', 'ts_published'),
        Index('idx_news_raw_hash', 'hash'),
    )


class NewsEnriched(Base):
    """Enriched news with embeddings and risk assessment"""
    __tablename__ = "news_enriched"
    
    news_id = Column(Integer, ForeignKey('news_raw.news_id'), primary_key=True)
    embedding = Column(BYTEA, nullable=True)  # Binary serialized embedding
    risk_tags = Column(ARRAY(String), nullable=True)
    risk_score = Column(Float, nullable=True)
    summary_3bul = Column(Text, nullable=True)
    
    # Relationship to raw data
    raw = relationship("NewsRaw", back_populates="enriched")


class FeaturesDaily(Base):
    """Daily features for each symbol"""
    __tablename__ = "features_daily"
    
    date = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    feature_json = Column(JSONB, nullable=False)
    
    __table_args__ = (
        Index('idx_features_daily_symbol', 'symbol'),
        Index('idx_features_daily_date', 'date'),
    )


class ModelRegistry(Base):
    """Registry of trained models"""
    __tablename__ = "model_registry"
    
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    model_type = Column(String(50), nullable=False)  # rank, fail, calibrated
    data_window = Column(JSONB, nullable=False)
    metrics_json = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, default='candidate')  # candidate, active, archived, rolled_back
    model_path = Column(Text, nullable=True)  # Path to serialized model
    
    # Relationships
    premarket_plans_rank = relationship("PremarketPlan", foreign_keys="PremarketPlan.model_id_rank", back_populates="rank_model")
    premarket_plans_fail = relationship("PremarketPlan", foreign_keys="PremarketPlan.model_id_fail", back_populates="fail_model")
    
    __table_args__ = (
        Index('idx_model_registry_status', 'status'),
        Index('idx_model_registry_created', 'created_at'),
    )


class PremarketPlan(Base):
    """Premarket trading plan"""
    __tablename__ = "premarket_plan"
    
    plan_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_date = Column(DateTime, nullable=False, unique=True)
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    model_id_rank = Column(Integer, ForeignKey('model_registry.model_id'), nullable=True)
    model_id_fail = Column(Integer, ForeignKey('model_registry.model_id'), nullable=True)
    plan_json = Column(JSONB, nullable=False)
    
    # Relationships
    rank_model = relationship("ModelRegistry", foreign_keys=[model_id_rank], back_populates="premarket_plans_rank")
    fail_model = relationship("ModelRegistry", foreign_keys=[model_id_fail], back_populates="premarket_plans_fail")
    trades = relationship("PlanTrade", back_populates="plan")
    alerts = relationship("Alert", back_populates="plan")
    fills = relationship("Fill", back_populates="plan")
    outcomes = relationship("TradeOutcome", back_populates="plan")
    
    __table_args__ = (
        Index('idx_premarket_plan_date', 'plan_date'),
    )


class PlanTrade(Base):
    """Individual trades in a premarket plan"""
    __tablename__ = "plan_trades"
    
    plan_id = Column(Integer, ForeignKey('premarket_plan.plan_id'), primary_key=True)
    symbol = Column(String(20), primary_key=True, nullable=False)
    rank = Column(Integer, nullable=False)
    side = Column(String(10), nullable=False, default='LONG')
    entry_type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP
    entry_price = Column(Float, nullable=False)
    stop_price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    size_shares = Column(Integer, nullable=False)
    trigger_json = Column(JSONB, nullable=False)
    score_rank = Column(Float, nullable=False)
    p_fail = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    do_not_trade_if = Column(JSONB, nullable=False)
    
    # Relationship
    plan = relationship("PremarketPlan", back_populates="trades")
    
    __table_args__ = (
        Index('idx_plan_trades_rank', 'rank'),
        Index('idx_plan_trades_symbol', 'symbol'),
    )


class Alert(Base):
    """Alerts and email notifications"""
    __tablename__ = "alerts"
    
    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey('premarket_plan.plan_id'), nullable=True)
    symbol = Column(String(20), nullable=True)
    alert_type = Column(String(50), nullable=False)  # PLAN, EXECUTE_NOW, NEWS
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    dedupe_key = Column(String(200), unique=True, nullable=False)
    payload_json = Column(JSONB, nullable=False)
    email_to = Column(ARRAY(String), nullable=False)
    email_status = Column(String(20), nullable=False, default='pending')  # pending, sent, failed
    sent_at = Column(DateTime, nullable=True)
    
    # Relationship
    plan = relationship("PremarketPlan", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_alerts_type', 'alert_type'),
        Index('idx_alerts_created', 'created_at'),
        Index('idx_alerts_dedupe', 'dedupe_key'),
    )


class Fill(Base):
    """Trade execution fills"""
    __tablename__ = "fills"
    
    fill_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey('premarket_plan.plan_id'), nullable=True)
    symbol = Column(String(20), nullable=False)
    ts_fill = Column(DateTime, nullable=False)
    action = Column(String(10), nullable=False)  # BUY, SELL
    qty = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    fees = Column(Float, nullable=False)
    slippage_bps = Column(Float, nullable=False)
    venue = Column(String(50), nullable=True)
    meta_json = Column(JSONB, nullable=True)
    
    # Relationship
    plan = relationship("PremarketPlan", back_populates="fills")
    
    __table_args__ = (
        Index('idx_fills_symbol', 'symbol'),
        Index('idx_fills_ts', 'ts_fill'),
    )


class TradeOutcome(Base):
    """Trade outcomes and performance metrics"""
    __tablename__ = "trade_outcomes"
    
    trade_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey('premarket_plan.plan_id'), nullable=True)
    symbol = Column(String(20), nullable=False)
    entry_ts = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_ts = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl_net = Column(Float, nullable=True)
    return_net = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)  # Maximum Adverse Excursion
    mfe = Column(Float, nullable=True)  # Maximum Favorable Excursion
    holding_mins = Column(Integer, nullable=True)
    cost_total = Column(Float, nullable=True)
    outcome_json = Column(JSONB, nullable=True)
    
    # Relationships
    plan = relationship("PremarketPlan", back_populates="outcomes")
    failure_labels = relationship("FailureLabel", back_populates="trade_outcome")
    
    __table_args__ = (
        Index('idx_trade_outcomes_symbol', 'symbol'),
        Index('idx_trade_outcomes_entry_ts', 'entry_ts'),
    )


class FailureLabel(Base):
    """Failure labels for trade outcomes"""
    __tablename__ = "failure_labels"
    
    trade_id = Column(Integer, ForeignKey('trade_outcomes.trade_id'), primary_key=True)
    label = Column(String(50), primary_key=True, nullable=False)
    severity = Column(String(20), nullable=False)  # critical, high, medium, low
    details_json = Column(JSONB, nullable=True)
    
    # Relationship
    trade_outcome = relationship("TradeOutcome", back_populates="failure_labels")


class LearningRun(Base):
    """Learning and model update runs"""
    __tablename__ = "learning_runs"
    
    run_id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    data_range = Column(JSONB, nullable=False)
    drift_flags = Column(JSONB, nullable=True)
    candidate_models = Column(JSONB, nullable=True)
    promoted_model_id = Column(Integer, ForeignKey('model_registry.model_id'), nullable=True)
    decision = Column(String(20), nullable=False)  # PROMOTE, HOLD, ROLLBACK
    notes = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_learning_runs_started', 'started_at'),
    )
