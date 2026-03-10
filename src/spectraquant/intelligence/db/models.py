"""SQLAlchemy ORM models for the Intelligence layer.

All timestamp columns use UTC.  Indexes target the most common query
patterns: ``(symbol, ts)``, ``(dedupe_key)``, ``(plan_id, symbol)``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all Intelligence models."""


# ---------------------------------------------------------------------------
# Pre-market plan
# ---------------------------------------------------------------------------

class PremarketPlan(Base):
    __tablename__ = "premarket_plan"

    plan_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_date = Column(DateTime(timezone=True), nullable=False, unique=True)
    plan_json = Column(Text, nullable=True)
    rank_model_id = Column(Integer, ForeignKey("model_registry.model_id"), nullable=True)
    simulation = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    trades = relationship("PlanTrade", back_populates="plan", lazy="selectin")
    alerts = relationship("Alert", back_populates="plan", lazy="selectin")


# ---------------------------------------------------------------------------
# Plan trades
# ---------------------------------------------------------------------------

class PlanTrade(Base):
    __tablename__ = "plan_trades"

    trade_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey("premarket_plan.plan_id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), default="long")
    entry = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    risk_per_share = Column(Float, nullable=True)
    shares = Column(Float, nullable=True)
    trigger_json = Column(Text, nullable=True)
    do_not_trade_if = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    plan = relationship("PremarketPlan", back_populates="trades")

    __table_args__ = (
        Index("ix_plan_trades_plan_symbol", "plan_id", "symbol"),
    )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class Alert(Base):
    __tablename__ = "alerts"

    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey("premarket_plan.plan_id"), nullable=True)
    symbol = Column(String(20), nullable=True)
    category = Column(String(30), nullable=False)
    dedupe_key = Column(String(255), nullable=False, unique=True)
    payload = Column(Text, nullable=True)
    status = Column(String(20), default="pending")
    sent_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    plan = relationship("PremarketPlan", back_populates="alerts")

    __table_args__ = (
        Index("ix_alerts_dedupe", "dedupe_key"),
        Index("ix_alerts_symbol_ts", "symbol", "created_at"),
    )


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------

class Fill(Base):
    __tablename__ = "fills"

    fill_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey("premarket_plan.plan_id"), nullable=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    ts = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_fills_symbol_ts", "symbol", "ts"),
    )


# ---------------------------------------------------------------------------
# Trade outcomes
# ---------------------------------------------------------------------------

class TradeOutcome(Base):
    __tablename__ = "trade_outcomes"

    outcome_id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, ForeignKey("plan_trades.trade_id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    pnl = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    mfe = Column(Float, nullable=True)
    holding_seconds = Column(Integer, nullable=True)
    exit_reason = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_outcomes_symbol_ts", "symbol", "created_at"),
    )


# ---------------------------------------------------------------------------
# Failure labels (feedback loop)
# ---------------------------------------------------------------------------

class FailureLabel(Base):
    __tablename__ = "failure_labels"

    label_id = Column(Integer, primary_key=True, autoincrement=True)
    outcome_id = Column(Integer, ForeignKey("trade_outcomes.outcome_id"), nullable=False)
    label = Column(String(50), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Learning runs
# ---------------------------------------------------------------------------

class LearningRun(Base):
    __tablename__ = "learning_runs"

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default="running")
    metrics_json = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

class NewsRaw(Base):
    __tablename__ = "news_raw"

    news_id = Column(Integer, primary_key=True, autoincrement=True)
    headline = Column(Text, nullable=False)
    source = Column(String(100), nullable=True)
    url = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=True, unique=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    fetched_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class NewsEnriched(Base):
    __tablename__ = "news_enriched"

    enriched_id = Column(Integer, primary_key=True, autoincrement=True)
    news_id = Column(Integer, ForeignKey("news_raw.news_id"), nullable=False)
    risk_score = Column(Float, nullable=True)
    risk_tags = Column(Text, nullable=True)
    symbols = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class FeaturesDaily(Base):
    __tablename__ = "features_daily"

    feature_id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    features_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_features_symbol_date", "symbol", "date", unique=True),
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=True)
    status = Column(String(20), default="candidate")
    artifact_path = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Extended Event Store Schemas
# ---------------------------------------------------------------------------

class Ticker(Base):
    """Ticker/symbol master table."""
    __tablename__ = "ticker"

    ticker_id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True)
    exchange = Column(String(20), nullable=True)
    company_name = Column(String(255), nullable=True)
    sector = Column(String(100), nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_ticker_symbol", "symbol"),
    )


class PriceBar(Base):
    """OHLCV price bars."""
    __tablename__ = "price_bar"

    bar_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey("ticker.ticker_id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    interval = Column(String(10), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_price_bar_ticker_ts", "ticker_id", "timestamp"),
        Index("ix_price_bar_ticker_interval_ts", "ticker_id", "interval", "timestamp", unique=True),
    )


class NewsArticle(Base):
    """News articles with deduplication support."""
    __tablename__ = "news_article"

    article_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    source = Column(String(100), nullable=True)
    url = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=False, unique=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    fetched_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    dedupe_reason = Column(String(50), nullable=True)
    cluster_id = Column(Integer, nullable=True)
    embedding_vector = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_news_article_content_hash", "content_hash"),
        Index("ix_news_article_cluster_id", "cluster_id"),
        Index("ix_news_article_published_at", "published_at"),
    )


class NewsMention(Base):
    """Ticker mentions in news articles."""
    __tablename__ = "news_mention"

    mention_id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("news_article.article_id"), nullable=False)
    ticker_id = Column(Integer, ForeignKey("ticker.ticker_id"), nullable=False)
    relevance_score = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_news_mention_article_ticker", "article_id", "ticker_id", unique=True),
        Index("ix_news_mention_ticker", "ticker_id"),
    )


class Event(Base):
    """Detected events from news/filings."""
    __tablename__ = "event"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    ticker_id = Column(Integer, ForeignKey("ticker.ticker_id"), nullable=False)
    article_id = Column(Integer, ForeignKey("news_article.article_id"), nullable=True)
    detected_at = Column(DateTime(timezone=True), nullable=False)
    confidence_score = Column(Float, nullable=True)
    event_date = Column(DateTime(timezone=True), nullable=True)
    sentiment = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_event_ticker_type", "ticker_id", "event_type"),
        Index("ix_event_detected_at", "detected_at"),
    )


class EventArgument(Base):
    """Arguments/attributes extracted from events."""
    __tablename__ = "event_argument"

    arg_id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("event.event_id"), nullable=False)
    arg_name = Column(String(50), nullable=False)
    arg_value = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_event_argument_event_id", "event_id"),
    )


class EventImpact(Base):
    """Measured impact of events on price/volume."""
    __tablename__ = "event_impact"

    impact_id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("event.event_id"), nullable=False, unique=True)
    abnormal_return_1d = Column(Float, nullable=True)
    abnormal_return_5d = Column(Float, nullable=True)
    abnormal_return_20d = Column(Float, nullable=True)
    volume_spike = Column(Float, nullable=True)
    volatility_spike = Column(Float, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_event_impact_event_id", "event_id"),
    )


class ModelVersion(Base):
    """Model versioning and metadata."""
    __tablename__ = "model_version"

    version_id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)
    version_tag = Column(String(50), nullable=False)
    framework = Column(String(50), nullable=True)
    hyperparams_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    artifact_path = Column(Text, nullable=True)
    status = Column(String(20), default="candidate")
    trained_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    promoted_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_model_version_type_tag", "model_type", "version_tag", unique=True),
        Index("ix_model_version_status", "status"),
    )


class Prediction(Base):
    """Model predictions with conformal intervals."""
    __tablename__ = "prediction"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey("ticker.ticker_id"), nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_version.version_id"), nullable=False)
    prediction_date = Column(DateTime(timezone=True), nullable=False)
    target_horizon_days = Column(Integer, nullable=False)
    predicted_return = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    coverage_level = Column(Float, nullable=True)
    features_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_prediction_ticker_date", "ticker_id", "prediction_date"),
        Index("ix_prediction_model_ticker", "model_version_id", "ticker_id"),
    )
