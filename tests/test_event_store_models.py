"""Tests for extended event store database models."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from spectraquant.intelligence.db.models import (
    Base,
    Event,
    EventArgument,
    EventImpact,
    ModelVersion,
    NewsArticle,
    NewsMention,
    Prediction,
    PriceBar,
    Ticker,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_ticker_model(db_session: Session) -> None:
    """Test Ticker model creation and constraints."""
    ticker = Ticker(
        symbol="AAPL",
        exchange="NASDAQ",
        company_name="Apple Inc.",
        sector="Technology",
        active=True,
    )
    db_session.add(ticker)
    db_session.commit()
    
    retrieved = db_session.query(Ticker).filter_by(symbol="AAPL").first()
    assert retrieved is not None
    assert retrieved.company_name == "Apple Inc."
    assert retrieved.active is True


def test_ticker_unique_symbol(db_session: Session) -> None:
    """Test that ticker symbol is unique."""
    ticker1 = Ticker(symbol="AAPL", exchange="NASDAQ")
    ticker2 = Ticker(symbol="AAPL", exchange="NYSE")
    
    db_session.add(ticker1)
    db_session.commit()
    
    db_session.add(ticker2)
    with pytest.raises(Exception):
        db_session.commit()


def test_price_bar_model(db_session: Session) -> None:
    """Test PriceBar model with ticker relationship."""
    ticker = Ticker(symbol="AAPL", exchange="NASDAQ")
    db_session.add(ticker)
    db_session.commit()
    
    price_bar = PriceBar(
        ticker_id=ticker.ticker_id,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000000.0,
        interval="1d",
    )
    db_session.add(price_bar)
    db_session.commit()
    
    retrieved = db_session.query(PriceBar).filter_by(ticker_id=ticker.ticker_id).first()
    assert retrieved is not None
    assert retrieved.close == 151.0
    assert retrieved.interval == "1d"


def test_price_bar_unique_constraint(db_session: Session) -> None:
    """Test that PriceBar has unique constraint on ticker+interval+timestamp."""
    ticker = Ticker(symbol="AAPL", exchange="NASDAQ")
    db_session.add(ticker)
    db_session.commit()
    
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bar1 = PriceBar(
        ticker_id=ticker.ticker_id,
        timestamp=timestamp,
        close=150.0,
        interval="1d",
    )
    bar2 = PriceBar(
        ticker_id=ticker.ticker_id,
        timestamp=timestamp,
        close=151.0,
        interval="1d",
    )
    
    db_session.add(bar1)
    db_session.commit()
    
    db_session.add(bar2)
    with pytest.raises(Exception):
        db_session.commit()


def test_news_article_model(db_session: Session) -> None:
    """Test NewsArticle model with deduplication fields."""
    article = NewsArticle(
        title="Apple announces new product",
        content="Full article content here",
        source="TechNews",
        url="https://example.com/article",
        content_hash="abc123def456",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        dedupe_reason="exact_hash",
        cluster_id=1,
    )
    db_session.add(article)
    db_session.commit()
    
    retrieved = db_session.query(NewsArticle).filter_by(content_hash="abc123def456").first()
    assert retrieved is not None
    assert retrieved.title == "Apple announces new product"
    assert retrieved.dedupe_reason == "exact_hash"
    assert retrieved.cluster_id == 1


def test_news_article_unique_content_hash(db_session: Session) -> None:
    """Test that content_hash is unique."""
    article1 = NewsArticle(
        title="Article 1",
        content_hash="same_hash",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    article2 = NewsArticle(
        title="Article 2",
        content_hash="same_hash",
        published_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    
    db_session.add(article1)
    db_session.commit()
    
    db_session.add(article2)
    with pytest.raises(Exception):
        db_session.commit()


def test_news_mention_model(db_session: Session) -> None:
    """Test NewsMention model linking articles to tickers."""
    ticker = Ticker(symbol="AAPL", exchange="NASDAQ")
    article = NewsArticle(
        title="Apple news",
        content_hash="hash123",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add_all([ticker, article])
    db_session.commit()
    
    mention = NewsMention(
        article_id=article.article_id,
        ticker_id=ticker.ticker_id,
        relevance_score=0.95,
        sentiment_score=0.8,
    )
    db_session.add(mention)
    db_session.commit()
    
    retrieved = db_session.query(NewsMention).filter_by(article_id=article.article_id).first()
    assert retrieved is not None
    assert retrieved.relevance_score == 0.95
    assert retrieved.sentiment_score == 0.8


def test_event_model(db_session: Session) -> None:
    """Test Event model for detected events."""
    ticker = Ticker(symbol="AAPL", exchange="NASDAQ")
    article = NewsArticle(
        title="Apple buyback",
        content_hash="hash456",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add_all([ticker, article])
    db_session.commit()
    
    event = Event(
        event_type="buyback",
        ticker_id=ticker.ticker_id,
        article_id=article.article_id,
        detected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        confidence_score=0.92,
        sentiment="positive",
    )
    db_session.add(event)
    db_session.commit()
    
    retrieved = db_session.query(Event).filter_by(event_type="buyback").first()
    assert retrieved is not None
    assert retrieved.confidence_score == 0.92
    assert retrieved.sentiment == "positive"


def test_event_argument_model(db_session: Session) -> None:
    """Test EventArgument model for event attributes."""
    ticker = Ticker(symbol="AAPL", exchange="NASDAQ")
    db_session.add(ticker)
    db_session.commit()
    
    event = Event(
        event_type="buyback",
        ticker_id=ticker.ticker_id,
        detected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add(event)
    db_session.commit()
    
    arg = EventArgument(
        event_id=event.event_id,
        arg_name="amount",
        arg_value="$10B",
        confidence=0.88,
    )
    db_session.add(arg)
    db_session.commit()
    
    retrieved = db_session.query(EventArgument).filter_by(event_id=event.event_id).first()
    assert retrieved is not None
    assert retrieved.arg_name == "amount"
    assert retrieved.arg_value == "$10B"


def test_event_impact_model(db_session: Session) -> None:
    """Test EventImpact model for measured impacts."""
    ticker = Ticker(symbol="AAPL", exchange="NASDAQ")
    db_session.add(ticker)
    db_session.commit()
    
    event = Event(
        event_type="earnings",
        ticker_id=ticker.ticker_id,
        detected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add(event)
    db_session.commit()
    
    impact = EventImpact(
        event_id=event.event_id,
        abnormal_return_1d=0.05,
        abnormal_return_5d=0.08,
        abnormal_return_20d=0.12,
        volume_spike=2.5,
        volatility_spike=1.8,
    )
    db_session.add(impact)
    db_session.commit()
    
    retrieved = db_session.query(EventImpact).filter_by(event_id=event.event_id).first()
    assert retrieved is not None
    assert retrieved.abnormal_return_1d == 0.05
    assert retrieved.volume_spike == 2.5


def test_event_impact_unique_event_id(db_session: Session) -> None:
    """Test that EventImpact has unique constraint on event_id."""
    ticker = Ticker(symbol="AAPL", exchange="NASDAQ")
    db_session.add(ticker)
    db_session.commit()
    
    event = Event(
        event_type="earnings",
        ticker_id=ticker.ticker_id,
        detected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add(event)
    db_session.commit()
    
    impact1 = EventImpact(event_id=event.event_id, abnormal_return_1d=0.05)
    impact2 = EventImpact(event_id=event.event_id, abnormal_return_1d=0.06)
    
    db_session.add(impact1)
    db_session.commit()
    
    db_session.add(impact2)
    with pytest.raises(Exception):
        db_session.commit()


def test_model_version_model(db_session: Session) -> None:
    """Test ModelVersion model."""
    model = ModelVersion(
        model_type="sentiment_classifier",
        version_tag="v1.0.0",
        framework="transformers",
        hyperparams_json='{"learning_rate": 0.001}',
        metrics_json='{"f1": 0.85}',
        artifact_path="/models/sentiment_v1",
        status="production",
    )
    db_session.add(model)
    db_session.commit()
    
    retrieved = db_session.query(ModelVersion).filter_by(version_tag="v1.0.0").first()
    assert retrieved is not None
    assert retrieved.model_type == "sentiment_classifier"
    assert retrieved.status == "production"


def test_model_version_unique_type_tag(db_session: Session) -> None:
    """Test that ModelVersion has unique constraint on model_type+version_tag."""
    model1 = ModelVersion(model_type="classifier", version_tag="v1.0")
    model2 = ModelVersion(model_type="classifier", version_tag="v1.0")
    
    db_session.add(model1)
    db_session.commit()
    
    db_session.add(model2)
    with pytest.raises(Exception):
        db_session.commit()


def test_prediction_model(db_session: Session) -> None:
    """Test Prediction model with conformal intervals."""
    ticker = Ticker(symbol="AAPL", exchange="NASDAQ")
    model = ModelVersion(model_type="return_predictor", version_tag="v1.0")
    db_session.add_all([ticker, model])
    db_session.commit()
    
    prediction = Prediction(
        ticker_id=ticker.ticker_id,
        model_version_id=model.version_id,
        prediction_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        target_horizon_days=5,
        predicted_return=0.03,
        confidence_lower=0.01,
        confidence_upper=0.05,
        coverage_level=0.95,
        features_json='{"rsi": 55, "volume": 1000000}',
    )
    db_session.add(prediction)
    db_session.commit()
    
    retrieved = db_session.query(Prediction).filter_by(ticker_id=ticker.ticker_id).first()
    assert retrieved is not None
    assert retrieved.predicted_return == 0.03
    assert retrieved.confidence_lower == 0.01
    assert retrieved.confidence_upper == 0.05
    assert retrieved.coverage_level == 0.95
