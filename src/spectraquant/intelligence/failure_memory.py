"""Failure Memory — learn from trade mistakes.

Stores trade intents and outcomes in SQLite, labels failure types, and
produces rolling statistics to guide the meta-learner.

Failure taxonomy
----------------
MODEL_ERROR        Prediction was systematically wrong (feature drift).
TIMING_ERROR       Right direction, wrong timing (early/late entry).
NEWS_SHOCK         Unexpected news event reversed the trade.
VOLATILITY_SPIKE   Intraday volatility exceeded expectations.
OVERCONFIDENCE     Confidence score >> realised outcome.
REGIME_SHIFT       Market regime changed mid-trade.
LIQUIDITY_SLIPPAGE Actual slippage far exceeded modelled slippage.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

FAILURE_TYPES = (
    "MODEL_ERROR",
    "TIMING_ERROR",
    "NEWS_SHOCK",
    "VOLATILITY_SPIKE",
    "OVERCONFIDENCE",
    "REGIME_SHIFT",
    "LIQUIDITY_SLIPPAGE",
)

# Thresholds for automatic labelling
_OVERCONFIDENCE_THRESHOLD = 0.6   # confidence minus |realized_return|
_SLIPPAGE_THRESHOLD_BPS = 20       # bps


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Intent recorded before trade execution."""

    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = ""
    ticker: str = ""
    side: str = "long"                 # long | short
    confidence: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    news_context: Dict[str, Any] = field(default_factory=dict)
    regime: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    # Outcome fields (filled later)
    realized_return: Optional[float] = None
    mae: Optional[float] = None          # maximum adverse excursion
    mfe: Optional[float] = None          # maximum favourable excursion
    slippage_bps: Optional[float] = None
    exit_reason: Optional[str] = None
    outcome_timestamp: Optional[str] = None
    failure_type: Optional[str] = None


@dataclass
class FailureRecord:
    """Processed failure record derived from a TradeRecord."""

    failure_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trade_id: str = ""
    ticker: str = ""
    regime: str = ""
    failure_type: str = ""
    confidence: float = 0.0
    realized_return: float = 0.0
    slippage_bps: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# SQLite storage helpers
# ---------------------------------------------------------------------------

def _get_connection(storage_dir: str) -> sqlite3.Connection:
    """Return (creating if needed) a SQLite connection to the failure DB."""
    path = Path(storage_dir)
    path.mkdir(parents=True, exist_ok=True)
    db_path = path / "failure_memory.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist yet."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trade_records (
            trade_id          TEXT PRIMARY KEY,
            run_id            TEXT,
            ticker            TEXT,
            side              TEXT,
            confidence        REAL,
            features_json     TEXT,
            news_context_json TEXT,
            regime            TEXT,
            timestamp         TEXT,
            realized_return   REAL,
            mae               REAL,
            mfe               REAL,
            slippage_bps      REAL,
            exit_reason       TEXT,
            outcome_timestamp TEXT,
            failure_type      TEXT
        );

        CREATE TABLE IF NOT EXISTS failure_records (
            failure_id       TEXT PRIMARY KEY,
            trade_id         TEXT,
            ticker           TEXT,
            regime           TEXT,
            failure_type     TEXT,
            confidence       REAL,
            realized_return  REAL,
            slippage_bps     REAL,
            timestamp        TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tr_ticker  ON trade_records(ticker);
        CREATE INDEX IF NOT EXISTS idx_tr_regime  ON trade_records(regime);
        CREATE INDEX IF NOT EXISTS idx_fr_ticker  ON failure_records(ticker);
        CREATE INDEX IF NOT EXISTS idx_fr_type    ON failure_records(failure_type);
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def record_trade_intent(
    run_id: str,
    ticker: str,
    side: str,
    confidence: float,
    features: Dict[str, Any],
    news_context: Dict[str, Any],
    regime: str,
    timestamp: Optional[str] = None,
    storage_dir: str = "data/state/failure_memory",
) -> str:
    """Persist a trade intent and return the generated trade_id.

    Parameters
    ----------
    run_id : str
        Identifier of the pipeline run that generated this signal.
    ticker : str
        Ticker symbol.
    side : str
        ``"long"`` or ``"short"``.
    confidence : float
        Model confidence score (0–1).
    features : dict
        Snapshot of input features used for the prediction.
    news_context : dict
        News metadata at signal time (optional; may be empty).
    regime : str
        Market regime label at signal time.
    timestamp : str, optional
        ISO-8601 UTC timestamp; defaults to ``datetime.now(utc)``.
    storage_dir : str
        Directory for the SQLite database.

    Returns
    -------
    str
        The ``trade_id`` of the newly recorded intent.
    """
    record = TradeRecord(
        run_id=run_id,
        ticker=ticker,
        side=side,
        confidence=confidence,
        features=features,
        news_context=news_context,
        regime=regime,
        timestamp=timestamp or datetime.now(tz=timezone.utc).isoformat(),
    )
    conn = _get_connection(storage_dir)
    try:
        conn.execute(
            """
            INSERT INTO trade_records
                (trade_id, run_id, ticker, side, confidence,
                 features_json, news_context_json, regime, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                record.trade_id,
                record.run_id,
                record.ticker,
                record.side,
                record.confidence,
                json.dumps(record.features),
                json.dumps(record.news_context),
                record.regime,
                record.timestamp,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    logger.debug("Recorded trade intent %s for %s", record.trade_id, ticker)
    return record.trade_id


def record_trade_outcome(
    trade_id: str,
    realized_return: float,
    mae: float,
    mfe: float,
    slippage_bps: float,
    exit_reason: str,
    timestamp: Optional[str] = None,
    storage_dir: str = "data/state/failure_memory",
) -> None:
    """Fill in the outcome fields for an existing trade record and label failures.

    Parameters
    ----------
    trade_id : str
        Trade identifier returned by :func:`record_trade_intent`.
    realized_return : float
        Net P&L as fraction of entry price (e.g. -0.02 = -2 %).
    mae : float
        Maximum adverse excursion (negative number, e.g. -0.03).
    mfe : float
        Maximum favourable excursion (positive number, e.g. 0.05).
    slippage_bps : float
        Realised slippage in basis points.
    exit_reason : str
        Human-readable exit reason (e.g. ``"stop_hit"``, ``"target_hit"``).
    timestamp : str, optional
        ISO-8601 UTC timestamp of outcome; defaults to now.
    storage_dir : str
        Directory containing the SQLite database.
    """
    ts = timestamp or datetime.now(tz=timezone.utc).isoformat()
    conn = _get_connection(storage_dir)
    try:
        conn.execute(
            """
            UPDATE trade_records
            SET realized_return=?, mae=?, mfe=?, slippage_bps=?,
                exit_reason=?, outcome_timestamp=?
            WHERE trade_id=?
            """,
            (realized_return, mae, mfe, slippage_bps, exit_reason, ts, trade_id),
        )
        conn.commit()

        # Load the updated record
        row = conn.execute(
            "SELECT * FROM trade_records WHERE trade_id=?", (trade_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        logger.warning("record_trade_outcome: unknown trade_id=%s", trade_id)
        return

    trade = TradeRecord(
        trade_id=row["trade_id"],
        run_id=row["run_id"] or "",
        ticker=row["ticker"] or "",
        side=row["side"] or "long",
        confidence=row["confidence"] or 0.0,
        features=json.loads(row["features_json"] or "{}"),
        news_context=json.loads(row["news_context_json"] or "{}"),
        regime=row["regime"] or "",
        timestamp=row["timestamp"] or "",
        realized_return=row["realized_return"],
        mae=row["mae"],
        mfe=row["mfe"],
        slippage_bps=row["slippage_bps"],
        exit_reason=row["exit_reason"],
        outcome_timestamp=row["outcome_timestamp"],
    )

    ftype = label_failure(trade)
    if ftype:
        conn2 = _get_connection(storage_dir)
        try:
            conn2.execute(
                "UPDATE trade_records SET failure_type=? WHERE trade_id=?",
                (ftype, trade_id),
            )
            failure = FailureRecord(
                trade_id=trade_id,
                ticker=trade.ticker,
                regime=trade.regime,
                failure_type=ftype,
                confidence=trade.confidence,
                realized_return=realized_return,
                slippage_bps=slippage_bps,
                timestamp=ts,
            )
            conn2.execute(
                """
                INSERT INTO failure_records
                    (failure_id, trade_id, ticker, regime, failure_type,
                     confidence, realized_return, slippage_bps, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    failure.failure_id,
                    failure.trade_id,
                    failure.ticker,
                    failure.regime,
                    failure.failure_type,
                    failure.confidence,
                    failure.realized_return,
                    failure.slippage_bps,
                    failure.timestamp,
                ),
            )
            conn2.commit()
        finally:
            conn2.close()

    logger.debug(
        "Recorded outcome for %s: return=%.4f failure=%s",
        trade_id,
        realized_return,
        ftype or "none",
    )


def label_failure(trade: TradeRecord) -> Optional[str]:
    """Classify why a trade failed; returns *None* for winners.

    Parameters
    ----------
    trade : TradeRecord
        Record with outcome fields populated.

    Returns
    -------
    str or None
        One of :data:`FAILURE_TYPES`, or *None* if the trade was not a failure.
    """
    ret = trade.realized_return
    if ret is None:
        return None  # outcome not yet recorded

    # Not a failure
    if ret >= 0:
        return None

    conf = trade.confidence or 0.0
    slip = trade.slippage_bps or 0.0
    news = trade.news_context or {}
    exit_r = (trade.exit_reason or "").lower()

    # News shock — unexpected news triggered exit
    if news.get("risk_score", 0.0) > 0.7 or "news" in exit_r:
        return "NEWS_SHOCK"

    # Overconfidence — high confidence but bad outcome
    if conf - abs(ret) > _OVERCONFIDENCE_THRESHOLD:
        return "OVERCONFIDENCE"

    # Liquidity/slippage
    if slip > _SLIPPAGE_THRESHOLD_BPS:
        return "LIQUIDITY_SLIPPAGE"

    # Regime shift
    if "regime" in exit_r or trade.regime.startswith("PANIC"):
        return "REGIME_SHIFT"

    # Timing — MAE shows the trade went the right direction eventually
    mfe = trade.mfe or 0.0
    mae = trade.mae or 0.0
    if mfe > abs(ret) * 1.5 and abs(mae) < abs(ret):
        return "TIMING_ERROR"

    # Volatility spike
    if abs(mae) > abs(ret) * 3:
        return "VOLATILITY_SPIKE"

    # Default: model error
    return "MODEL_ERROR"


def update_failure_stats(
    storage_dir: str = "data/state/failure_memory",
) -> Dict[str, Any]:
    """Compute rolling failure metrics per ticker / regime / failure_type.

    Parameters
    ----------
    storage_dir : str
        Directory containing the SQLite database.

    Returns
    -------
    dict
        Nested metrics: ``{ticker: {failure_type: count}}``,
        ``{regime: {failure_type: count}}``, total_trades, total_failures.
    """
    conn = _get_connection(storage_dir)
    try:
        rows = conn.execute(
            "SELECT ticker, regime, failure_type, COUNT(*) as cnt "
            "FROM failure_records GROUP BY ticker, regime, failure_type"
        ).fetchall()
        total_trades = conn.execute(
            "SELECT COUNT(*) FROM trade_records"
        ).fetchone()[0]
        total_failures = conn.execute(
            "SELECT COUNT(*) FROM failure_records"
        ).fetchone()[0]
    finally:
        conn.close()

    by_ticker: Dict[str, Dict[str, int]] = {}
    by_regime: Dict[str, Dict[str, int]] = {}

    for row in rows:
        ticker = row["ticker"] or "UNKNOWN"
        regime = row["regime"] or "UNKNOWN"
        ftype = row["failure_type"] or "UNKNOWN"
        cnt = row["cnt"]

        by_ticker.setdefault(ticker, {})[ftype] = (
            by_ticker.get(ticker, {}).get(ftype, 0) + cnt
        )
        by_regime.setdefault(regime, {})[ftype] = (
            by_regime.get(regime, {}).get(ftype, 0) + cnt
        )

    return {
        "total_trades": total_trades,
        "total_failures": total_failures,
        "by_ticker": by_ticker,
        "by_regime": by_regime,
    }


def export_failure_report(
    path: str,
    storage_dir: str = "data/state/failure_memory",
) -> None:
    """Write a JSON failure report to *path*.

    Parameters
    ----------
    path : str
        Output file path (e.g. ``reports/intelligence/failure_report.json``).
    storage_dir : str
        Directory containing the SQLite database.
    """
    stats = update_failure_stats(storage_dir=storage_dir)
    stats["generated_at"] = datetime.now(tz=timezone.utc).isoformat()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(stats, fh, indent=2)

    logger.info("Failure report written to %s", path)
