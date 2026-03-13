"""Analog Market Memory — embed and recall similar market states.

Stores (state, outcome) pairs using compact numeric embeddings derived from:
- Market regime features
- Technical snapshot
- News embedding (if present; degrades gracefully)

Embeddings are stored in SQLite as JSON blobs (compact, no external deps).
Nearest-neighbour search uses L2 distance over the numeric feature vector.

Usage
-----
>>> mem = AnalogMarketMemory(storage_dir="data/state/analog_memory")
>>> state = {"regime": "TRENDING", "rsi": 55.0, "atr_pct": 0.02, ...}
>>> mem.add_state(state, outcome={"return_5d": 0.03})
>>> neighbors = mem.query_similar(state, k=20)
>>> adj_conf = mem.calibrate_confidence(0.65, neighbors)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Seed for any stochastic operations (kept deterministic)
_SEED = 42

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

# Ordered list of numeric keys extracted from a state_dict.
# Missing keys default to 0.0.  Order is fixed for embedding stability.
_FEATURE_KEYS: Tuple[str, ...] = (
    "rsi_14",
    "rsi_7",
    "atr_pct",
    "vol_20",
    "vol_5",
    "sma_slope",
    "close_vs_sma50",
    "close_vs_sma200",
    "volume_ratio",
    "adv_ratio",
    "news_sentiment",
    "news_score",
    "regime_numeric",     # encoded from string regime label
    "market_vol_proxy",   # e.g. India VIX
    "breadth",            # advance/decline ratio
)

_REGIME_ENCODING: Dict[str, float] = {
    "TRENDING": 1.0,
    "CHOPPY": 0.2,
    "RISK_ON": 0.8,
    "RISK_OFF": -0.8,
    "EVENT_DRIVEN": 0.5,
    "PANIC": -1.0,
    # Legacy labels from simple_regime
    "LOW_VOL_TREND": 0.9,
    "LOW_VOL_CHOP": 0.1,
    "HIGH_VOL_TREND": 0.6,
    "HIGH_VOL_CHOP": -0.3,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AnalogEntry:
    """A stored (state, outcome) pair."""

    entry_id: str
    embedding: np.ndarray        # shape (len(_FEATURE_KEYS),)
    state_json: str              # original state dict as JSON
    outcome_json: str            # outcome dict as JSON
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class AnalogNeighbor:
    """A retrieved neighbor with distance and outcome."""

    entry_id: str
    distance: float
    state: Dict[str, Any]
    outcome: Dict[str, Any]
    timestamp: str


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_connection(storage_dir: str) -> sqlite3.Connection:
    path = Path(storage_dir)
    path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path / "analog_memory.db"), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS analog_entries (
            entry_id     TEXT PRIMARY KEY,
            embedding    TEXT NOT NULL,
            state_json   TEXT NOT NULL,
            outcome_json TEXT NOT NULL,
            timestamp    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ae_ts ON analog_entries(timestamp);
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_state(state_dict: Dict[str, Any]) -> np.ndarray:
    """Convert a state dictionary to a fixed-length numeric vector.

    Parameters
    ----------
    state_dict : dict
        Market state snapshot.  Unknown keys are ignored; missing keys
        are filled with 0.0.

    Returns
    -------
    np.ndarray
        Shape ``(len(_FEATURE_KEYS),)``, dtype float32.
    """
    regime_str = str(state_dict.get("regime", ""))
    regime_val = _REGIME_ENCODING.get(regime_str, 0.0)

    vec = []
    for key in _FEATURE_KEYS:
        if key == "regime_numeric":
            vec.append(regime_val)
        else:
            val = state_dict.get(key, 0.0)
            try:
                vec.append(float(val))
            except (TypeError, ValueError):
                vec.append(0.0)

    return np.array(vec, dtype=np.float32)


def _make_entry_id(state_dict: Dict[str, Any], timestamp: str) -> str:
    """Deterministic ID: SHA-1 of state JSON + timestamp (first 16 chars)."""
    payload = json.dumps(state_dict, sort_keys=True) + timestamp
    return hashlib.sha1(payload.encode()).hexdigest()[:16]  # noqa: S324


# ---------------------------------------------------------------------------
# AnalogMarketMemory class
# ---------------------------------------------------------------------------

class AnalogMarketMemory:
    """Persistent analog market memory backed by SQLite.

    Parameters
    ----------
    storage_dir : str
        Directory where ``analog_memory.db`` is stored.
    max_items : int
        Maximum number of entries to keep (oldest dropped on overflow).
    """

    def __init__(
        self,
        storage_dir: str = "data/state/analog_memory",
        max_items: int = 10_000,
    ) -> None:
        self.storage_dir = storage_dir
        self.max_items = max_items
        # In-memory cache: list of (embedding, entry_id)
        self._cache: List[Tuple[np.ndarray, str, Dict, Dict, str]] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load all embeddings into memory cache."""
        if self._loaded:
            return
        conn = _get_connection(self.storage_dir)
        try:
            rows = conn.execute(
                "SELECT entry_id, embedding, state_json, outcome_json, timestamp "
                "FROM analog_entries ORDER BY timestamp DESC LIMIT ?",
                (self.max_items,),
            ).fetchall()
        finally:
            conn.close()

        self._cache = []
        for row in rows:
            emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
            state = json.loads(row["state_json"])
            outcome = json.loads(row["outcome_json"])
            self._cache.append((emb, row["entry_id"], state, outcome, row["timestamp"]))

        self._loaded = True
        logger.debug("AnalogMarketMemory: loaded %d entries", len(self._cache))

    def _prune(self, conn: sqlite3.Connection) -> None:
        """Remove oldest entries if over max_items."""
        count = conn.execute("SELECT COUNT(*) FROM analog_entries").fetchone()[0]
        if count > self.max_items:
            excess = count - self.max_items
            conn.execute(
                "DELETE FROM analog_entries WHERE entry_id IN "
                "(SELECT entry_id FROM analog_entries ORDER BY timestamp ASC LIMIT ?)",
                (excess,),
            )
            conn.commit()
            self._loaded = False  # invalidate cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_state(
        self,
        state: Dict[str, Any],
        outcome: Dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> str:
        """Store a (state, outcome) pair.

        Parameters
        ----------
        state : dict
            Market state snapshot (see :func:`encode_state`).
        outcome : dict
            Observed outcome; e.g. ``{"return_5d": 0.03, "direction": 1}``.
        timestamp : str, optional
            ISO-8601 UTC string; defaults to now.

        Returns
        -------
        str
            The ``entry_id`` of the stored entry.
        """
        ts = timestamp or datetime.now(tz=timezone.utc).isoformat()
        emb = encode_state(state)
        entry_id = _make_entry_id(state, ts)

        conn = _get_connection(self.storage_dir)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO analog_entries
                    (entry_id, embedding, state_json, outcome_json, timestamp)
                VALUES (?,?,?,?,?)
                """,
                (
                    entry_id,
                    json.dumps(emb.tolist()),
                    json.dumps(state),
                    json.dumps(outcome),
                    ts,
                ),
            )
            conn.commit()
            self._prune(conn)
        finally:
            conn.close()

        self._loaded = False  # invalidate cache
        logger.debug("AnalogMarketMemory: added entry %s", entry_id)
        return entry_id

    def query_similar(
        self,
        state: Dict[str, Any],
        k: int = 20,
    ) -> List[AnalogNeighbor]:
        """Return the *k* most similar stored states by L2 distance.

        Parameters
        ----------
        state : dict
            Query state snapshot.
        k : int
            Number of neighbours to return.

        Returns
        -------
        list[AnalogNeighbor]
            Sorted by distance (ascending).
        """
        self._ensure_loaded()
        if not self._cache:
            return []

        query_emb = encode_state(state)

        distances: List[Tuple[float, int]] = []
        for i, (emb, _, _, _, _) in enumerate(self._cache):
            diff = query_emb - emb
            dist = float(np.sqrt(np.dot(diff, diff)))
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        top = distances[:k]

        return [
            AnalogNeighbor(
                entry_id=self._cache[i][1],
                distance=d,
                state=self._cache[i][2],
                outcome=self._cache[i][3],
                timestamp=self._cache[i][4],
            )
            for d, i in top
        ]

    def calibrate_confidence(
        self,
        base_conf: float,
        neighbors: List[AnalogNeighbor],
        return_key: str = "return_5d",
        blend_weight: float = 0.3,
        min_neighbors: int = 3,
    ) -> float:
        """Adjust *base_conf* using historical neighbor outcomes.

        Parameters
        ----------
        base_conf : float
            Raw model confidence score (0–1).
        neighbors : list[AnalogNeighbor]
            Retrieved neighbors (from :meth:`query_similar`).
        return_key : str
            Key in neighbor outcome dict to use for calibration.
        blend_weight : float
            Weight on analog mean vs base confidence (0 = no adjustment).
        min_neighbors : int
            Minimum neighbors required to adjust confidence.

        Returns
        -------
        float
            Adjusted confidence, clamped to [0, 1].
        """
        outcomes = [
            n.outcome[return_key]
            for n in neighbors
            if return_key in n.outcome
        ]
        if len(outcomes) < min_neighbors:
            return base_conf

        # Positive-direction ratio acts as analog "confidence"
        positive_ratio = sum(1 for r in outcomes if r > 0) / len(outcomes)
        adjusted = (1.0 - blend_weight) * base_conf + blend_weight * positive_ratio
        adjusted = max(0.0, min(1.0, adjusted))
        logger.debug(
            "calibrate_confidence: base=%.3f analog_ratio=%.3f → %.3f",
            base_conf,
            positive_ratio,
            adjusted,
        )
        return adjusted

    def __len__(self) -> int:
        conn = _get_connection(self.storage_dir)
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM analog_entries"
            ).fetchone()[0]
        finally:
            conn.close()
