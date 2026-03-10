from __future__ import annotations

from typing import Protocol


class SentimentModel(Protocol):
    def score(self, text: str) -> float: ...


class LexiconSentimentModel:
    """Offline deterministic lexicon scorer."""

    POSITIVE = {"surge", "bull", "bullish", "gain", "up", "rally", "approval"}
    NEGATIVE = {"drop", "bear", "bearish", "hack", "exploit", "down", "lawsuit"}

    def score(self, text: str) -> float:
        words = [w.strip(".,:;!?()[]{}\"'").lower() for w in text.split()]
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in self.POSITIVE)
        neg = sum(1 for w in words if w in self.NEGATIVE)
        total = pos + neg
        if total == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / total))


class DeterministicSentimentScorer:
    def __init__(self, model: SentimentModel | None = None) -> None:
        self._model = model or LexiconSentimentModel()

    def score(self, text: str) -> float:
        return float(self._model.score(text))
