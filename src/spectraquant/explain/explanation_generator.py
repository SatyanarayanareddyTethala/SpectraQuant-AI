"""Explanation generator for portfolio and prediction rationale.

This module generates structured JSON explanations with top events,
mechanistic templates, evidence links, and confidence intervals.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generates structured explanations for predictions and portfolios."""

    def __init__(
        self,
        template_dir: str | Path | None = None,
    ) -> None:
        """Initialize explanation generator.
        
        Args:
            template_dir: Directory containing explanation templates
        """
        self.template_dir = Path(template_dir) if template_dir else None
        logger.info("Initializing explanation generator")
    
    def generate_explanation(
        self,
        ticker: str,
        prediction: float,
        confidence_interval: tuple[float, float],
        events: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        alternatives: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate structured explanation.
        
        Args:
            ticker: Stock symbol
            prediction: Predicted value
            confidence_interval: (lower, upper) bounds
            events: List of detected events
            evidence: List of evidence items (news articles, etc.)
            alternatives: Optional alternative scenarios
            
        Returns:
            Structured explanation dictionary
        """
        explanation = {
            "ticker": ticker,
            "prediction": {
                "value": prediction,
                "confidence_interval": {
                    "lower": confidence_interval[0],
                    "upper": confidence_interval[1],
                },
            },
            "top_events": self._rank_events(events)[:5],
            "mechanistic_text": self._generate_mechanistic_text(ticker, events),
            "evidence_links": self._format_evidence(evidence),
            "alternatives": alternatives or [],
            "generated_at": datetime.now().isoformat(),
        }
        
        return explanation
    
    def _rank_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank events by importance."""
        # Sort by confidence score (descending)
        return sorted(events, key=lambda e: e.get("confidence", 0), reverse=True)
    
    def _generate_mechanistic_text(
        self,
        ticker: str,
        events: list[dict[str, Any]],
    ) -> str:
        """Generate human-readable mechanistic explanation.
        
        Args:
            ticker: Stock symbol
            events: List of events
            
        Returns:
            Mechanistic explanation text
        """
        if not events:
            return f"No significant events detected for {ticker}."
        
        top_event = events[0]
        event_type = top_event.get("event_type", "unknown")
        sentiment = top_event.get("sentiment", "neutral")
        
        templates = {
            "buyback": f"{ticker} announced a share buyback program, which is typically {sentiment} for the stock price.",
            "lawsuit": f"{ticker} is facing a lawsuit, which may have a {sentiment} impact on shareholder value.",
            "earnings": f"{ticker} reported earnings, with {sentiment} market reaction expected.",
            "merger": f"{ticker} is involved in merger discussions, creating {sentiment} sentiment.",
        }
        
        template = templates.get(event_type, f"{ticker} has {sentiment} news.")
        return template
    
    def _format_evidence(
        self,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Format evidence items for display.
        
        Args:
            evidence: List of evidence dictionaries
            
        Returns:
            Formatted evidence list
        """
        formatted = []
        for item in evidence:
            formatted.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": item.get("published_at", ""),
            })
        return formatted
    
    def save_explanation(
        self,
        explanation: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Save explanation to JSON file.
        
        Args:
            explanation: Explanation dictionary
            output_path: Path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(explanation, f, indent=2)
        
        logger.info("Saved explanation to %s", output_path)


# NOTE: Full implementation would include:
# 1. Template-based text generation
# 2. Multi-lingual support
# 3. Causal reasoning chains
# 4. Uncertainty propagation
# 5. Interactive visualization generation
