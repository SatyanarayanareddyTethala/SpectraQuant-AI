"""Policy enforcement: trigger evaluation and do-not-trade rules."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trigger evaluator
# ---------------------------------------------------------------------------

class TriggerEvaluator:
    """Evaluate whether a planned trade's trigger conditions are met.

    Supported trigger types: *price*, *breakout*, *momentum*, *volume*.
    """

    def evaluate(
        self,
        bars_5m: List[Dict[str, Any]],
        plan_trade: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate triggers for a single planned trade against recent bars.

        Parameters
        ----------
        bars_5m : list[dict]
            Recent 5-minute bars, each with keys:
            ``open, high, low, close, volume, ts``.
        plan_trade : dict
            Planned trade with keys:
            ``symbol, direction, entry, stop_loss, target, trigger_json``.

        Returns
        -------
        list[dict]
            Fired triggers, each with ``type``, ``triggered_at``, ``details``.
        """
        if not bars_5m:
            return []

        triggers: List[Dict[str, Any]] = []
        trigger_spec = plan_trade.get("trigger_json") or {}
        now = datetime.now(tz=timezone.utc)

        latest = bars_5m[-1]
        price = latest.get("close", 0.0)
        direction = plan_trade.get("direction", "long")

        # --- Price trigger ---------------------------------------------------
        if "price" in trigger_spec:
            target_price = trigger_spec["price"].get("target", 0.0)
            if direction == "long" and price >= target_price:
                triggers.append(self._fire("price", now, {"price": price, "target": target_price}))
            elif direction == "short" and price <= target_price:
                triggers.append(self._fire("price", now, {"price": price, "target": target_price}))

        # --- Breakout trigger -------------------------------------------------
        if "breakout" in trigger_spec:
            lookback = trigger_spec["breakout"].get("lookback", 20)
            window = bars_5m[-lookback:] if len(bars_5m) >= lookback else bars_5m
            highs = [b.get("high", 0.0) for b in window[:-1]] or [0.0]
            lows = [b.get("low", float("inf")) for b in window[:-1]] or [float("inf")]
            resistance = max(highs)
            support = min(lows)
            if direction == "long" and price > resistance:
                triggers.append(self._fire("breakout", now, {"price": price, "resistance": resistance}))
            elif direction == "short" and price < support:
                triggers.append(self._fire("breakout", now, {"price": price, "support": support}))

        # --- Momentum trigger -------------------------------------------------
        if "momentum" in trigger_spec:
            period = trigger_spec["momentum"].get("period", 5)
            threshold = trigger_spec["momentum"].get("threshold", 0.0)
            if len(bars_5m) >= period + 1:
                old_close = bars_5m[-(period + 1)].get("close", 0.0)
                if old_close > 0:
                    momentum = (price - old_close) / old_close
                    if direction == "long" and momentum >= threshold:
                        triggers.append(self._fire("momentum", now, {"momentum": momentum, "threshold": threshold}))
                    elif direction == "short" and momentum <= -threshold:
                        triggers.append(self._fire("momentum", now, {"momentum": momentum, "threshold": threshold}))

        # --- Volume trigger ---------------------------------------------------
        if "volume" in trigger_spec:
            multiplier = trigger_spec["volume"].get("multiplier", 2.0)
            lookback_v = trigger_spec["volume"].get("lookback", 20)
            window_v = bars_5m[-lookback_v:] if len(bars_5m) >= lookback_v else bars_5m
            avg_vol = (
                sum(b.get("volume", 0) for b in window_v[:-1]) / max(len(window_v) - 1, 1)
            )
            current_vol = latest.get("volume", 0)
            if avg_vol > 0 and current_vol >= avg_vol * multiplier:
                triggers.append(
                    self._fire("volume", now, {"volume": current_vol, "avg_volume": avg_vol, "multiplier": multiplier})
                )

        return triggers

    @staticmethod
    def _fire(ttype: str, ts: datetime, details: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": ttype, "triggered_at": ts.isoformat(), "details": details}


# ---------------------------------------------------------------------------
# Policy rules
# ---------------------------------------------------------------------------

class PolicyRules:
    """Enforce do-not-trade conditions before order submission."""

    def do_not_trade_checks(
        self,
        symbol: str,
        current_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[str]:
        """Return a list of reasons the trade should be blocked.

        An empty list means the trade is allowed.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        current_state : dict
            Keys may include ``adv``, ``spread_bps``, ``news_risk_score``,
            ``daily_pnl``, ``volatility``.
        config : dict
            Risk / policy parameters (``min_adv``, ``max_spread_bps``,
            ``news_risk_threshold``, ``max_daily_loss``, ``max_volatility``).
        """
        reasons: List[str] = []

        # Liquidity check
        min_adv = config.get("min_adv", 100_000)
        adv = current_state.get("adv", 0)
        if adv < min_adv:
            reasons.append(f"ADV too low: {adv:,.0f} < {min_adv:,.0f}")

        # Spread check
        max_spread = config.get("max_spread_bps", 30.0)
        spread = current_state.get("spread_bps", 0.0)
        if spread > max_spread:
            reasons.append(f"Spread too wide: {spread:.1f} bps > {max_spread:.1f} bps")

        # News-shock check
        threshold = config.get("news_risk_threshold", 0.7)
        news_score = current_state.get("news_risk_score", 0.0)
        if news_score > threshold:
            reasons.append(f"News risk elevated: {news_score:.2f} > {threshold:.2f}")

        # Daily loss check
        max_loss = config.get("max_daily_loss", 2_000.0)
        daily_pnl = current_state.get("daily_pnl", 0.0)
        if daily_pnl <= -abs(max_loss):
            reasons.append(f"Daily loss limit hit: {daily_pnl:,.2f}")

        # Extreme volatility check
        max_vol = config.get("max_volatility", 1.0)
        vol = current_state.get("volatility", 0.0)
        if vol > max_vol:
            reasons.append(f"Volatility too high: {vol:.2f} > {max_vol:.2f}")

        if reasons:
            logger.info("DNT for %s: %s", symbol, "; ".join(reasons))

        return reasons
