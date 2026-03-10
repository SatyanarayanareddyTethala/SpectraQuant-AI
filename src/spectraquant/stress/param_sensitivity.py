"""Parameter sensitivity stress testing."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def _set_nested(config: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _score_returns(returns: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0}
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    vol = returns.std() * np.sqrt(252)
    mean_return = returns.mean() * 252
    sharpe = mean_return / vol if vol not in (0, np.nan) else 0.0
    hit_rate = float((returns > 0).mean())
    return {
        "sharpe": float(sharpe) if sharpe == sharpe else 0.0,
        "max_drawdown": float(max_drawdown) if max_drawdown == max_drawdown else 0.0,
        "hit_rate": hit_rate,
    }


def run_param_sensitivity(
    base_config: dict,
    param_grid: dict,
    run_pipeline_fn: Callable[[dict], dict],
) -> pd.DataFrame:
    if not param_grid:
        raise ValueError("param_grid must not be empty")

    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]

    rows: list[dict[str, Any]] = []
    for combo in product(*values):
        cfg = deepcopy(base_config)
        for key, val in zip(keys, combo):
            _set_nested(cfg, key, val)

        result = run_pipeline_fn(cfg)
        metrics = result.get("metrics") if isinstance(result, dict) else None
        if metrics is None and isinstance(result, dict) and "returns" in result:
            metrics = _score_returns(pd.Series(result["returns"]))
        if metrics is None:
            metrics = {}
        row = {"param_set": "|".join(f"{k}={v}" for k, v in zip(keys, combo))}
        row.update({key: val for key, val in zip(keys, combo)})
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("reports/stress")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"param_sensitivity_{run_id}.csv"
    df.to_csv(output_path, index=False)
    return df


__all__ = ["run_param_sensitivity"]
