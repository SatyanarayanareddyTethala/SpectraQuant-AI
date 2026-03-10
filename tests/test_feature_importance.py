from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.feature_importance import parse_feature_importance


def test_parse_feature_importance_from_fixture() -> None:
    fixture_path = ROOT / "tests" / "fixtures" / "model.txt"
    df = parse_feature_importance(str(fixture_path))

    assert len(df) >= 5
    assert df["importance_pct"].sum() == pytest.approx(100.0, rel=1e-3)
    assert df["feature"].str.startswith(("sentiment_", "rsi_", "sma_")).any()
