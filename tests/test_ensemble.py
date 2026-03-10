import pandas as pd

from spectraquant.models.ensemble import compute_ensemble_scores


def test_ensemble_determinism():
    df = pd.DataFrame(
        {
            "date": ["2023-01-02", "2023-01-02", "2023-01-03", "2023-01-03"],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "prob_up_5d": [0.6, 0.4, 0.7, 0.2],
            "pred_ret_5d": [0.02, -0.01, 0.03, -0.02],
        }
    )
    scored_a = compute_ensemble_scores(df)
    scored_b = compute_ensemble_scores(df)
    assert scored_a["ensemble_score"].equals(scored_b["ensemble_score"])
