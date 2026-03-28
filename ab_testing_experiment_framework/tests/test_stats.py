import pandas as pd

from experimentation_framework.stats import (
    bootstrap_uplift_ci,
    evaluate_binary_metric,
    evaluate_continuous_metric,
)

def test_evaluate_binary_metric() -> None:
    df = pd.DataFrame(
        {
            "group": ["control"] * 5 + ["treatment"] * 5,
            "converted": [0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        }
    )

    result = evaluate_binary_metric(df, "group", "converted")
    assert "p_value" in result
    assert result["control_rate"] == 0.2
    assert result["treatment_rate"] == 0.8

def test_evaluate_continuous_metric() -> None:
    df = pd.DataFrame(
        {
            "group": ["control"] * 5 + ["treatment"] * 5,
            "revenue": [10, 11, 9, 10, 10, 14, 15, 16, 14, 15],
        }
    )

    result = evaluate_continuous_metric(df, "group", "revenue")
    assert "p_value" in result
    assert result["treatment_mean"] > result["control_mean"]

def test_bootstrap_ci() -> None:
    df = pd.DataFrame(
        {
            "group": ["control"] * 10 + ["treatment"] * 10,
            "metric": [1] * 10 + [2] * 10,
        }
    )
    low, high = bootstrap_uplift_ci(
        df=df,
        group_col="group",
        outcome_col="metric",
        n_bootstrap=200,
        random_state=42,
    )
    assert low <= high
