from __future__ import annotations

import numpy as np
import pandas as pd

def generate_ab_experiment(
    n_control: int,
    n_treatment: int,
    control_conversion_rate: float,
    treatment_conversion_rate: float,
    control_revenue_mean: float,
    treatment_revenue_mean: float,
    revenue_std: float,
    seed: int | None = None,
) -> pd.DataFrame:
    '''
    Generate synthetic A/B experiment data for a product or ML experiment.

    Columns:
    - user_id
    - group
    - converted
    - revenue
    '''
    rng = np.random.default_rng(seed)

    control_converted = rng.binomial(1, control_conversion_rate, n_control)
    treatment_converted = rng.binomial(1, treatment_conversion_rate, n_treatment)

    control_revenue = np.maximum(
        rng.normal(control_revenue_mean, revenue_std, n_control), 0
    )
    treatment_revenue = np.maximum(
        rng.normal(treatment_revenue_mean, revenue_std, n_treatment), 0
    )

    control_df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_control + 1),
            "group": "control",
            "converted": control_converted,
            "revenue": control_revenue,
        }
    )

    treatment_df = pd.DataFrame(
        {
            "user_id": np.arange(n_control + 1, n_control + n_treatment + 1),
            "group": "treatment",
            "converted": treatment_converted,
            "revenue": treatment_revenue,
        }
    )

    return pd.concat([control_df, treatment_df], ignore_index=True)
