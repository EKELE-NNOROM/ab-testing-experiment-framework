from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

def _extract_groups(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label: str,
    treatment_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    control = df.loc[df[group_col] == control_label, outcome_col].to_numpy()
    treatment = df.loc[df[group_col] == treatment_label, outcome_col].to_numpy()

    if len(control) == 0 or len(treatment) == 0:
        raise ValueError("Both control and treatment groups must contain records.")

    return control, treatment

def evaluate_binary_metric(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label: str = "control",
    treatment_label: str = "treatment",
    alpha: float = 0.05,
) -> dict[str, Any]:
    control, treatment = _extract_groups(
        df, group_col, outcome_col, control_label, treatment_label
    )

    control_success = int(control.sum())
    treatment_success = int(treatment.sum())
    control_n = len(control)
    treatment_n = len(treatment)

    stat, p_value = proportions_ztest(
        count=[treatment_success, control_success],
        nobs=[treatment_n, control_n],
        alternative="two-sided",
    )

    control_rate = control_success / control_n
    treatment_rate = treatment_success / treatment_n
    abs_uplift = treatment_rate - control_rate
    rel_uplift = abs_uplift / control_rate if control_rate != 0 else np.nan

    ci_low, ci_high = confint_proportions_2indep(
        count1=treatment_success,
        nobs1=treatment_n,
        count2=control_success,
        nobs2=control_n,
        method="wald",
        compare="diff",
        alpha=alpha,
    )

    return {
        "metric_type": "binary",
        "control_rate": round(control_rate, 6),
        "treatment_rate": round(treatment_rate, 6),
        "absolute_uplift": round(abs_uplift, 6),
        "relative_uplift_pct": round(rel_uplift * 100, 4),
        "z_statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "ci_low": round(float(ci_low), 6),
        "ci_high": round(float(ci_high), 6),
        "is_significant": bool(p_value < alpha),
    }

def evaluate_continuous_metric(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label: str = "control",
    treatment_label: str = "treatment",
    alpha: float = 0.05,
) -> dict[str, Any]:
    control, treatment = _extract_groups(
        df, group_col, outcome_col, control_label, treatment_label
    )

    stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    control_mean = float(np.mean(control))
    treatment_mean = float(np.mean(treatment))
    abs_uplift = treatment_mean - control_mean
    rel_uplift = abs_uplift / control_mean if control_mean != 0 else np.nan

    ci_low, ci_high = _welch_confidence_interval(
        treatment=treatment,
        control=control,
        alpha=alpha,
    )

    return {
        "metric_type": "continuous",
        "control_mean": round(control_mean, 6),
        "treatment_mean": round(treatment_mean, 6),
        "absolute_uplift": round(abs_uplift, 6),
        "relative_uplift_pct": round(rel_uplift * 100, 4),
        "t_statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "ci_low": round(float(ci_low), 6),
        "ci_high": round(float(ci_high), 6),
        "is_significant": bool(p_value < alpha),
    }

def _welch_confidence_interval(
    treatment: np.ndarray,
    control: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    mean_diff = np.mean(treatment) - np.mean(control)

    var_t = np.var(treatment, ddof=1)
    var_c = np.var(control, ddof=1)
    n_t = len(treatment)
    n_c = len(control)

    se = np.sqrt(var_t / n_t + var_c / n_c)

    df_num = (var_t / n_t + var_c / n_c) ** 2
    df_den = ((var_t / n_t) ** 2) / (n_t - 1) + ((var_c / n_c) ** 2) / (n_c - 1)
    dof = df_num / df_den

    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    margin = t_crit * se

    return float(mean_diff - margin), float(mean_diff + margin)

def bootstrap_uplift_ci(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label: str = "control",
    treatment_label: str = "treatment",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float]:
    control, treatment = _extract_groups(
        df, group_col, outcome_col, control_label, treatment_label
    )

    rng = np.random.default_rng(random_state)
    uplifts = []

    for _ in range(n_bootstrap):
        control_sample = rng.choice(control, size=len(control), replace=True)
        treatment_sample = rng.choice(treatment, size=len(treatment), replace=True)
        uplifts.append(np.mean(treatment_sample) - np.mean(control_sample))

    lower = np.quantile(uplifts, alpha / 2)
    upper = np.quantile(uplifts, 1 - alpha / 2)
    return float(lower), float(upper)
