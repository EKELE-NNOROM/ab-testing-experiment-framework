from math import ceil
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

def estimate_required_sample_size_binary(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    '''
    Estimate required sample size per group for a binary metric.

    Example:
    baseline_rate = 0.11
    minimum_detectable_effect = 0.011  # absolute uplift of 1.1 percentage points
    '''
    treatment_rate = baseline_rate + minimum_detectable_effect
    effect_size = proportion_effectsize(baseline_rate, treatment_rate)

    analysis = NormalIndPower()
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return ceil(sample_size)
