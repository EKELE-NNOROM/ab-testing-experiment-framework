from experimentation_framework.io_utils import save_dataframe
from experimentation_framework.power import estimate_required_sample_size_binary
from experimentation_framework.simulation import generate_ab_experiment
from experimentation_framework.stats import (
    bootstrap_uplift_ci,
    evaluate_binary_metric,
    evaluate_continuous_metric,
)

def main() -> None:
    df = generate_ab_experiment(
        n_control=8000,
        n_treatment=8000,
        control_conversion_rate=0.110,
        treatment_conversion_rate=0.121,
        control_revenue_mean=42.0,
        treatment_revenue_mean=44.0,
        revenue_std=20.0,
        seed=42,
    )

    save_dataframe(df, "data/sample_experiment.csv")

    print("=" * 80)
    print("A/B TEST DEMO")
    print("=" * 80)

    binary_result = evaluate_binary_metric(
        df=df,
        group_col="group",
        outcome_col="converted",
        control_label="control",
        treatment_label="treatment",
        alpha=0.05,
    )

    print("\n[1] Binary metric: Conversion")
    for key, value in binary_result.items():
        print(f"{key}: {value}")

    revenue_result = evaluate_continuous_metric(
        df=df,
        group_col="group",
        outcome_col="revenue",
        control_label="control",
        treatment_label="treatment",
        alpha=0.05,
    )

    print("\n[2] Continuous metric: Revenue")
    for key, value in revenue_result.items():
        print(f"{key}: {value}")

    ci_low, ci_high = bootstrap_uplift_ci(
        df=df,
        group_col="group",
        outcome_col="revenue",
        control_label="control",
        treatment_label="treatment",
        n_bootstrap=2000,
        random_state=42,
    )
    print("\n[3] Bootstrap CI for revenue uplift (treatment - control)")
    print(f"95% CI: ({round(ci_low, 4)}, {round(ci_high, 4)})")

    required_n = estimate_required_sample_size_binary(
        baseline_rate=0.11,
        minimum_detectable_effect=0.011,
        alpha=0.05,
        power=0.80,
    )
    print("\n[4] Power analysis")
    print(f"Estimated sample size per variant for binary metric: {required_n}")

    print("\nInterpretation:")
    print(
        "Use statistically significant uplift plus business impact "
        "(e.g., revenue lift, conversion lift) to support product decisions."
    )

if __name__ == "__main__":
    main()
