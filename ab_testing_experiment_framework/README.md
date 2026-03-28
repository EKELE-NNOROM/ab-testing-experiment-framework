# Experimentation Framework for A/B Testing and Hypothesis Testing

A production-style Python project you can publish on GitHub to demonstrate:

- A/B testing workflows
- Hypothesis testing for binary and continuous metrics
- Confidence intervals
- Bootstrap-based uncertainty estimation
- Power and minimum detectable effect (MDE) analysis
- Synthetic experiment generation
- Reusable, testable code structure

## Project structure

```text
ab-testing-experiment-framework/
├── README.md
├── requirements.txt
├── run_demo.py
├── data/
│   └── sample_experiment.csv
├── src/
│   └── experimentation_framework/
│       ├── __init__.py
│       ├── io_utils.py
│       ├── power.py
│       ├── simulation.py
│       └── stats.py
└── tests/
    └── test_stats.py
```

## Use cases

This project is designed to support resume bullets like:

> Developed and applied statistical analysis and experimentation frameworks (A/B testing, hypothesis testing) to evaluate model performance and drive data-informed product and business decisions.

## What it does

The framework supports common experiment workflows:

1. Generate or ingest experiment data
2. Compare treatment vs control
3. Run:
   - two-sample t-test for continuous outcomes
   - two-proportion z-test for binary outcomes
   - bootstrap confidence intervals for uplift
4. Estimate power and minimum detectable effect
5. Output interpretable business summaries

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Run the demo

```bash
python run_demo.py
```

## Expected demo output

The demo simulates an A/B test with:
- a binary conversion metric
- a continuous revenue metric

It prints:
- baseline and treatment metrics
- absolute and relative uplift
- p-values
- confidence intervals
- power / sample size guidance

## Run tests

```bash
pytest
```

## Extend this project

You can easily extend this project with:
- CUPED adjustment
- sequential testing
- Bayesian A/B testing
- multiple testing correction
- support for experiment platforms / feature flags
- dashboards via Streamlit

## GitHub positioning

Suggested repo name:

`ab-testing-experiment-framework`

Suggested GitHub description:

`A reusable Python framework for A/B testing, hypothesis testing, confidence intervals, and power analysis for product, ML, and experimentation workflows.`
