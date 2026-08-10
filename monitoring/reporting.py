from pathlib import Path

import matplotlib.pyplot as plt

from .monitoring import build_monitoring_report
from .synthetic import SyntheticConfig


def build_report(config: SyntheticConfig = SyntheticConfig(), output_dir: Path = Path("reports")) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_monitoring_report(config)
    metrics = report["monitoring_metrics"]
    psi_values = metrics["feature_psi"]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(list(psi_values), list(psi_values.values()), color="#c94c4c")
    axes[0].axvline(0.10, color="#e6a23c", linestyle="--", label="warning")
    axes[0].axvline(0.25, color="#8f1d2c", linestyle="--", label="critical")
    axes[0].set_title("Feature drift: PSI", loc="left", fontweight="bold")
    axes[0].set_xlabel("Population stability index")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].bar(
        ["Reference", "Current"],
        [metrics["prediction_mean_reference"], metrics["prediction_mean_current"]],
        color=["#345995", "#e07a3f"],
    )
    axes[1].set_title("Prediction distribution shift", loc="left", fontweight="bold")
    axes[1].set_ylabel("Mean predicted probability")
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output_dir / "monitoring_dashboard.png", dpi=160)
    plt.close(figure)

    metric_rows = "\n".join(
        f"| {name} | {value:.6f} |" for name, value in {
            "ROC AUC": report["training_metrics"]["roc_auc"],
            "Training holdout Brier": report["training_metrics"]["brier_score"],
            "Current Brier": report["current_brier"],
            "Max feature PSI": metrics["max_feature_psi"],
            "Prediction mean shift": metrics["prediction_mean_shift"],
        }.items()
    )
    psi_rows = "\n".join(f"| {name} | {value:.6f} |" for name, value in psi_values.items())
    markdown = f"""# Default Risk Lifecycle Report

> Synthetic model lifecycle evaluation. No production model files or real data are used.

## Executive summary

**Lifecycle status:** `{report['recommendation']['status']}`  
**Recommended action:** `{report['recommendation']['action']}`

The model is trained and calibrated on historical synthetic data, then evaluated against a simulated current population. Monitoring covers feature drift, prediction shift, and calibration degradation.

## Metrics

| Metric | Value |
| --- | ---: |
{metric_rows}

## Monitoring dashboard

![Monitoring dashboard](monitoring_dashboard.png)

## Feature drift

PSI is used as a simple distribution-shift diagnostic. Thresholds are versioned in the monitoring policy and are illustrative.

| Feature | PSI |
| --- | ---: |
{psi_rows}

## Lifecycle interpretation

- `healthy`: continue monitoring;
- `warning`: start shadow monitoring and investigate;
- `critical`: pause automation and recalibrate or rollback.

## Methodology

1. Generate a stable historical synthetic population.
2. Train a logistic model on the earliest time segment.
3. Fit isotonic calibration on the next segment.
4. Evaluate ranking and probability quality on the holdout.
5. Generate a current population with an optional drift scenario.
6. Measure PSI, prediction shift, missingness, and current Brier score.
7. Convert diagnostics into a lifecycle recommendation.

## Limitations

The dataset is synthetic and the monitoring thresholds are illustrative. A production system would add model registry integration, rolling windows, alert deduplication, shadow model comparison, recalibration validation, rollback automation, and governance review.

Reproduce with:

```powershell
python scripts/run_monitoring_report.py --scenario all
```
"""
    (output_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    return report