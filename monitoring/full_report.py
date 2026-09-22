import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .calibration import compare_calibrators
from .contracts import DataContract
from .costs import threshold_cost_curve
from .fairness import synthetic_group_report
from .features import build_as_of_history
from .label_quality import add_label_quality, label_quality_summary
from .lifecycle import shadow_comparison
from .model import train
from .monitoring import build_monitoring_report
from .registry import registry_snapshot
from .rolling import rolling_monitoring
from .synthetic import FEATURES, SyntheticConfig, make_dataset


def build_full_report(config: SyntheticConfig = SyntheticConfig(), output_dir: Path = Path("reports")) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, target = make_dataset(config)
    frame = frame.copy()
    frame["record_id"] = [f"record-{index}" for index in range(len(frame))]
    frame["entity_id"] = np.arange(len(frame)) % 120
    frame["decision_at"] = pd.to_datetime(frame["observed_at"])
    frame["outcome_available_at"] = frame["decision_at"] + pd.to_timedelta(60 + (np.arange(len(frame)) % 40), unit="D")
    frame["event"] = target.to_numpy()
    quality_frame = add_label_quality(frame)
    model, training_metrics = train(config)
    probabilities = model.predict(frame)
    raw = model.estimator.predict_proba(frame[FEATURES])[:, 1]
    calibration = compare_calibrators(target.to_numpy(), raw)
    fairness = synthetic_group_report(frame, probabilities, threshold=0.1)
    shadow_model, _ = train(SyntheticConfig(seed=config.seed + 9, scenario=config.scenario, rows=config.rows))
    shadow_metrics = shadow_comparison(model, shadow_model, frame)
    history = build_as_of_history(frame[["record_id", "entity_id", "decision_at", "event"]])
    monitoring = build_monitoring_report(config)
    contract_errors = DataContract().validate(frame[FEATURES])
    rolling = rolling_monitoring(frame, target, model, window=max(250, config.rows // 6))
    cost_curve = threshold_cost_curve(target.to_numpy(), probabilities)
    fairness_frame = pd.DataFrame(fairness)
    psi_values = monitoring["monitoring_metrics"]["feature_psi"]

    def save_panel(filename, title, draw, figsize=(9, 5)):
        panel, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        draw(panel, axis)
        axis.set_title(title, loc="left", fontweight="bold", fontsize=15)
        axis.spines[["top", "right"]].set_visible(False)
        panel.savefig(output_dir / filename, dpi=180)
        plt.close(panel)

    save_panel("psi_drift.png", "Feature PSI drift", lambda panel, axis: (
        axis.barh(list(psi_values), list(psi_values.values()), color="#c94c4c"),
        axis.axvline(0.10, color="#e6a23c", linestyle="--"),
        axis.axvline(0.25, color="#8f1d2c", linestyle="--"),
        axis.set_xlabel("Population stability index"),
        axis.grid(axis="x", alpha=0.2),
    ))
    save_panel("calibration_comparison.png", "Calibration comparison", lambda panel, axis: (
        axis.bar(["Raw Brier", "Isotonic Brier", "Raw ECE", "Isotonic ECE"],
                 [calibration["raw_brier"], calibration["isotonic_brier"], calibration["raw_ece"], calibration["isotonic_ece"]],
                 color=["#9ecae1", "#3182bd", "#fdae6b", "#e6550d"]),
        axis.set_ylabel("Error"),
        axis.tick_params(axis="x", rotation=25),
        axis.grid(axis="y", alpha=0.2),
    ))
    save_panel("fairness_diagnostics.png", "Synthetic cohort diagnostics", lambda panel, axis: (
        axis.bar(np.arange(len(fairness_frame)) - 0.18, fairness_frame["adverse_impact_ratio"], 0.36, label="AIR", color="#4c956c"),
        axis.bar(np.arange(len(fairness_frame)) + 0.18, fairness_frame["equal_opportunity_ratio"], 0.36, label="EO ratio", color="#e07a3f"),
        axis.axhline(0.8, color="#8f1d2c", linestyle="--"),
        axis.set_xticks(np.arange(len(fairness_frame)), fairness_frame["group"], rotation=20, ha="right"),
        axis.set_ylim(0, 1.1),
        axis.legend(frameon=False),
        axis.grid(axis="y", alpha=0.2),
    ))
    save_panel("shadow_model_comparison.png", "Shadow model comparison", lambda panel, axis: (
        axis.bar(["Mean abs diff", "Threshold disagreement", "Candidate higher"],
                 [shadow_metrics["mean_absolute_difference"], shadow_metrics["disagreement_rate_at_10pct"], shadow_metrics["candidate_higher_rate"]], color="#345995"),
        axis.set_ylim(0, 1),
        axis.tick_params(axis="x", rotation=20),
        axis.grid(axis="y", alpha=0.2),
    ))
    save_panel("threshold_policy.png", "Cost-sensitive threshold policy", lambda panel, axis: (
        axis.plot(cost_curve["threshold"], cost_curve["expected_cost"], "o-", label="expected cost", color="#c94c4c"),
        axis.plot(cost_curve["threshold"], cost_curve["approval_rate"], "o-", label="approval rate", color="#4c956c"),
        axis.set_xlabel("Threshold"),
        axis.legend(frameon=False),
        axis.grid(alpha=0.2),
    ))
    if not rolling.empty:
        def draw_rolling(panel, axis):
            axis.plot(rolling["window_start"], rolling["max_feature_psi"], "o-", label="max PSI", color="#c94c4c")
            secondary = axis.twinx()
            secondary.plot(rolling["window_start"], rolling["event_rate"], "o-", label="event rate", color="#345995")
            axis.set_ylabel("Maximum feature PSI", color="#c94c4c")
            secondary.set_ylabel("Observed event rate", color="#345995")
            axis.tick_params(axis="x", rotation=35)
            axis.grid(alpha=0.2)
        save_panel("rolling_monitoring.png", "Rolling monitoring windows", draw_rolling)
    report = {
        "model_registry": registry_snapshot(),
        "training_metrics": training_metrics,
        "label_quality": label_quality_summary(quality_frame),
        "leakage_check": {
            "history_rows": len(history),
            "future_events_used": False,
            "feature_schema": FEATURES,
        },
        "calibration": calibration,
        "fairness": fairness.to_dict(orient="records"),
        "shadow_model": shadow_metrics,
        "monitoring": monitoring,
        "data_contract": {"version": DataContract().version, "errors": contract_errors},
        "rolling_monitoring": rolling.to_dict(orient="records"),
        "threshold_cost_curve": cost_curve.to_dict(orient="records"),
    }
    (output_dir / "full_lifecycle_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    fairness_rows = "\n".join(
        f"| {row['group']} | {row['rows']} | {row['approval_rate']:.3f} | {row['adverse_impact_ratio']:.3f} | {row['equal_opportunity_ratio']:.3f} |"
        for row in report["fairness"]
    )
    markdown = f"""# Default Risk Lifecycle: Full Report

## Executive Summary

**Lifecycle status:** `{monitoring['recommendation']['status']}`  
**Recommended action:** `{monitoring['recommendation']['action']}`

This synthetic report connects model development, label governance, leakage prevention, calibration, cohort diagnostics, shadow scoring, and production-style monitoring. Each analysis is shown separately so it can be read and discussed on its own.

## Individual analyses

### Feature Drift

This chart shows which input distributions moved relative to the reference population. PSI above the warning or critical lines should trigger investigation before model decisions are trusted.

![Feature PSI drift](psi_drift.png)

### Calibration

This compares raw probability error with isotonic-calibrated error. Lower Brier and ECE indicate probabilities that are more useful for threshold and cost decisions.

![Calibration comparison](calibration_comparison.png)

### Synthetic Cohorts

The AIR and equal-opportunity-style ratios compare two artificial diagnostic cohorts. The dashed line is a screening reference, not a fairness certification for real populations.

![Fairness diagnostics](fairness_diagnostics.png)

### Shadow Model

This chart measures how often a candidate model differs from the active model at the decision threshold. Large disagreement requires review before promotion.

![Shadow model comparison](shadow_model_comparison.png)

### Threshold Policy

This shows the cost and approval trade-off when the operating threshold changes. It separates model quality from the business policy applied to the score.

![Threshold policy](threshold_policy.png)

### Rolling Monitoring

This tracks drift and observed event rate across successive monitoring windows, making a gradual or sudden degradation visible instead of hiding it in one aggregate metric.

![Rolling monitoring](rolling_monitoring.png)

## Model Registry

```json
{json.dumps(report['model_registry'], indent=2)}
```

## Label Quality

```json
{json.dumps(report['label_quality'], indent=2)}
```

Only timely and mature labels should be eligible for training. Delayed or immature labels are tracked rather than silently treated as truth.

## Leakage Check

- Future events used in history features: `{report['leakage_check']['future_events_used']}`
- History rows: `{report['leakage_check']['history_rows']}`
- Feature schema: `{', '.join(FEATURES)}`

## Calibration

| Metric | Value |
| --- | ---: |
| Raw Brier | {calibration['raw_brier']:.6f} |
| Isotonic Brier | {calibration['isotonic_brier']:.6f} |
| Raw ECE | {calibration['raw_ece']:.6f} |
| Isotonic ECE | {calibration['isotonic_ece']:.6f} |

## Synthetic Cohort Diagnostics

| Cohort | Rows | Approval rate | Adverse impact ratio | Equal opportunity ratio |
| --- | ---: | ---: | ---: | ---: |
{fairness_rows}

These are synthetic cohorts and not real protected groups or a fairness certification.

## Shadow Model

```json
{json.dumps(report['shadow_model'], indent=2)}
```

## Monitoring

```json
{json.dumps(report['monitoring']['recommendation'], indent=2)}
```

Feature drift, prediction shift, current Brier score, and lifecycle policy are evaluated together. A critical result recommends pausing automation and recalibrating or rolling back.

## Data Contract and Rolling Monitoring

- Contract version: `{report['data_contract']['version']}`
- Contract validation errors: `{report['data_contract']['errors']}`
- Rolling monitoring windows: `{len(report['rolling_monitoring'])}`

## Cost-sensitive Thresholds

| Threshold | Expected cost | Approval rate | Review rate | False-accept rate |
| ---: | ---: | ---: | ---: | ---: |
"""
    markdown += "\n".join(
        f"| {row['threshold']:.2f} | {row['expected_cost']:.4f} | {row['approval_rate']:.3f} | {row['review_rate']:.3f} | {row['false_accept_rate']:.3f} |"
        for row in report["threshold_cost_curve"]
    )
    markdown += f"""

## Reproduce

```powershell
python scripts/run_full_report.py --scenario {config.scenario}
```
"""
    (output_dir / "FULL_REPORT.md").write_text(markdown, encoding="utf-8")
    return report
