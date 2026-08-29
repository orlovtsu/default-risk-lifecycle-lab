import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import compare_calibrators
from .contracts import DataContract
from .costs import threshold_cost_curve
from .features import build_as_of_history
from .fairness import synthetic_group_report
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
    history = build_as_of_history(frame[["record_id", "entity_id", "decision_at", "event"]])
    monitoring = build_monitoring_report(config)
    contract_errors = DataContract().validate(frame[FEATURES])
    rolling = rolling_monitoring(frame, target, model, window=max(250, config.rows // 6))
    cost_curve = threshold_cost_curve(target.to_numpy(), probabilities)
    report = {
        "model_registry": registry_snapshot(),
        "training_metrics": training_metrics,
        "label_quality": label_quality_summary(quality_frame),
        "leakage_check": {
            "history_rows": int(len(history)),
            "future_events_used": False,
            "feature_schema": FEATURES,
        },
        "calibration": calibration,
        "fairness": fairness.to_dict(orient="records"),
        "shadow_model": shadow_comparison(model, shadow_model, frame),
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

This synthetic report connects model development, label governance, leakage prevention, calibration, cohort diagnostics, shadow scoring, and production-style monitoring.

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
    markdown += """

## Reproduce

```powershell
python scripts/run_monitoring_report.py --scenario {config.scenario}
```
"""
    (output_dir / "FULL_REPORT.md").write_text(markdown, encoding="utf-8")
    return report
