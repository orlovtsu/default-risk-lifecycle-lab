# Model Card

## Intended use

This repository demonstrates lifecycle monitoring for a synthetic default-risk-like binary model. It is not a production model and must not be used for decisions about real people.

## Data and validation

Data is generated locally from seeded distributions. Training, calibration, and holdout are chronological. Current populations can simulate feature drift, prevalence drift, and concept drift. Additional checks cover label maturity, leakage-safe as-of history, calibration comparison, synthetic cohort diagnostics, and shadow-model disagreement.

## Monitoring

The report evaluates feature PSI, prediction mean shift, missingness, current Brier score, and lifecycle policy thresholds. The monitoring policy maps diagnostics to `healthy`, `warning`, or `critical` with an operational recommendation.

## Limitations

Synthetic drift is not evidence of real-world drift behavior. PSI thresholds and lifecycle actions are illustrative. A production system requires representative data governance, rolling windows, alert ownership, shadow models, recalibration validation, rollback controls, and documented approval.
