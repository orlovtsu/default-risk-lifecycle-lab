# Lifecycle Design

```text
train -> calibrate -> holdout validate -> deploy simulation -> monitor -> action
                                             |                  |
                                             |                  +--> healthy
                                             |                  +--> warning / shadow score
                                             |                  +--> critical / recalibrate or rollback
```

## Ownership boundaries

- Model training owns feature order, estimator version, and calibration artifact.
- Monitoring owns drift metrics and alert policy.
- Lifecycle policy owns the recommendation, not the model score.
- Recalibration is a new versioned artifact and must be validated on a fresh holdout.
- Rollback is a deployment action, not a silent model mutation.

## Why calibration is lifecycle work

A model can preserve ranking quality while its probabilities become unreliable after population or process changes. The lab therefore monitors both discrimination (`ROC AUC`) and probability quality (`Brier score`) and treats them as separate signals.
