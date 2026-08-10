import numpy as np

from .drift import monitoring_metrics
from .model import train
from .policy import MonitoringPolicy, recommend
from .synthetic import SyntheticConfig, make_dataset


def build_monitoring_report(config: SyntheticConfig = SyntheticConfig()) -> dict:
    model, training_metrics = train(SyntheticConfig(seed=config.seed, scenario="stable"))
    reference, _ = make_dataset(SyntheticConfig(seed=config.seed, scenario="stable"))
    current, current_target = make_dataset(config)
    reference_pred = model.predict(reference)
    current_pred = model.predict(current)
    metrics = monitoring_metrics(reference, current, reference_pred, current_pred)
    current_brier = float(np.mean((current_target.to_numpy() - current_pred) ** 2))
    decision = recommend(metrics, current_brier, MonitoringPolicy())
    return {"model_version": model.version, "training_metrics": training_metrics, "monitoring_metrics": metrics, "current_brier": current_brier, "recommendation": decision}
