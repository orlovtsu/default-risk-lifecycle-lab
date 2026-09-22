import pandas as pd

from .drift import monitoring_metrics
from .model import CalibratedModel


def rolling_monitoring(frame: pd.DataFrame, target: pd.Series, model: CalibratedModel, window: int = 500) -> pd.DataFrame:
    ordered = frame.sort_values("observed_at").reset_index(drop=True)
    predictions = model.predict(ordered)
    reference = ordered.iloc[:window]
    reference_pred = predictions[:window]
    rows = []
    for start in range(window, len(ordered), window):
        current = ordered.iloc[start : start + window]
        if len(current) < max(50, window // 4):
            continue
        metrics = monitoring_metrics(reference, current, reference_pred, predictions[start : start + len(current)])
        rows.append({
            "window_start": str(current["observed_at"].min().date()),
            "rows": len(current),
            "max_feature_psi": metrics["max_feature_psi"],
            "prediction_shift": metrics["prediction_mean_shift"],
            "event_rate": float(target.iloc[current.index].mean()),
        })
    return pd.DataFrame(rows)
