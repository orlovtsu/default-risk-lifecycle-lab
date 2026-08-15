import numpy as np
import pandas as pd


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts = np.histogram(reference, edges)[0] + 1e-6
    cur_counts = np.histogram(current, edges)[0] + 1e-6
    ref_share = ref_counts / ref_counts.sum()
    cur_share = cur_counts / cur_counts.sum()
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def monitoring_metrics(reference: pd.DataFrame, current: pd.DataFrame, ref_pred: np.ndarray, cur_pred: np.ndarray) -> dict:
    feature_psi = {
        column: psi(reference[column], current[column])
        for column in reference.columns
        if column not in {"observed_at", "latent_probability", "synthetic_cohort"}
        and pd.api.types.is_numeric_dtype(reference[column])
    }
    return {
        "feature_psi": feature_psi,
        "max_feature_psi": max(feature_psi.values(), default=0.0),
        "prediction_mean_reference": float(np.mean(ref_pred)),
        "prediction_mean_current": float(np.mean(cur_pred)),
        "prediction_mean_shift": float(abs(np.mean(cur_pred) - np.mean(ref_pred))),
        "missingness_current": float(current.isna().mean().mean()),
    }
