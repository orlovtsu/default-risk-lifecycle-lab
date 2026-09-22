from itertools import pairwise

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def ece(y_true, probability, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in pairwise(edges):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - y_true[mask].mean())
    return float(error)


def compare_calibrators(y_true, raw_probability) -> dict:
    y_true = np.asarray(y_true)
    raw_probability = np.asarray(raw_probability)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw_probability, y_true)
    calibrated = isotonic.predict(raw_probability)
    return {
        "raw_brier": float(brier_score_loss(y_true, raw_probability)),
        "isotonic_brier": float(brier_score_loss(y_true, calibrated)),
        "raw_ece": ece(y_true, raw_probability),
        "isotonic_ece": ece(y_true, calibrated),
        "raw_calibration_points": len(calibration_curve(y_true, raw_probability, n_bins=10)[0]),
        "isotonic_calibration_points": len(calibration_curve(y_true, calibrated, n_bins=10)[0]),
    }
