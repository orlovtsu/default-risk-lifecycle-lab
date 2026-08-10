from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .synthetic import FEATURES, SyntheticConfig, make_dataset


@dataclass
class CalibratedModel:
    estimator: LogisticRegression
    calibrator: IsotonicRegression
    version: str = "model-v1"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(frame[FEATURES])[:, 1]
        return np.clip(self.calibrator.predict(raw), 0, 1)


def train(config: SyntheticConfig = SyntheticConfig()) -> tuple[CalibratedModel, dict[str, float]]:
    frame, target = make_dataset(config)
    dates = pd.to_datetime(frame["observed_at"])
    train_mask = dates <= dates.quantile(0.6)
    calibration_mask = (dates > dates.quantile(0.6)) & (dates <= dates.quantile(0.8))
    holdout_mask = dates > dates.quantile(0.8)
    estimator = LogisticRegression(max_iter=1000, random_state=config.seed)
    estimator.fit(frame.loc[train_mask, FEATURES], target.loc[train_mask])
    raw_calibration = estimator.predict_proba(frame.loc[calibration_mask, FEATURES])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calibration, target.loc[calibration_mask])
    model = CalibratedModel(estimator, calibrator)
    raw_holdout = estimator.predict_proba(frame.loc[holdout_mask, FEATURES])[:, 1]
    probabilities = model.predict(frame.loc[holdout_mask])
    return model, {
        "roc_auc": float(roc_auc_score(target.loc[holdout_mask], raw_holdout)),
        "brier_score": float(brier_score_loss(target.loc[holdout_mask], probabilities)),
        "holdout_rows": float(holdout_mask.sum()),
    }
