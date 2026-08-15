from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LabelQualityPolicy:
    outcome_window_days: int = 60
    max_snapshot_delay_days: int = 75


def add_label_quality(frame: pd.DataFrame, policy: LabelQualityPolicy = LabelQualityPolicy()) -> pd.DataFrame:
    result = frame.copy()
    result["maturity_days"] = (result["outcome_available_at"] - result["decision_at"]).dt.days
    result["label_available"] = result["maturity_days"] >= policy.outcome_window_days
    result["label_timely"] = result["maturity_days"] <= policy.max_snapshot_delay_days
    result["training_eligible"] = result["label_available"] & result["label_timely"]
    result["label_quality"] = "immature"
    result.loc[result["label_available"] & ~result["label_timely"], "label_quality"] = "delayed"
    result.loc[result["training_eligible"], "label_quality"] = "timely"
    return result


def label_quality_summary(frame: pd.DataFrame) -> dict[str, float]:
    quality = frame["label_quality"].value_counts(normalize=True)
    return {str(key): float(value) for key, value in quality.items()}
