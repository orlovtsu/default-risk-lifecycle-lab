from dataclasses import dataclass

import pandas as pd

from .synthetic import FEATURES


@dataclass(frozen=True)
class DataContract:
    version: str = "features-v1"
    required_features: tuple[str, ...] = tuple(FEATURES)
    forbidden_columns: tuple[str, ...] = ("event", "latent_probability", "outcome_available_at")

    def validate(self, frame: pd.DataFrame) -> list[str]:
        errors = [f"missing:{name}" for name in self.required_features if name not in frame.columns]
        errors.extend(f"forbidden:{name}" for name in self.forbidden_columns if name in frame.columns)
        return errors
