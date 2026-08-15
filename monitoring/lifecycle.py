import numpy as np

from .model import CalibratedModel


def shadow_comparison(active: CalibratedModel, candidate: CalibratedModel, frame) -> dict[str, float]:
    active_probability = active.predict(frame)
    candidate_probability = candidate.predict(frame)
    return {
        "mean_absolute_difference": float(np.mean(np.abs(active_probability - candidate_probability))),
        "disagreement_rate_at_10pct": float(
            np.mean((active_probability <= 0.10) != (candidate_probability <= 0.10))
        ),
        "candidate_higher_rate": float(np.mean(candidate_probability > active_probability)),
    }
