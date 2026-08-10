from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["income_stability", "activity_rate", "balance_volatility", "evidence_completeness"]


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    rows: int = 3000
    scenario: str = "stable"


def make_dataset(config: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(config.seed)
    dates = pd.date_range("2024-01-01", periods=config.rows, freq="D")
    phase = np.arange(config.rows) >= int(config.rows * 0.6)
    shifted = config.scenario in {"feature-drift", "concept-drift", "all"}
    prevalence_shift = config.scenario in {"prevalence-drift", "concept-drift", "all"}
    income_stability = rng.beta(8, 2, config.rows)
    activity_rate = rng.gamma(2.0, 0.8, config.rows)
    balance_volatility = rng.gamma(2.0, 0.22, config.rows)
    evidence_completeness = rng.beta(14, 2, config.rows)
    if shifted:
        income_stability[phase] = np.clip(income_stability[phase] - 0.16, 0, 1)
        activity_rate[phase] *= 1.25
        balance_volatility[phase] *= 1.45
    if prevalence_shift:
        evidence_completeness[phase] = np.clip(evidence_completeness[phase] - 0.12, 0, 1)
    logit = (-2.5 - 2.0 * income_stability + 0.28 * activity_rate
             + 0.85 * balance_volatility - 1.3 * evidence_completeness
             + (0.8 if prevalence_shift else 0.0) * phase
             + rng.normal(0, 0.3, config.rows))
    probability = 1 / (1 + np.exp(-logit))
    target = pd.Series(rng.binomial(1, probability), name="event")
    return pd.DataFrame({
        "observed_at": dates,
        "income_stability": income_stability,
        "activity_rate": activity_rate,
        "balance_volatility": balance_volatility,
        "evidence_completeness": evidence_completeness,
        "latent_probability": probability,
    }), target
