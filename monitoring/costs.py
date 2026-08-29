import numpy as np
import pandas as pd


def threshold_cost_curve(
    y_true,
    probability,
    thresholds=(0.05, 0.10, 0.15, 0.20),
    false_accept_cost=5.0,
    false_reject_cost=1.0,
    review_cost=0.25,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    rows = []
    for threshold in thresholds:
        approved = probability <= threshold
        review = (probability > threshold) & (probability <= threshold + 0.03)
        false_accept = approved & (y_true == 1)
        false_reject = (~approved & ~review) & (y_true == 0)
        total_cost = (
            false_accept.sum() * false_accept_cost
            + false_reject.sum() * false_reject_cost
            + review.sum() * review_cost
        )
        rows.append({
            "threshold": threshold,
            "expected_cost": float(total_cost / len(y_true)),
            "approval_rate": float(approved.mean()),
            "review_rate": float(review.mean()),
            "false_accept_rate": float(false_accept.mean()),
        })
    return pd.DataFrame(rows)
