import pandas as pd

HISTORY_FEATURES = ["prior_event_count", "prior_bad_count", "days_since_prior", "prior_bad_rate"]


def build_as_of_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Build history features using only events strictly before decision_at."""
    ordered = frame.sort_values(["entity_id", "decision_at"]).copy()
    rows = []
    for _, group in ordered.groupby("entity_id", sort=False):
        prior_dates: list[pd.Timestamp] = []
        prior_bad: list[int] = []
        for _, current in group.iterrows():
            decision_at = current["decision_at"]
            eligible = [index for index, value in enumerate(prior_dates) if value < decision_at]
            count = len(eligible)
            bad_count = sum(prior_bad[index] for index in eligible)
            days_since = (
                (decision_at - max(prior_dates[index] for index in eligible)).days
                if eligible else None
            )
            rows.append({
                "record_id": current["record_id"],
                "prior_event_count": count,
                "prior_bad_count": bad_count,
                "days_since_prior": days_since,
                "prior_bad_rate": bad_count / count if count else 0.0,
            })
            prior_dates.append(decision_at)
            prior_bad.append(int(current["event"]))
    return pd.DataFrame(rows)
