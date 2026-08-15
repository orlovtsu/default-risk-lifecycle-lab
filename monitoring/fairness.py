import pandas as pd


def synthetic_group_report(frame: pd.DataFrame, probability, threshold: float = 0.1) -> pd.DataFrame:
    scored = frame.copy()
    scored["probability"] = probability
    scored["approved"] = scored["probability"] <= threshold
    rows = []
    for group, data in scored.groupby("synthetic_cohort", sort=True):
        rows.append({
            "group": group,
            "rows": len(data),
            "approval_rate": float(data["approved"].mean()),
            "mean_probability": float(data["probability"].mean()),
            "observed_event_rate": float(data["event"].mean()),
            "approval_if_good": float(data.loc[data["event"] == 0, "approved"].mean()),
        })
    report = pd.DataFrame(rows)
    if not report.empty:
        reference = report["approval_rate"].max()
        good_reference = report["approval_if_good"].max()
        report["adverse_impact_ratio"] = report["approval_rate"] / reference if reference else 0.0
        report["equal_opportunity_ratio"] = report["approval_if_good"] / good_reference if good_reference else 0.0
    return report
