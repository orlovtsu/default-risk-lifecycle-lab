import numpy as np
import pandas as pd

from monitoring.calibration import compare_calibrators, ece
from monitoring.features import build_as_of_history
from monitoring.fairness import synthetic_group_report
from monitoring.label_quality import add_label_quality, label_quality_summary
from monitoring.lifecycle import shadow_comparison
from monitoring.contracts import DataContract
from monitoring.costs import threshold_cost_curve
from monitoring.rolling import rolling_monitoring
from monitoring.model import train
from monitoring.synthetic import SyntheticConfig, make_dataset


def test_label_quality_distinguishes_maturity_and_delay():
    frame = pd.DataFrame({
        "decision_at": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
        "outcome_available_at": pd.to_datetime(["2024-01-30", "2024-03-01", "2024-06-01"]),
    })
    result = add_label_quality(frame)
    assert list(result["label_quality"]) == ["immature", "timely", "delayed"]
    assert label_quality_summary(result)["timely"] == 1 / 3


def test_as_of_history_has_no_current_or_future_events():
    frame = pd.DataFrame({
        "record_id": ["a", "b", "c"],
        "entity_id": [1, 1, 1],
        "decision_at": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "event": [1, 0, 0],
    })
    history = build_as_of_history(frame)
    assert history.iloc[0].prior_event_count == 0
    assert history.iloc[1].prior_event_count == 1
    assert history.iloc[1].prior_bad_count == 1


def test_calibration_metrics_are_bounded():
    result = compare_calibrators(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.8]))
    assert 0 <= result["raw_ece"] <= 1
    assert 0 <= result["isotonic_ece"] <= 1


def test_fairness_report_has_ratios():
    frame, target = make_dataset(SyntheticConfig(rows=100))
    report = synthetic_group_report(frame.assign(event=target), np.full(len(frame), 0.1))
    assert not report.empty
    assert {"adverse_impact_ratio", "equal_opportunity_ratio"} <= set(report.columns)


def test_shadow_comparison_is_bounded():
    active, _ = train(SyntheticConfig(rows=500))
    candidate, _ = train(SyntheticConfig(seed=9, rows=500))
    frame, _ = make_dataset(SyntheticConfig(rows=500))
    report = shadow_comparison(active, candidate, frame)
    assert report["mean_absolute_difference"] >= 0
    assert 0 <= report["disagreement_rate_at_10pct"] <= 1


def test_data_contract_rejects_target_leakage():
    frame, _ = make_dataset(SyntheticConfig(rows=100))
    errors = DataContract().validate(frame.assign(event=0))
    assert "forbidden:event" in errors


def test_threshold_cost_curve_is_bounded():
    curve = threshold_cost_curve(np.array([0, 1, 0, 1]), np.array([0.02, 0.12, 0.08, 0.2]))
    assert len(curve) == 4
    assert curve["expected_cost"].ge(0).all()
    assert curve["approval_rate"].between(0, 1).all()


def test_rolling_monitoring_returns_windows():
    model, _ = train(SyntheticConfig(rows=1000))
    frame, target = make_dataset(SyntheticConfig(rows=1000, scenario="all"))
    result = rolling_monitoring(frame, target, model, window=250)
    assert not result.empty
    assert result["max_feature_psi"].ge(0).all()
