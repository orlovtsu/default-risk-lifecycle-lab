from fastapi.testclient import TestClient

from monitoring.api import app
from monitoring.drift import psi
from monitoring.monitoring import build_monitoring_report
from monitoring.reporting import build_report
from monitoring.full_report import build_full_report
from monitoring.policy import MonitoringPolicy, recommend
from monitoring.synthetic import SyntheticConfig, make_dataset


client = TestClient(app)


def test_stable_scenario_is_healthy():
    report = build_monitoring_report(SyntheticConfig(seed=7, scenario="stable"))
    assert report["recommendation"]["status"] in {"healthy", "warning"}
    assert report["monitoring_metrics"]["max_feature_psi"] >= 0


def test_all_scenario_becomes_actionable():
    report = build_monitoring_report(SyntheticConfig(seed=7, scenario="all"))
    assert report["recommendation"]["status"] in {"warning", "critical"}
    assert report["recommendation"]["action"] != "continue_monitoring"


def test_psi_detects_shift():
    reference, _ = make_dataset(SyntheticConfig(seed=1, scenario="stable"))
    current, _ = make_dataset(SyntheticConfig(seed=1, scenario="feature-drift"))
    assert psi(reference["income_stability"], current["income_stability"]) > 0


def test_policy_recommendations_are_explicit():
    metrics = {"max_feature_psi": 0.4, "prediction_mean_shift": 0.0}
    result = recommend(metrics, 0.05, MonitoringPolicy())
    assert result == {"status": "critical", "action": "pause_automation_and_recalibrate_or_rollback"}


def test_monitoring_api():
    response = client.get("/monitoring/report")
    assert response.status_code == 200
    assert response.json()["model_version"] == "model-v1"
    assert response.json()["recommendation"]["status"] in {"healthy", "warning", "critical"}


def test_markdown_report_is_reproducible(tmp_path):
    report = build_report(SyntheticConfig(seed=3, scenario="all"), tmp_path)
    assert report["recommendation"]["status"] == "critical"
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "monitoring_dashboard.png").exists()


def test_full_lifecycle_report_connects_governance_layers(tmp_path):
    report = build_full_report(SyntheticConfig(seed=3, scenario="all", rows=500), tmp_path)
    assert report["leakage_check"]["future_events_used"] is False
    assert report["calibration"]["isotonic_brier"] >= 0
    assert report["fairness"]
    assert "model_registry" in report
    assert (tmp_path / "FULL_REPORT.md").exists()
