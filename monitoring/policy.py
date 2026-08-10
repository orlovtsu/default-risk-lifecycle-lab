from dataclasses import dataclass


@dataclass(frozen=True)
class MonitoringPolicy:
    version: str = "monitoring-policy-1.0"
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    prediction_shift_warning: float = 0.03
    prediction_shift_critical: float = 0.08
    brier_warning: float = 0.10
    brier_critical: float = 0.16


def recommend(metrics: dict, current_brier: float, policy: MonitoringPolicy = MonitoringPolicy()) -> dict[str, str]:
    critical = (
        metrics["max_feature_psi"] >= policy.psi_critical
        or metrics["prediction_mean_shift"] >= policy.prediction_shift_critical
        or current_brier >= policy.brier_critical
    )
    warning = (
        metrics["max_feature_psi"] >= policy.psi_warning
        or metrics["prediction_mean_shift"] >= policy.prediction_shift_warning
        or current_brier >= policy.brier_warning
    )
    if critical:
        return {"status": "critical", "action": "pause_automation_and_recalibrate_or_rollback"}
    if warning:
        return {"status": "warning", "action": "start_shadow_monitoring_and_investigate"}
    return {"status": "healthy", "action": "continue_monitoring"}
