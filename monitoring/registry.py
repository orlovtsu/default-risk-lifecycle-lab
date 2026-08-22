from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelManifest:
    model_version: str
    status: str
    calibration: str
    training_window: str
    feature_schema_version: str
    rollback_target: str | None = None


ACTIVE_MODEL = ModelManifest(
    model_version="model-v1",
    status="active",
    calibration="isotonic",
    training_window="synthetic-chronological-60-20-20",
    feature_schema_version="features-v1",
    rollback_target=None,
)


def registry_snapshot(candidate: ModelManifest | None = None) -> dict:
    return {
        "active": asdict(ACTIVE_MODEL),
        "candidate": asdict(candidate) if candidate else None,
        "promotion_rule": "promote only after holdout, calibration, drift, and cohort checks",
        "rollback_rule": "rollback when monitoring status is critical or candidate degrades cohort metrics",
    }
