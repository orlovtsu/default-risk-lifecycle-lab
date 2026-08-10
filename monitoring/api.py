from fastapi import FastAPI

from .monitoring import build_monitoring_report
from .synthetic import SyntheticConfig

app = FastAPI(title="Default Risk Lifecycle Lab", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/monitoring/report")
def report() -> dict:
    return build_monitoring_report(SyntheticConfig(scenario="all"))
