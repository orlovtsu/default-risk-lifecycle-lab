# Default Risk Lifecycle Lab

A domain-neutral synthetic laboratory for default-risk-like model lifecycle management.

```text
synthetic history -> train -> calibrate -> holdout -> deploy simulation -> monitor -> recalibrate / rollback
```

The project contains no production model files, real feature names, customer data, business thresholds, or employer-specific logic.

## Demonstrated capabilities

- chronological train/calibration/holdout evaluation;
- probability calibration with Brier score;
- synthetic feature, prediction, missingness, and prevalence drift;
- monitoring policy with healthy/warning/critical states;
- shadow model comparison;
- lifecycle recommendation: keep, investigate, recalibrate, or rollback;
- reproducible Markdown/PNG/JSON report;
- FastAPI, Docker, CI, and tests.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python scripts/run_monitoring_report.py --scenario all
uvicorn monitoring.api:app --reload
```

Open `reports/REPORT.md` for the lifecycle report and `http://127.0.0.1:8000/docs` for the API.

The report includes a monitoring dashboard, PSI by feature, prediction shift, Brier score comparison, and lifecycle recommendation.

All metrics are synthetic demonstrations, not production evidence.

See [lifecycle design](docs/lifecycle.md) and [model card](MODEL_CARD.md) for validation boundaries, monitoring ownership, and limitations.
