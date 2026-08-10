import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.reporting import build_report
from monitoring.synthetic import SyntheticConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["stable", "feature-drift", "prevalence-drift", "concept-drift", "all"])
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring_report.json"))
    args = parser.parse_args()
    report = build_report(SyntheticConfig(scenario=args.scenario))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
