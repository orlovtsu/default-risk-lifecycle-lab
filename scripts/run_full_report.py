import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.full_report import build_full_report
from monitoring.synthetic import SyntheticConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["stable", "feature-drift", "prevalence-drift", "concept-drift", "all"])
    args = parser.parse_args()
    build_full_report(SyntheticConfig(scenario=args.scenario))
    print("report=reports/FULL_REPORT.md")


if __name__ == "__main__":
    main()
