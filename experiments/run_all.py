"""Run dataset generation, training, comparison, plotting, and manifest output."""

import argparse
import json

from _bootstrap import ROOT
from ev_thermal.pipeline import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small end-to-end verification run")
    args = parser.parse_args()
    print(json.dumps(run_all(ROOT, quick=args.quick), indent=2))


if __name__ == "__main__":
    main()

