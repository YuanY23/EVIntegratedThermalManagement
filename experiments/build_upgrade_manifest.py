"""Bind all verified upgrade evidence to the current formal plant and run."""

import json

from _bootstrap import ROOT
from ev_thermal.artifacts import write_upgrade_manifest


if __name__ == "__main__":
    print(json.dumps(write_upgrade_manifest(ROOT), indent=2))
