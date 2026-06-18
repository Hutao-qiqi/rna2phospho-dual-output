"""
SCP682-PPKO V10B rerun dispatcher.

This active entry no longer reads release-v1 V10 outputs. Use the frozen
SCP682_PPKO_V10B_transferable package and the current Fig4 source-table scripts.
The old release-v1 helper is archived at:
organized_code/legacy/ppko/rerun_missing_items_release_v10.py
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("SCP682_DATA_ROOT", r"D:\data\lsy\vm_lsy_parent\lsy" if os.name == "nt" else "/mnt/d/data/lsy/vm_lsy_parent/lsy"))
PPKO_PACKAGE = ROOT / "SCP682_PPKO_V10B_transferable"


def main() -> None:
    scripts = [
        PPKO_PACKAGE / "scripts" / "validate_v10b_p100_all_drugs.py",
        ROOT / "03_code" / "model_validation" / "evaluate_ppko_p100_published_baselines.py",
        ROOT / "03_code" / "figure_generation" / "export_v10b_p100_sitelevel_all125.py",
    ]
    print("SCP682-PPKO V10B active rerun entry")
    print(f"Frozen package: {PPKO_PACKAGE}")
    print("Use these current scripts:")
    for script in scripts:
        print(f"- {script}")
    print("\nExample:")
    print(f"cd {PPKO_PACKAGE}")
    print(r"python .\scripts\validate_v10b_p100_all_drugs.py --device cuda:0")


if __name__ == "__main__":
    main()
