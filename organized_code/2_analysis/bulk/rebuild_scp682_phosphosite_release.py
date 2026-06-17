#!/usr/bin/env python3
"""Rebuild the frozen SCP682 CPTAC/PDC phosphosite release inside this package.

This wrapper regenerates the official sample-median-centered phosphosite matrices
from the fixed +0.2 raw fusion matrices already sealed in the SCP682 package.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
ENV = os.environ.copy()
ENV["SCP682_PROJECT_ROOT"] = str(ROOT)

steps = [
    ROOT / "03_code/model_validation/finalize_scp682_v4_0_official_phosphosite_release_20260503.py",
]

for step in steps:
    print(f"[SCP682] running {step}", flush=True)
    subprocess.run([PYTHON, str(step)], cwd=str(ROOT), env=ENV, check=True)
print("[SCP682] phosphosite release rebuilt", flush=True)
