#!/usr/bin/env python3
"""Deprecated strict-holdout wrapper.

The old file called a removed historical training entry. It is intentionally
disabled so reviewer-facing numbers cannot be regenerated from the historical
branch by accident.
"""

from __future__ import annotations

from pathlib import Path


CURRENT_TRAINER = Path(__file__).resolve().parents[2] / "1_training" / "bulk" / "train_scp682_general_graph_residual.py"


def main() -> int:
    raise SystemExit(
        "This historical strict-holdout runner has been disabled. "
        f"Use the current SCP682 main trainer instead: {CURRENT_TRAINER}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
