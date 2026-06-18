#!/usr/bin/env python3
"""Compatibility wrapper for the current SCP682 bulk predictor.

Historically this filename pointed to a web-only runner. It now delegates to
predict_scp682.py so old scheduler commands execute the current SCP682 main
model.
"""

from __future__ import annotations

from predict_scp682 import main


if __name__ == "__main__":
    raise SystemExit(main())
