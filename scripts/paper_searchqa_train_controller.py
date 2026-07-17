#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textskill_optimizer.paper.searchqa_controller_runtime import run_controller


if __name__ == "__main__":
    raise SystemExit(run_controller("train"))
