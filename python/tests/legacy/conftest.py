"""Make the v1 scripts in the repository's ``py/`` directory importable."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

LEGACY_DIR = Path(__file__).resolve().parents[3] / "py"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))
