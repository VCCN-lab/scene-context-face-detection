"""Make the analysis package and the run_* scripts importable from the tests."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # no display needed

CODE = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE))
