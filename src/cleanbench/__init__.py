"""CleanBench scorer v0.1 — LeRobot dataset quality reports."""

__version__ = "0.1.0"

from .scorer import score_dataset

__all__ = ["score_dataset", "__version__"]