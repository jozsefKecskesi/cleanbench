from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import __version__
from .checks import ALL_CHECKS
from .loader import load_dataset_sample
from .types import CheckResult, DatasetSample

WEIGHTS = {"physical": 0.45, "annotation": 0.45, "integrity": 0.10}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _band(overall: float | None) -> str:
    if overall is None:
        return "unknown"
    if overall >= 80:
        return "clean"
    if overall >= 60:
        return "review"
    return "flagged"


def _error_report(dataset_id: str, revision: str | None, message: str) -> dict[str, Any]:
    return {
        "schema_version": __version__,
        "dataset_id": dataset_id,
        "revision": revision,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "format": {"detected": "unknown", "valid_lerobot": False, "robot_type": None},
        "sample": {"n_episodes": 0, "n_frames": 0, "max_episodes": 0},
        "scores": {"overall": None, "physical": None, "annotation": None, "integrity": None},
        "band": "unknown",
        "checks": [],
        "flagged_episodes": [],
        "warnings": [],
        "error": message,
    }


def assemble_report(sample: DatasetSample, results: list[CheckResult], n_episodes: int) -> dict[str, Any]:
    by_cat: dict[str, list[float]] = {"physical": [], "annotation": [], "integrity": []}
    flagged: set[int] = set()
    for r in results:
        if r.score is not None and r.status != "skipped":
            by_cat[r.category].append(r.score)
        flagged.update(r.flagged_episodes)
    cat_scores = {k: (round(_mean(v), 2) if v else None) for k, v in by_cat.items()}
    weighted = []
    weight_sum = 0.0
    for cat, wgt in WEIGHTS.items():
        if cat_scores[cat] is not None:
            weighted.append(wgt * cat_scores[cat])
            weight_sum += wgt
    overall = round(sum(weighted) / weight_sum, 2) if weight_sum else None
    n_ep = 0
    n_fr = int(sample.frames.height) if sample.frames is not None else 0
    if sample.frames is not None and not sample.frames.is_empty() and "episode_index" in sample.frames.columns:
        n_ep = int(sample.frames.get_column("episode_index").n_unique())
    valid = bool(sample.info.get("features")) or n_fr > 0
    return {
        "schema_version": __version__,
        "dataset_id": sample.dataset_id,
        "revision": sample.revision,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "format": {
            "detected": sample.format_version,
            "valid_lerobot": valid,
            "robot_type": sample.info.get("robot_type"),
            "fps": sample.info.get("fps"),
            "total_episodes": sample.info.get("total_episodes"),
            "total_frames": sample.info.get("total_frames"),
            "total_tasks": sample.info.get("total_tasks"),
        },
        "sample": {"n_episodes": n_ep, "n_frames": n_fr, "max_episodes": n_episodes},
        "scores": {"overall": overall, **cat_scores},
        "band": _band(overall),
        "checks": [r.to_dict() for r in results],
        "flagged_episodes": sorted(flagged),
        "warnings": sample.warnings,
        "error": None,
    }


def score_sample(sample: DatasetSample, n_episodes: int = 50) -> dict[str, Any]:
    results: list[CheckResult] = []
    for fn in ALL_CHECKS:
        try:
            results.append(fn(sample))
        except Exception as exc:
            results.append(CheckResult(fn.__name__, "integrity", "error", None, f"{type(exc).__name__}: {exc}"))
    return assemble_report(sample, results, n_episodes)


def score_dataset(dataset_id: str, n_episodes: int = 50, max_frames: int = 100_000, revision: str | None = None, local_dir: str | None = None) -> dict[str, Any]:
    try:
        sample = load_dataset_sample(dataset_id, n_episodes=n_episodes, max_frames=max_frames, revision=revision, local_dir=local_dir)
    except Exception as exc:
        return _error_report(dataset_id, revision, f"{type(exc).__name__}: {exc}")
    return score_sample(sample, n_episodes=n_episodes)
