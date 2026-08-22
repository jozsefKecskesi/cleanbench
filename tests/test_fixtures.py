from __future__ import annotations

import polars as pl

from cleanbench.scorer import score_sample
from cleanbench.types import DatasetSample


def _frames(*, velocity: bool, spike: bool) -> pl.DataFrame:
    rows = []
    for ep in range(3):
        for frame in range(16):
            state = [0.1 * frame, 0.0]
            vel = [0.1, 0.0]
            if spike and ep == 1 and frame == 8:
                vel = [20.0, 0.0]
                state = [50.0, -50.0]
            row = {
                "episode_index": ep,
                "frame_index": frame,
                "timestamp": frame / 10,
                "task_index": 0,
                "observation.state": state,
                "action": [0.01, 0.0],
            }
            if velocity:
                row["observation.velocity"] = vel
            rows.append(row)
    return pl.DataFrame(rows)


def _sample(data: pl.DataFrame) -> DatasetSample:
    return DatasetSample(
        "local/fixture",
        None,
        {
            "codebase_version": "v3.0",
            "fps": 10,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [2]},
                "action": {"dtype": "float32", "shape": [2]},
            },
            "data_path": "data/file.parquet",
            "total_episodes": 3,
            "total_tasks": 1,
        },
        [{"task_index": 0, "task": "Pick the red cube and place it in the bin"}],
        [],
        data,
        [],
        [],
        "v3.0",
    )


def _check(report: dict, check_id: str) -> dict:
    return next(item for item in report["checks"] if item["id"] == check_id)


def test_explicit_velocity_is_not_derived() -> None:
    report = score_sample(_sample(_frames(velocity=True, spike=True)))
    vel = _check(report, "velocity_spike")
    assert vel["metrics"]["derived_velocity"] is False
    assert "finite-differenced" not in " ".join(report["warnings"])
    assert 1 in vel["flagged_episodes"]


def test_state_only_velocity_is_derived() -> None:
    report = score_sample(_sample(_frames(velocity=False, spike=True)))
    vel = _check(report, "velocity_spike")
    assert vel["metrics"]["derived_velocity"] is True
    assert any("finite-differenced" in item for item in report["warnings"])


def test_missing_torque_and_force_are_skipped() -> None:
    report = score_sample(_sample(_frames(velocity=True, spike=False)))
    torque = _check(report, "torque_saturation")
    force = _check(report, "force_spike")
    assert torque["status"] == "skipped"
    assert force["status"] == "skipped"
    assert torque["score"] is None
    assert force["score"] is None
    assert report["scores"]["physical"] is not None