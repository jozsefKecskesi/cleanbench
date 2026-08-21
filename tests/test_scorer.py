from __future__ import annotations

import json
import numpy as np
import polars as pl

from cleanbench.scorer import score_dataset, score_sample
from cleanbench.types import DatasetSample


def frames(n_episodes=4, length=12):
    rows=[]
    for ep in range(n_episodes):
        for frame in range(length):
            rows.append({"episode_index":ep,"frame_index":frame,"timestamp":frame/10,"task_index":0,"observation.state":[0.1*frame,0.2*frame],"action":[0.05*frame,0.01]})
    return pl.DataFrame(rows)


def sample(data, tasks=None):
    return DatasetSample("local/test",None,{"codebase_version":"v3.0","fps":10,"features":{"observation.state":{"dtype":"float32","shape":[2]},"action":{"dtype":"float32","shape":[2]}},"data_path":"data/file.parquet","total_episodes":4,"total_tasks":1},tasks or [{"task_index":0,"task":"Pick the red cube and place it in the bin"}],[],data,[],[],"v3.0")


def test_schema_and_checks():
    report=score_sample(sample(frames()))
    assert report["error"] is None
    assert report["schema_version"] == "0.1.0"
    assert {"schema_version","dataset_id","format","sample","scores","band","checks","flagged_episodes","warnings","error"} <= report.keys()
    assert len(report["checks"]) == 15
    json.dumps(report)


def test_placeholder_task():
    report=score_sample(sample(frames(), [{"task_index":0,"task":"task"}]))
    check=next(x for x in report["checks"] if x["id"] == "empty_or_placeholder_task")
    assert check["status"] == "fail"


def test_velocity_spike():
    rows=frames().to_dicts()
    for row in rows:
        if row["episode_index"] == 2 and row["frame_index"] == 6:
            row["observation.state"]=[50.0,-50.0]
    report=score_sample(sample(pl.DataFrame(rows)))
    check=next(x for x in report["checks"] if x["id"] == "velocity_spike")
    assert 2 in check["flagged_episodes"]


def test_nan_inf():
    rows=frames().to_dicts(); rows[0]["observation.state"]=[np.nan,0.0]
    report=score_sample(sample(pl.DataFrame(rows)))
    check=next(x for x in report["checks"] if x["id"] == "nan_inf")
    assert check["status"] == "fail"


def test_invalid_id_preserves_schema():
    report=score_dataset("not-a-real-org/not-a-real-dataset")
    assert report["error"] is not None
    assert "scores" in report and "checks" in report
