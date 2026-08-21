from __future__ import annotations

import re
import numpy as np
import polars as pl

from cleanbench.columns import ACTION_EXACT, FRAME_EXACT, STATE_EXACT, TIME_EXACT, first_present
from cleanbench.types import CheckResult, DatasetSample

PLACEHOLDERS = {"", "task", "todo", "n/a", "na", "none", "unknown", "test", "asdf", ".", "-", "null", "string", "description", "do the task", "untitled"}
PLACEHOLDER_RE = re.compile(r"^(task[\\s_-]*\\d+|instruction|label)$", re.I)


def _skip(check_id, reason):
    return CheckResult(check_id, "annotation", "skipped", None, reason)


def _score(check_id, score, summary, metrics, flagged=None):
    return CheckResult(check_id, "annotation", "pass" if score >= 80 else "fail", round(score, 2), summary, flagged or [], metrics)


def _episodes(frames):
    if frames.is_empty():
        return
    if "episode_index" not in frames.columns:
        yield 0, frames
        return
    for (ep,), data in frames.group_by("episode_index", maintain_order=True):
        yield int(ep), data


def _texts(sample):
    texts = [str(t.get("task", "")) for t in sample.tasks]
    if texts:
        return texts
    if not sample.frames.is_empty() and "task" in sample.frames.columns:
        return [str(x) for x in sample.frames.get_column("task").unique().to_list()]
    return []


def empty_or_placeholder_task(sample):
    texts = _texts(sample)
    if not texts:
        if sample.info.get("total_tasks"):
            return _score("empty_or_placeholder_task", 40, "info.json declares tasks but no task strings were found", {})
        return _skip("empty_or_placeholder_task", "no task metadata")
    bad = [t.strip() for t in texts if t.strip().lower() in PLACEHOLDERS or PLACEHOLDER_RE.match(t.strip())]
    return _score("empty_or_placeholder_task", 100 * (1 - len(bad) / len(texts)), f"{len(bad)}/{len(texts)} task labels are empty or placeholders", {"placeholders": bad[:20], "n_tasks": len(texts)})


def task_coverage(sample):
    if not sample.tasks:
        return _skip("task_coverage", "no task table")
    valid = {int(t["task_index"]) for t in sample.tasks}
    if not sample.frames.is_empty() and "task_index" in sample.frames.columns:
        indices = [int(x) for x in sample.frames.get_column("task_index").unique().to_list() if x is not None]
    else:
        indices = [int(x["task_index"]) for x in sample.episode_meta if x.get("task_index") is not None]
    if not indices:
        return _skip("task_coverage", "no task_index on frames or episodes")
    missing = sorted(set(indices) - valid)
    score = 100 if not missing else max(0, 100 - 15 * len(missing))
    return _score("task_coverage", score, "all task_index values resolve" if not missing else f"{len(missing)} task_index values missing from meta/tasks", {"missing_task_index": missing[:50], "n_task_rows": len(valid)})


def task_specificity(sample):
    texts = [x.strip() for x in _texts(sample) if x.strip()]
    if not texts:
        return _skip("task_specificity", "no task text")
    short = [x for x in texts if len(x) < 12]
    unique = {x.lower() for x in texts}
    episodes = sample.info.get("total_episodes", 0)
    score = 100 - 40 * len(short) / len(texts)
    if len(unique) == 1 and episodes >= 20 and len(next(iter(unique))) < 24:
        score -= 25
    return _score("task_specificity", max(0, score), f"{len(unique)} unique task string(s); {len(short)} shorter than 12 characters", {"n_unique": len(unique), "short_examples": short[:10], "n_episodes": episodes})


def required_lerobot_fields(sample):
    info = sample.info
    missing_info = [k for k in ("codebase_version", "fps", "features", "data_path") if k not in info]
    columns = sample.frames.columns if not sample.frames.is_empty() else []
    missing_columns = []
    if columns:
        if first_present(columns, ("episode_index", "episode_idx")) is None:
            missing_columns.append("episode_index")
        if first_present(columns, FRAME_EXACT) is None:
            missing_columns.append("frame_index")
        if first_present(columns, TIME_EXACT) is None:
            missing_columns.append("timestamp")
        if first_present(columns, ACTION_EXACT) is None and first_present(columns, STATE_EXACT) is None:
            missing_columns.append("action|observation.state")
    elif info:
        missing_columns.append("tabular_frames")
    score = max(0, 100 - 20 * len(missing_info) - 15 * len(missing_columns))
    if not info:
        score = min(score, 20)
    return _score("required_lerobot_fields", score, "LeRobot schema looks complete" if score >= 80 else "required LeRobot metadata or columns are missing", {"missing_info_keys": missing_info, "missing_columns": missing_columns, "n_features": len(info.get("features") or {})})


def fps_consistency(sample):
    fps = sample.info.get("fps")
    column = first_present(sample.frames.columns, TIME_EXACT)
    if not fps or column is None or sample.frames.is_empty():
        return _skip("fps_consistency", "need declared fps and timestamp column")
    expected = 1 / float(fps)
    flagged = []
    total = 0
    for ep, data in _episodes(sample.frames):
        ts = data.get_column(column).cast(pl.Float64, strict=False).to_numpy()
        if len(ts) < 3:
            continue
        total += 1
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if not len(dt) or abs(float(np.median(dt)) - expected) / expected > .15:
            flagged.append(ep)
    if not total:
        return _skip("fps_consistency", "episodes too short to estimate dt")
    return _score("fps_consistency", 100 * (1 - len(set(flagged)) / total), f"{len(set(flagged))}/{total} episodes disagree with declared fps={fps}", {"declared_fps": fps}, sorted(set(flagged)))


def timestamp_monotonic(sample):
    column = first_present(sample.frames.columns, TIME_EXACT)
    if column is None or sample.frames.is_empty():
        return _skip("timestamp_monotonic", "no timestamp column")
    flagged = []
    total = 0
    for ep, data in _episodes(sample.frames):
        ts = data.get_column(column).cast(pl.Float64, strict=False).to_numpy()
        if len(ts) < 2:
            continue
        total += 1
        if np.any(np.diff(ts) <= 0):
            flagged.append(ep)
    if not total:
        return _skip("timestamp_monotonic", "no multi-frame episodes")
    return _score("timestamp_monotonic", 100 * (1 - len(set(flagged)) / total), f"{len(set(flagged))}/{total} episodes have non-increasing timestamps", {}, sorted(set(flagged)))


def episode_length_anomaly(sample):
    if sample.frames.is_empty() or "episode_index" not in sample.frames.columns:
        return _skip("episode_length_anomaly", "no episode lengths")
    lengths = {int(row["episode_index"]): int(row["len"]) for row in sample.frames.group_by("episode_index").len().iter_rows(named=True)}
    values = np.array(list(lengths.values()), dtype=float)
    median = float(np.median(values))
    flagged = [ep for ep, length in lengths.items() if length <= 1 or (median > 0 and length > max(50, 8 * median))]
    return _score("episode_length_anomaly", 100 * (1 - len(flagged) / len(lengths)), f"{len(flagged)}/{len(lengths)} episodes are empty, single-frame, or extreme outliers", {"median_length": median, "min": int(values.min()), "max": int(values.max())}, flagged[:200])


def declared_video_missing(sample):
    features = sample.info.get("features") or {}
    video_keys = [key for key, spec in features.items() if isinstance(spec, dict) and spec.get("dtype") == "video"]
    if not video_keys:
        return _skip("declared_video_missing", "no video features declared")
    if not sample.repo_files:
        return _skip("declared_video_missing", "file listing unavailable")
    video_files = [x for x in sample.repo_files if x.replace("\\", "/").startswith("videos/") and x.endswith(".mp4")]
    return _score("declared_video_missing", 100 if video_files else 0, "video shards present" if video_files else "info.json declares video features but no videos/*.mp4 files were found", {"n_video_features": len(video_keys), "n_mp4": len(video_files), "video_keys": video_keys[:12]})


ANNOTATION_CHECKS = (empty_or_placeholder_task, task_coverage, task_specificity, required_lerobot_fields, fps_consistency, timestamp_monotonic, episode_length_anomaly, declared_video_missing)
