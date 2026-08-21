from __future__ import annotations

import numpy as np
import polars as pl

from cleanbench.columns import ACTION_EXACT, FORCE_RE, STATE_EXACT, TORQUE_RE, VELOCITY_RE, first_present, l2_rows, match_columns, series_to_2d, zscores
from cleanbench.types import CheckResult, DatasetSample


def _skipped(check_id: str, reason: str) -> CheckResult:
    return CheckResult(check_id, "physical", "skipped", None, reason)


def _groups(frames: pl.DataFrame):
    if frames.is_empty():
        return
    if "episode_index" not in frames.columns:
        yield 0, frames
        return
    for (ep,), ep_df in frames.group_by("episode_index", maintain_order=True):
        yield int(ep), ep_df


def _matrix(ep_df: pl.DataFrame, col: str):
    if col not in ep_df.columns:
        return None
    arr = series_to_2d(ep_df.get_column(col))
    return arr if arr.size else None


def _diff(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] < 2:
        return np.zeros_like(arr)
    return np.vstack([np.diff(arr, axis=0)[:1], np.diff(arr, axis=0)])


def _result(check_id, flagged, total, summary, metrics):
    if not total:
        return _skipped(check_id, "no episodes in sample")
    score = round(100.0 * (1.0 - len(set(flagged)) / total), 2)
    return CheckResult(check_id, "physical", "pass" if score >= 80 else "fail", score, summary, sorted(set(flagged)), metrics)


def velocity_spike(sample: DatasetSample):
    frames = sample.frames
    velocity = match_columns(frames.columns, VELOCITY_RE)
    state = first_present(frames.columns, STATE_EXACT)
    values = []
    derived = False
    for ep, data in _groups(frames):
        if velocity:
            arr = _matrix(data, velocity[0])
        elif state:
            arr = _matrix(data, state)
            if arr is not None:
                arr = _diff(arr)
                derived = True
        else:
            return _skipped("velocity_spike", "no velocity or state columns")
        if arr is not None:
            values.append((ep, l2_rows(arr)))
    if derived:
        sample.warnings.append("velocity_spike used finite-differenced state")
    if not values:
        return _skipped("velocity_spike", "could not build velocity series")
    z = zscores(np.concatenate([x for _, x in values]))
    flagged, offset = [], 0
    for ep, vals in values:
        if np.any(z[offset:offset + len(vals)] > 3.5):
            flagged.append(ep)
        offset += len(vals)
    return _result("velocity_spike", flagged, len(values), f"{len(set(flagged))}/{len(values)} episodes have velocity z > 3.5", {"derived_velocity": derived, "threshold_z": 3.5})


def acceleration_spike(sample: DatasetSample):
    frames = sample.frames
    velocity = match_columns(frames.columns, VELOCITY_RE)
    state = first_present(frames.columns, STATE_EXACT)
    values = []
    for ep, data in _groups(frames):
        arr = _matrix(data, velocity[0]) if velocity else (_matrix(data, state) if state else None)
        if arr is None:
            return _skipped("acceleration_spike", "no velocity or state columns")
        if not velocity:
            arr = _diff(arr)
        values.append((ep, l2_rows(_diff(arr))))
    if not values:
        return _skipped("acceleration_spike", "could not build acceleration series")
    z = zscores(np.concatenate([x for _, x in values]))
    flagged, offset = [], 0
    for ep, vals in values:
        if np.any(z[offset:offset + len(vals)] > 4.0):
            flagged.append(ep)
        offset += len(vals)
    return _result("acceleration_spike", flagged, len(values), f"{len(set(flagged))}/{len(values)} episodes have acceleration z > 4.0", {"threshold_z": 4.0})


def torque_saturation(sample: DatasetSample):
    columns = match_columns(sample.frames.columns, TORQUE_RE)
    if not columns:
        return _skipped("torque_saturation", "no torque/effort columns")
    values = []
    maxima = []
    for ep, data in _groups(sample.frames):
        arr = _matrix(data, columns[0])
        if arr is not None:
            row_max = np.nanmax(np.abs(arr), axis=1)
            values.append((ep, row_max))
            maxima.append(np.nanmax(row_max))
    if not values:
        return _skipped("torque_saturation", "empty torque series")
    ceiling = float(np.nanmax(maxima))
    if ceiling <= 0:
        return _skipped("torque_saturation", "torque magnitude is zero")
    flagged = [ep for ep, vals in values if np.any(vals > 0.85 * ceiling)]
    return _result("torque_saturation", flagged, len(values), f"{len(set(flagged))}/{len(values)} episodes reach >85% of sample torque max", {"observed_max": ceiling, "ratio": 0.85})


def force_spike(sample: DatasetSample):
    columns = match_columns(sample.frames.columns, FORCE_RE)
    if not columns:
        return _skipped("force_spike", "no force/wrench columns")
    values = []
    for ep, data in _groups(sample.frames):
        arr = _matrix(data, columns[0])
        if arr is not None:
            values.append((ep, l2_rows(arr)))
    if not values:
        return _skipped("force_spike", "empty force series")
    z = zscores(np.concatenate([x for _, x in values]))
    flagged, offset = [], 0
    for ep, vals in values:
        if np.any(z[offset:offset + len(vals)] > 4.0):
            flagged.append(ep)
        offset += len(vals)
    return _result("force_spike", flagged, len(values), f"{len(set(flagged))}/{len(values)} episodes have force z > 4.0", {"threshold_z": 4.0, "column": columns[0]})


def repeated_action(sample: DatasetSample):
    column = first_present(sample.frames.columns, ACTION_EXACT)
    if not column:
        return _skipped("repeated_action", "no action column")
    flagged, total = [], 0
    for ep, data in _groups(sample.frames):
        arr = _matrix(data, column)
        if arr is None or len(arr) < 2:
            continue
        total += 1
        run = longest = 0
        for same in np.max(np.abs(np.diff(arr, axis=0)), axis=1) < 1e-4:
            run = run + 1 if same else 0
            longest = max(longest, run)
        if longest >= 8:
            flagged.append(ep)
    return _result("repeated_action", flagged, total, f"{len(set(flagged))}/{total} episodes repeat an action for 8+ frames", {"run_threshold": 8, "eps": 1e-4})


PHYSICAL_CHECKS = (velocity_spike, acceleration_spike, torque_saturation, force_spike, repeated_action)
