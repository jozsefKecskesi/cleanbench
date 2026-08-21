from __future__ import annotations

import numpy as np
import polars as pl

from cleanbench.columns import ACTION_EXACT, STATE_EXACT, first_present, series_to_2d
from cleanbench.types import CheckResult, DatasetSample


def _skip(check_id: str, reason: str) -> CheckResult:
    return CheckResult(check_id, "integrity", "skipped", None, reason)


def _iter_episodes(frames: pl.DataFrame):
    if "episode_index" in frames.columns:
        for (ep,), ep_df in frames.group_by("episode_index", maintain_order=True):
            yield int(ep), ep_df
    else:
        yield 0, frames


def nan_inf(sample: DatasetSample) -> CheckResult:
    frames = sample.frames
    if frames.is_empty():
        return _skip("nan_inf", "no frames")
    cols = [c for c in (first_present(frames.columns, STATE_EXACT), first_present(frames.columns, ACTION_EXACT)) if c]
    if not cols:
        numeric = [c for c, dt in zip(frames.columns, frames.dtypes) if dt.is_numeric()]
        cols = numeric[:6]
    if not cols:
        return _skip("nan_inf", "no numeric columns")
    flagged = []
    total = 0
    for ep, ep_df in _iter_episodes(frames):
        total += 1
        for col in cols:
            arr = series_to_2d(ep_df.get_column(col))
            if np.isnan(arr).any() or np.isinf(arr).any():
                flagged.append(ep)
                break
    score = 100.0 * (1.0 - len(set(flagged)) / max(total, 1))
    return CheckResult("nan_inf", "integrity", "pass" if score >= 80 else "fail", round(score, 2), f"{len(set(flagged))}/{total} episodes contain NaN or Inf", sorted(set(flagged)), {"columns_checked": cols})


def constant_sequence(sample: DatasetSample) -> CheckResult:
    frames = sample.frames
    state_col = first_present(frames.columns, STATE_EXACT)
    if frames.is_empty() or state_col is None or "episode_index" not in frames.columns:
        return _skip("constant_sequence", "need observation.state and episode_index")
    mats = []
    ids = []
    for ep, ep_df in _iter_episodes(frames):
        mat = series_to_2d(ep_df.get_column(state_col))
        if mat.shape[0] < 4:
            continue
        mats.append(mat)
        ids.append(ep)
    if not mats:
        return _skip("constant_sequence", "episodes too short")
    stacked = np.vstack(mats)
    global_std = np.nanstd(stacked, axis=0)
    variable = global_std > 1e-3
    if not np.any(variable):
        return _skip("constant_sequence", "all state channels globally constant")
    ep_stds = [np.nanstd(m, axis=0) for m in mats]
    typical = np.nanmean(np.vstack(ep_stds), axis=0)
    flagged = []
    for ep, std in zip(ids, ep_stds):
        frozen = variable & (typical > 1e-6) & (std < 0.05 * typical)
        if np.any(frozen):
            flagged.append(ep)
    score = 100.0 * (1.0 - len(flagged) / len(ids))
    return CheckResult("constant_sequence", "integrity", "pass" if score >= 80 else "fail", round(score, 2), f"{len(flagged)}/{len(ids)} episodes freeze a normally-variable state channel", flagged, {})


INTEGRITY_CHECKS = (nan_inf, constant_sequence)
