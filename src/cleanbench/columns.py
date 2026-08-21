from __future__ import annotations

import re

import numpy as np
import polars as pl

STATE_EXACT = ("observation.state", "observation.qpos", "state", "observation.joint_positions")
ACTION_EXACT = ("action", "observation.action")
FRAME_EXACT = ("frame_index", "frame_idx")
TIME_EXACT = ("timestamp", "time", "t")

VELOCITY_RE = re.compile(r"(velocity|qvel|joint_vel)", re.I)
TORQUE_RE = re.compile(r"(torque|effort|joint_effort|motor_torque)", re.I)
FORCE_RE = re.compile(r"(force|wrench|ft_sensor|contact_force)", re.I)


def first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def match_columns(columns: list[str], pattern: re.Pattern[str]) -> list[str]:
    return [c for c in columns if pattern.search(c)]


def flatten_singleton_lists(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    out = df
    for col in df.columns:
        dtype = out[col].dtype
        if dtype != pl.List:
            continue
        if dtype.inner == pl.List:
            continue
        lengths = out[col].list.len()
        if lengths.min() == 1 and lengths.max() == 1:
            out = out.with_columns(pl.col(col).list.first().alias(col))
    return out


def series_to_2d(series: pl.Series) -> np.ndarray:
    if series.dtype == pl.List:
        return np.asarray(series.to_list(), dtype=np.float64)
    arr = np.asarray(series.to_numpy(), dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def l2_rows(mat: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(mat), axis=1))


def zscores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    if not np.isfinite(sigma) or sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma
