from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class CheckResult:
    id: str
    category: str
    status: str
    score: float | None
    summary: str
    flagged_episodes: list[int] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "status": self.status,
            "score": self.score,
            "summary": self.summary,
            "flagged_episodes": self.flagged_episodes,
            "metrics": self.metrics,
        }


@dataclass
class DatasetSample:
    dataset_id: str
    revision: str | None
    info: dict[str, Any]
    tasks: list[dict[str, Any]]
    episode_meta: list[dict[str, Any]]
    frames: pl.DataFrame
    warnings: list[str] = field(default_factory=list)
    repo_files: list[str] = field(default_factory=list)
    format_version: str = "unknown"
