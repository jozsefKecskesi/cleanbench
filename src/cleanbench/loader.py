from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from .columns import flatten_singleton_lists
from .types import DatasetSample

INFO_CANDIDATES = ("meta/info.json", "info.json")
TASK_JSONL = ("meta/tasks.jsonl", "tasks.jsonl")
TASK_PARQUET = ("meta/tasks.parquet",)
EPISODE_JSONL = ("meta/episodes.jsonl", "episodes.jsonl")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def detect_version(info: dict[str, Any]) -> str:
    raw = str(info.get("codebase_version") or info.get("codebaseVersion") or "")
    if raw:
        return raw if raw.startswith("v") else f"v{raw}"
    data_path = str(info.get("data_path") or "")
    if "file-{file_index" in data_path:
        return "v3.0"
    if "episode_" in data_path:
        return "v2.1"
    return "unknown"


def load_local(root: Path, dataset_id: str, n_episodes: int, max_frames: int) -> DatasetSample:
    warnings: list[str] = []
    info: dict[str, Any] = {}
    for rel in INFO_CANDIDATES:
        path = root / rel
        if path.exists():
            info = _read_json(path)
            break
    if not info:
        warnings.append("meta/info.json not found")
    tasks: list[dict[str, Any]] = []
    for rel in TASK_JSONL:
        path = root / rel
        if path.exists():
            tasks = _read_jsonl(path)
            break
    if not tasks:
        for rel in TASK_PARQUET:
            path = root / rel
            if path.exists():
                tasks = pl.read_parquet(path).to_dicts()
                break
    episode_meta: list[dict[str, Any]] = []
    for rel in EPISODE_JSONL:
        path = root / rel
        if path.exists():
            episode_meta = _read_jsonl(path)
            break
    ep_dir = root / "meta" / "episodes"
    if not episode_meta and ep_dir.exists():
        parts = sorted(ep_dir.rglob("*.parquet"))
        if parts:
            episode_meta = pl.read_parquet([str(p) for p in parts]).to_dicts()
    data_files = sorted((root / "data").rglob("*.parquet")) if (root / "data").exists() else []
    frames = _read_episode_cap(data_files, n_episodes, max_frames, warnings)
    repo_files = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()]
    return DatasetSample(
        dataset_id=dataset_id,
        revision=None,
        info=info,
        tasks=_normalise_tasks(tasks),
        episode_meta=episode_meta,
        frames=frames,
        warnings=warnings,
        repo_files=repo_files,
        format_version=detect_version(info),
    )


def load_hub(dataset_id: str, n_episodes: int, max_frames: int, revision: str | None) -> DatasetSample:
    from huggingface_hub import hf_hub_download, list_repo_files

    warnings: list[str] = []
    files = list_repo_files(dataset_id, repo_type="dataset", revision=revision)
    info = _download_json(dataset_id, files, INFO_CANDIDATES, revision, warnings)
    tasks = _download_jsonl(dataset_id, files, TASK_JSONL, revision)
    if not tasks:
        tasks = _download_parquet_dicts(dataset_id, files, "meta/tasks.parquet", revision)
    episode_meta = _download_jsonl(dataset_id, files, EPISODE_JSONL, revision)
    if not episode_meta:
        ep_parts = sorted(f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet"))
        episode_meta = _download_parquet_dicts_many(dataset_id, ep_parts[:8], revision, warnings)
    data_rel = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    local_parquets: list[Path] = []
    for rel in data_rel[:64]:
        try:
            local_parquets.append(Path(hf_hub_download(dataset_id, rel, repo_type="dataset", revision=revision)))
        except Exception as exc:
            warnings.append(f"could not download {rel}: {exc}")
            break
        if _enough_episodes(local_parquets, n_episodes, max_frames):
            break
    frames = _read_episode_cap(local_parquets, n_episodes, max_frames, warnings)
    return DatasetSample(
        dataset_id=dataset_id,
        revision=revision or "main",
        info=info,
        tasks=_normalise_tasks(tasks),
        episode_meta=episode_meta,
        frames=frames,
        warnings=warnings,
        repo_files=files,
        format_version=detect_version(info),
    )


def load_dataset_sample(dataset_id: str, n_episodes: int = 50, max_frames: int = 100_000, revision: str | None = None, local_dir: str | Path | None = None) -> DatasetSample:
    if local_dir is not None:
        return load_local(Path(local_dir), dataset_id, n_episodes, max_frames)
    path = Path(dataset_id)
    if path.exists() and path.is_dir():
        return load_local(path, path.name, n_episodes, max_frames)
    return load_hub(dataset_id, n_episodes, max_frames, revision)


def _download_json(dataset_id, files, candidates, revision, warnings) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    for rel in candidates:
        if rel in files:
            path = Path(hf_hub_download(dataset_id, rel, repo_type="dataset", revision=revision))
            return json.loads(path.read_text(encoding="utf-8"))
    warnings.append("meta/info.json not found on Hub")
    return {}


def _download_jsonl(dataset_id, files, candidates, revision) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    for rel in candidates:
        if rel in files:
            path = Path(hf_hub_download(dataset_id, rel, repo_type="dataset", revision=revision))
            return _read_jsonl(path)
    return []


def _download_parquet_dicts(dataset_id, files, rel, revision) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    if rel not in files:
        return []
    path = hf_hub_download(dataset_id, rel, repo_type="dataset", revision=revision)
    return pl.read_parquet(path).to_dicts()


def _download_parquet_dicts_many(dataset_id, rels, revision, warnings) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    paths = []
    for rel in rels:
        try:
            paths.append(hf_hub_download(dataset_id, rel, repo_type="dataset", revision=revision))
        except Exception as exc:
            warnings.append(f"episode meta skip {rel}: {exc}")
    if not paths:
        return []
    return pl.read_parquet(paths).to_dicts()


def _normalise_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(tasks):
        text = row.get("task") or row.get("prompt") or row.get("instruction") or row.get("language") or ""
        idx = row.get("task_index", row.get("index", i))
        out.append({"task_index": int(idx) if idx is not None else i, "task": str(text)})
    return out


def _enough_episodes(paths: list[Path], n_episodes: int, max_frames: int) -> bool:
    if not paths:
        return False
    try:
        cols = pl.read_parquet(paths[0], n_rows=1).columns
        if "episode_index" not in cols:
            return len(paths) >= n_episodes
        lf = pl.scan_parquet([str(p) for p in paths])
        n = lf.select("episode_index").unique().select(pl.len()).collect().item()
        frames = lf.select(pl.len()).collect().item()
        return n >= n_episodes or frames >= max_frames
    except Exception:
        return len(paths) >= max(4, n_episodes // 10)


def _read_episode_cap(paths: list[Path], n_episodes: int, max_frames: int, warnings: list[str]) -> pl.DataFrame:
    if not paths:
        warnings.append("no parquet shards under data/")
        return pl.DataFrame()
    try:
        df = pl.read_parquet([str(p) for p in paths])
    except Exception as exc:
        warnings.append(f"parquet read failed: {exc}")
        return pl.DataFrame()
    df = flatten_singleton_lists(df)
    if "episode_index" in df.columns:
        keep = df.select("episode_index").unique(maintain_order=True).head(n_episodes).get_column("episode_index").to_list()
        df = df.filter(pl.col("episode_index").is_in(keep))
    if df.height > max_frames:
        df = df.head(max_frames)
    return df
