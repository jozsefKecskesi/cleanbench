# CleanBench

Open scorer for **LeRobot-format** robot datasets. v0.1 returns a structured JSON
report for any Hugging Face dataset ID (`org/name`) that follows LeRobot v2.1 or v3.0 layout.

CleanBench is a dataset quality & curation guild. This repo is the scoring engine.
Scores are a screening signal, not a certificate.

```bash
pip install -e .
cleanbench score lerobot/pusht -o reports/pusht.json
```

```python
from cleanbench import score_dataset
report = score_dataset("lerobot/pusht", n_episodes=50)
print(report["scores"]["overall"], report["band"])
```

Complementary to [HaptalAI's robotics quality leaderboard](https://huggingface.co/datasets/HaptalAI/robotics-quality-leaderboard): they screen physical failure modes; CleanBench adds annotation-quality checks and (later) a human curation workflow.

## Check catalog (v0.1)

### Physical anomaly checks

| ID | What it flags | Skip when |
| --- | --- | --- |
| `velocity_spike` | Frame whose velocity L2 z-score > 3.5 | no state/velocity |
| `acceleration_spike` | Acceleration magnitude z-score > 4.0 | no velocity or state |
| `torque_saturation` | |torque| > 85% of sample max | no torque/effort columns |
| `force_spike` | Force/wrench L2 z-score > 4.0 | no force/wrench columns |
| `repeated_action` | Same action vector for 8+ consecutive frames | no action column |

Velocity is taken from columns matching `velocity` / `qvel`. If missing, it is derived as the per-episode finite difference of `observation.state` (noisier; a warning is recorded).

### Annotation-quality checks

| ID | What it flags |
| --- | --- |
| `empty_or_placeholder_task` | Empty, whitespace, or placeholder task text (`task`, `n/a`, `todo`, `unknown`) |
| `task_coverage` | Episodes whose `task_index` is missing from `meta/tasks` |
| `task_specificity` | Very short task strings (< 12 chars) or a single generic label covering every episode |
| `required_lerobot_fields` | Missing `meta/info.json` fields or core columns (`episode_index`, `frame_index`, `timestamp`, `action` / `observation.state`) |
| `fps_consistency` | Median per-episode timestamp step disagrees with declared `fps` by > 15% |
| `timestamp_monotonic` | Non-increasing timestamps inside an episode |
| `episode_length_anomaly` | Length 0-1 frames, or length far outside the sample distribution |
| `declared_video_missing` | Video features declared in `info.json` but no `videos/` files on the Hub |

### Integrity checks

| ID | What it flags |
| --- | --- |
| `nan_inf` | NaN / Inf in numeric state or action |
| `constant_sequence` | A normally-variable channel is frozen inside an episode |

Missing sensors skip a check; they do not penalise the score. v0.1 reads parquet + `meta/` only — it does not download or decode videos.

## JSON report shape

Required keys on every run: `schema_version`, `dataset_id`, `format`, `sample`, `scores`, `band`, `checks`, `flagged_episodes`, `warnings`, `error`. Invalid Hub IDs still return this schema with `error` set.

Bands: `clean` >= 80, `review` 60-79, `flagged` < 60.

## Install

Python 3.10+.

```bash
git clone https://github.com/jozsefKecskesi/cleanbench.git
cd cleanbench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## License

Apache-2.0
