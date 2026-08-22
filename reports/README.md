# Sample reports

These JSON files are **sample** CleanBench v0.1 scans, not certificates.

Each file was produced with:

```bash
cleanbench score <hub-id> -n 10 -o reports/<slug>.json
```

v0.1 reads parquet + `meta/` only. It does not download or decode video.

Treat physical scores as a screening signal. If a report warns
`velocity_spike used finite-differenced state`, velocity and acceleration
were derived from `observation.state` and can false-positive.

| File | Dataset |
| --- | --- |
| `pusht.json` | `lerobot/pusht` |
| `aloha_sim_insertion_human.json` | `lerobot/aloha_sim_insertion_human` |
| `aloha_sim_transfer_cube_human.json` | `lerobot/aloha_sim_transfer_cube_human` |
| `columbia_cairlab_pusht_real.json` | `lerobot/columbia_cairlab_pusht_real` |
| `umi_cup_in_the_wild.json` | `lerobot/umi_cup_in_the_wild` |
| `droid_100.json` | `lerobot/droid_100` |
| `xarm_lift_medium.json` | `lerobot/xarm_lift_medium` |
| `nyu_franka_play_dataset.json` | `lerobot/nyu_franka_play_dataset` |
| `stanford_hydra_dataset.json` | `lerobot/stanford_hydra_dataset` |
| `berkeley_autolab_ur5.json` | `lerobot/berkeley_autolab_ur5` |