from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .scorer import score_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cleanbench", description="Score a LeRobot-format dataset.")
    parser.add_argument("--version", action="version", version=f"cleanbench {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    score = sub.add_parser("score", help="Score a Hub dataset ID or local folder")
    score.add_argument("dataset_id", help="Hugging Face dataset ID (org/name) or local path")
    score.add_argument("-o", "--output", type=Path, help="Write JSON report to this path")
    score.add_argument("-n", "--n-episodes", type=int, default=50)
    score.add_argument("--max-frames", type=int, default=100_000)
    score.add_argument("--revision", default=None)
    score.add_argument("--local-dir", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "score":
        report = score_dataset(
            args.dataset_id,
            n_episodes=args.n_episodes,
            max_frames=args.max_frames,
            revision=args.revision,
            local_dir=args.local_dir,
        )
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text)
        return 0 if report.get("error") is None else 2
    return 1


if __name__ == "__main__":n    sys.exit(main())
