#!/usr/bin/env python3
"""Run SFUMATO preprocessing followed by GPU BGM from one shared config.

The preprocessing step creates one sparse-SVD cache. The GPU step fits one
oversampled BGM using max(K_LIST), builds a shared component dendrogram, and
cuts it at all requested K values.
"""

from __future__ import annotations

import argparse

from preprocess import run_preprocess
from run_bgm_gpu import run_bgm
from utils_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Overwrite existing preprocessing cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_preprocess(config, force=args.force_preprocess)
    run_bgm(config)


if __name__ == "__main__":
    main()