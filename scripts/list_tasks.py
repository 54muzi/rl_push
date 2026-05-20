#!/usr/bin/env python3
"""List registered G1 loco-manipulation tasks."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import gymnasium as gym  # noqa: E402

import g1_loco_manipulation.tasks  # noqa: F401, E402


def main():
    rows = []
    for task_spec in gym.registry.values():
        if task_spec.id.startswith("G1-"):
            rows.append((task_spec.id, task_spec.entry_point, task_spec.kwargs.get("env_cfg_entry_point", "")))

    if not rows:
        print("No G1 tasks registered.")
        return

    print("Available G1 tasks")
    print("==================")
    for task_id, entry_point, cfg in sorted(rows):
        print(f"{task_id}\n  entry_point: {entry_point}\n  env_cfg:     {cfg}")


if __name__ == "__main__":
    main()
