"""Launch the official Trossen Stationary MuJoCo demo from GlassEgo."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TROSSEN_REPO = Path("/home/huasan/trossen_arm_mujoco")
DEFAULT_DEMO_NAME = "stationary_ai_pick_place.py"


def find_demo(repo_root: Path, demo_name: str) -> Path:
    repo_root = repo_root.expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Trossen MuJoCo repo not found: {repo_root}")

    matches = sorted(repo_root.rglob(demo_name))
    if not matches:
        raise FileNotFoundError(f"Demo script {demo_name!r} not found under {repo_root}")
    return matches[0]


def run_headless(repo_root: Path, steps: int) -> None:
    if steps <= 0:
        raise ValueError("--headless-steps must be positive")

    repo_root = repo_root.expanduser().resolve()
    sys.path.insert(0, str(repo_root))

    import mujoco
    from trossen_arm_mujoco.scripts.stationary_ai_pick_place import StationaryAIPickPlace

    old_cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        task = StationaryAIPickPlace()
        task.setup_scene()
        task.reset()

        for _ in range(steps):
            if not task.is_done():
                task.forward()
            mujoco.mj_step(task.model, task.data)

        cube_position, cube_orientation = task.get_cube_pose()
    finally:
        os.chdir(old_cwd)

    print(f"[ok] headless steps: {steps}")
    print(f"[ok] task done: {task.is_done()}")
    print(f"[ok] cube position: {cube_position.round(4).tolist()}")
    print(f"[ok] cube orientation: {cube_orientation.round(4).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_TROSSEN_REPO)
    parser.add_argument("--demo", default=DEFAULT_DEMO_NAME)
    parser.add_argument("--headless-steps", type=int, default=None)
    args = parser.parse_args()

    if args.headless_steps is not None:
        run_headless(args.repo_root, args.headless_steps)
        return

    demo_path = find_demo(args.repo_root, args.demo)
    print(f"[run] {demo_path}")
    subprocess.run([sys.executable, str(demo_path)], cwd=str(args.repo_root), check=True)


if __name__ == "__main__":
    main()
