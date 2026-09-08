"""Validate the local MuJoCo + Trossen simulation install."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


DEFAULT_TROSSEN_REPO = Path("/home/huasan/trossen_arm_mujoco")


def require_module(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(f"Python module not found: {name}")
    module = __import__(name)
    location = getattr(module, "__file__", spec.origin)
    print(f"[ok] import {name}: {location}")
    return module


def candidate_roots(package_module, repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    package_file = getattr(package_module, "__file__", None)
    if package_file:
        roots.append(Path(package_file).resolve().parent)
    roots.append(repo_root.expanduser().resolve())
    return list(dict.fromkeys(root for root in roots if root.exists()))


def find_stationary_xml(roots: list[Path]) -> Path:
    matches: list[Path] = []
    for root in roots:
        matches.extend(
            path
            for path in root.rglob("*.xml")
            if "stationary" in path.as_posix().lower()
        )

    if not matches:
        searched = ", ".join(str(root) for root in roots) or "<none>"
        raise RuntimeError(f"No Stationary XML found under: {searched}")

    matches.sort(key=lambda path: ("scene" not in path.name.lower(), len(path.parts), str(path)))
    return matches[0]


def load_model(mujoco, xml_path: Path):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"[ok] loaded XML: {xml_path}")
    print(f"[ok] model: bodies={model.nbody} joints={model.njnt} actuators={model.nu}")
    return model, data


def render_once(mujoco, model, data) -> None:
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        renderer.update_scene(data)
        image = renderer.render()
    finally:
        renderer.close()
    print(f"[ok] offscreen render: shape={image.shape} dtype={image.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_TROSSEN_REPO)
    parser.add_argument("--render-check", action="store_true")
    args = parser.parse_args()

    print(f"[info] python executable check is running in: {Path.cwd()}")
    print(f"[info] DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    print(f"[info] MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")

    mujoco = require_module("mujoco")
    trossen_pkg = require_module("trossen_arm_mujoco")
    xml_path = find_stationary_xml(candidate_roots(trossen_pkg, args.repo_root))
    model, data = load_model(mujoco, xml_path)

    if args.render_check:
        render_once(mujoco, model, data)


if __name__ == "__main__":
    main()
