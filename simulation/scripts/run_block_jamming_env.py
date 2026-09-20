"""Run and inspect the BRX20260825 Block Jamming MuJoCo scene."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = SIMULATION_ROOT / "tasks" / "block_jamming_brx.xml"
BLOCK_GEOM = "block_geom"
BLOCK_JOINT = "block_freejoint"
TARGET_BOUNDS_GEOM = "target_container_bounds"
TARGET_FLOOR_GEOM = "target_container_floor"
TABLE_GEOM = "task_table"
DEFAULT_CAMERA_NAME = "ego_rgbd"
BLOCK_HALF_SIZE = 0.015
TARGET_WALL_THICKNESS = 0.004


def object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(value)


def load_scene(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not xml_path.is_file():
        raise FileNotFoundError(f"BRX MuJoCo XML not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def geom_center_and_size(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    geom_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xpos[geom_id].copy(), model.geom_size[geom_id].copy()


def set_block_pose(model: mujoco.MjModel, data: mujoco.MjData, position: np.ndarray, yaw: float = 0.0) -> None:
    joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, BLOCK_JOINT)
    qpos = int(model.jnt_qposadr[joint_id])
    qvel = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos : qpos + 3] = np.asarray(position, dtype=float)
    data.qpos[qpos + 3 : qpos + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qvel[qvel : qvel + 6] = 0
    mujoco.mj_forward(model, data)


def set_target_pose(model: mujoco.MjModel, data: mujoco.MjData, position: np.ndarray) -> None:
    """Move the fixed blue tray body and refresh derived MuJoCo state."""
    body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "target_container")
    target = np.asarray(position, dtype=float).reshape(3)
    if not np.all(np.isfinite(target)):
        raise ValueError("target position must contain three finite values")
    model.body_pos[body_id] = target
    mujoco.mj_forward(model, data)


def randomize_block(model: mujoco.MjModel, data: mujoco.MjData, seed: int) -> np.ndarray:
    """Sample a reachable tabletop pose without overlapping the target container."""
    rng = np.random.default_rng(seed)
    target_center, target_size = geom_center_and_size(model, data, TARGET_BOUNDS_GEOM)
    _, block_size = geom_center_and_size(model, data, BLOCK_GEOM)
    table_center, table_size = geom_center_and_size(model, data, TABLE_GEOM)
    tabletop_z = table_center[2] + table_size[2] + BLOCK_HALF_SIZE
    for _ in range(100):
        candidate = np.array([rng.uniform(0.60, 0.65), rng.uniform(-0.23, -0.15), tabletop_z])
        if np.all(np.abs(candidate[:2] - target_center[:2]) <= target_size[:2] + block_size[:2] + 0.001):
            continue
        set_block_pose(model, data, candidate, rng.uniform(-np.pi, np.pi))
        return candidate
    raise RuntimeError("Unable to sample a non-overlapping red block pose")


def place_block_in_target(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    target_center, _ = geom_center_and_size(model, data, TARGET_BOUNDS_GEOM)
    floor_center, floor_size = geom_center_and_size(model, data, TARGET_FLOOR_GEOM)
    set_block_pose(model, data, [target_center[0], target_center[1], floor_center[2] + floor_size[2] + BLOCK_HALF_SIZE])


def scene_status(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    block_center, block_size = geom_center_and_size(model, data, BLOCK_GEOM)
    target_center, target_size = geom_center_and_size(model, data, TARGET_BOUNDS_GEOM)
    floor_center, floor_size = geom_center_and_size(model, data, TARGET_FLOOR_GEOM)
    xy_delta = block_center[:2] - target_center[:2]
    floor_top = floor_center[2] + floor_size[2]
    block_bottom = block_center[2] - block_size[2]
    inner_half = target_size[:2] - TARGET_WALL_THICKNESS - block_size[:2]
    block_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, BLOCK_JOINT)
    velocity = data.qvel[model.jnt_dofadr[block_joint] : model.jnt_dofadr[block_joint] + 3]
    inside = bool(np.all(np.abs(xy_delta) <= inner_half))
    clearance = float(block_bottom - floor_top)
    speed = float(np.linalg.norm(velocity))
    return {
        "block_position": block_center,
        "target_position": target_center,
        "xy_distance": float(np.linalg.norm(xy_delta)),
        "clearance": clearance,
        "linear_speed": speed,
        "inside_target": inside,
        "success": inside and abs(clearance) <= 0.008 and speed < 0.08,
    }


def render_once(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str) -> None:
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera_name)
        image = renderer.render()
    finally:
        renderer.close()
    print(f"[ok] offscreen render ({camera_name}): shape={image.shape} dtype={image.dtype}")


def print_summary(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    status = scene_status(model, data)
    print(f"[ok] model: bodies={model.nbody} joints={model.njnt} actuators={model.nu}")
    print(f"[ok] block center: {np.round(status['block_position'], 4).tolist()}")
    print(f"[ok] target center: {np.round(status['target_position'], 4).tolist()}")
    print(f"[ok] block->target xy distance: {status['xy_distance']:.4f} m")
    print(f"[ok] block clearance over target floor: {status['clearance']:.4f} m")
    print(f"[ok] block inside blue trapezoid container: {status['success']}")


def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.86, 0.0, 0.58]
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -32
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            delay = model.opt.timestep - (time.time() - start)
            if delay > 0:
                time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--camera", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless-steps", type=int)
    parser.add_argument("--render-check", action="store_true")
    parser.add_argument("--goal-state", action="store_true", help="Place the red block inside the blue container.")
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the blue tray body position in meters.",
    )
    parser.add_argument(
        "--block-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Use an exact red block position instead of randomization.",
    )
    args = parser.parse_args()

    model, data = load_scene(args.xml.expanduser().resolve())
    if args.target_position is not None:
        set_target_pose(model, data, np.asarray(args.target_position))
    if args.goal_state:
        if args.block_position is not None:
            raise ValueError("--goal-state and --block-position are mutually exclusive")
        place_block_in_target(model, data)
    elif args.block_position is not None:
        set_block_pose(model, data, np.asarray(args.block_position))
    else:
        randomize_block(model, data, args.seed)
    if args.headless_steps is not None:
        if args.headless_steps <= 0:
            raise ValueError("--headless-steps must be positive")
        for _ in range(args.headless_steps):
            mujoco.mj_step(model, data)
    if args.render_check:
        render_once(model, data, args.camera)
    print_summary(model, data)
    if args.headless_steps is None and not args.render_check:
        launch_viewer(model, data)


if __name__ == "__main__":
    main()
