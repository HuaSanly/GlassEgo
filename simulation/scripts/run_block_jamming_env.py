"""Run the GlassEgo Block Jamming MuJoCo scene."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = SIMULATION_ROOT / "tasks" / "block_jamming_wxai.xml"

BLOCK_GEOM = "block_geom"
BLOCK_JOINT = "block_freejoint"
DEFAULT_CAMERA_NAME = "ego_rgbd"
TARGET_GEOM = "target_box_geom"


def object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id_value = mujoco.mj_name2id(model, object_type, name)
    if object_id_value < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return object_id_value


def load_scene(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Block Jamming XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def geom_center_and_size(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    geom_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    return data.geom_xpos[geom_id].copy(), model.geom_size[geom_id].copy()


def place_block_on_target(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    _, block_size = geom_center_and_size(model, data, BLOCK_GEOM)
    target_center, target_size = geom_center_and_size(model, data, TARGET_GEOM)
    target_top_z = target_center[2] + target_size[2]

    joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, BLOCK_JOINT)
    qpos_address = model.jnt_qposadr[joint_id]
    qvel_address = model.jnt_dofadr[joint_id]

    data.qpos[qpos_address : qpos_address + 3] = [
        target_center[0],
        target_center[1],
        target_top_z + block_size[2],
    ]
    data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[qvel_address : qvel_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def step_scene(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    if steps <= 0:
        raise ValueError("--headless-steps must be positive")

    for _ in range(steps):
        mujoco.mj_step(model, data)


def render_once(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str) -> None:
    renderer = mujoco.Renderer(model, height=360, width=480)
    try:
        renderer.update_scene(data, camera=camera_name)
        image = renderer.render()
    finally:
        renderer.close()

    print(f"[ok] offscreen render ({camera_name}): shape={image.shape} dtype={image.dtype}")


def print_summary(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    block_center, block_size = geom_center_and_size(model, data, BLOCK_GEOM)
    target_center, target_size = geom_center_and_size(model, data, TARGET_GEOM)
    xy_delta = block_center[:2] - target_center[:2]
    xy_distance = float(np.linalg.norm(xy_delta))
    target_top_z = float(target_center[2] + target_size[2])
    block_bottom_z = float(block_center[2] - block_size[2])
    is_over_target = bool(np.all(np.abs(xy_delta) <= target_size[:2]))
    is_above_target = block_bottom_z >= target_top_z - 0.005

    print(f"[ok] model: bodies={model.nbody} joints={model.njnt} actuators={model.nu}")
    print(f"[ok] block center: {block_center.round(4).tolist()}")
    print(f"[ok] target center: {target_center.round(4).tolist()}")
    print(f"[ok] block->target xy distance: {xy_distance:.4f} m")
    print(f"[ok] target top z: {target_top_z:.4f} m")
    print(f"[ok] block bottom z: {block_bottom_z:.4f} m")
    print(f"[ok] block over yellow box: {is_over_target and is_above_target}")


def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.12]
        viewer.cam.distance = 0.95
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -28

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--camera", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--headless-steps", type=int, default=None)
    parser.add_argument("--render-check", action="store_true")
    parser.add_argument(
        "--goal-state",
        action="store_true",
        help="Place the block on the yellow box before running, for visual target-state checks.",
    )
    args = parser.parse_args()

    model, data = load_scene(args.xml.expanduser().resolve())
    if args.goal_state:
        place_block_on_target(model, data)

    if args.headless_steps is not None:
        step_scene(model, data, args.headless_steps)

    if args.render_check:
        render_once(model, data, args.camera)

    print_summary(model, data)

    if args.headless_steps is None and not args.render_check:
        launch_viewer(model, data)


if __name__ == "__main__":
    main()
