"""Run a geometry-driven BRX grasp and place episode for pipeline validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "simulation" / "scripts"))

from interface_sim import SimCamera, SimRobotArm, SimWorld  # noqa: E402
from run_block_jamming_env import (  # noqa: E402
    BLOCK_GEOM,
    DEFAULT_XML_PATH,
    geom_center_and_size,
    randomize_block,
    set_block_pose,
    set_target_pose,
    scene_status,
)


def _target_in_camera(camera: SimCamera, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    return camera.pose_world_to_camera(rotation, position)


def _move(
    arm: SimRobotArm,
    world: SimWorld,
    camera: SimCamera,
    rotation: np.ndarray,
    position: np.ndarray,
    label: str,
    duration: float = 1.8,
) -> bool:
    target = _target_in_camera(camera, rotation, position)
    ok = arm.move_ee_in_cam(target, duration=duration, blocking=True)
    print(
        f"[oracle] {label}: ok={ok} target={np.round(position, 4).tolist()} "
        f"achieved={np.round(world.data.site_xpos[arm.site_id], 4).tolist()} "
        f"pos_err={arm.last_ik_position_error:.4f}m rot_err={arm.last_ik_rotation_error_deg:.1f}deg",
        flush=True,
    )
    return ok


def _print_grasp_geometry(world: SimWorld, label: str) -> None:
    geom_ids = [
        mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("JawBlock01_Link_collision", "JawBlock02_Link_collision")
    ]
    centers = np.array([world.data.geom_xpos[index] for index in geom_ids])
    print(
        f"[oracle] {label}: finger_centers={np.round(centers, 4).tolist()} "
        f"distance={np.linalg.norm(centers[0] - centers[1]):.4f}m",
        flush=True,
    )
    block_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, BLOCK_GEOM)
    contacts = []
    for index in range(world.data.ncon):
        contact = world.data.contact[index]
        if block_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == block_id else contact.geom1
        contacts.append(
            {
                "other": mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_GEOM, other),
                "distance": round(float(contact.dist), 5),
                "normal": np.round(contact.frame[:3], 3).tolist(),
            }
        )
    print(f"[oracle] {label}: block_contacts={contacts}", flush=True)


def _activate_grasp_weld(world: SimWorld) -> None:
    weld_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_EQUALITY, "oracle_grasp_weld")
    block_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "block")
    wrist_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "ArmR08_Link")
    if min(weld_id, block_id, wrist_id) < 0:
        raise ValueError("oracle_grasp_weld requires block and ArmR08_Link bodies")
    block_rotation = world.data.xmat[block_id].reshape(3, 3)
    wrist_rotation = world.data.xmat[wrist_id].reshape(3, 3)
    relative_position = block_rotation.T @ (world.data.xpos[wrist_id] - world.data.xpos[block_id])
    relative_rotation = block_rotation.T @ wrist_rotation
    relative_quaternion = Rotation.from_matrix(relative_rotation).as_quat()
    world.model.eq_data[weld_id, 3:6] = relative_position
    world.model.eq_data[weld_id, 6:10] = [
        relative_quaternion[3],
        relative_quaternion[0],
        relative_quaternion[1],
        relative_quaternion[2],
    ]
    world.data.eq_active[weld_id] = 1
    mujoco.mj_forward(world.model, world.data)
    print("[oracle] grasp_mode=explicit_weld (simulation approximation)", flush=True)


def _deactivate_grasp_weld(world: SimWorld) -> None:
    weld_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_EQUALITY, "oracle_grasp_weld")
    if weld_id >= 0:
        world.data.eq_active[weld_id] = 0
        mujoco.mj_forward(world.model, world.data)


def run(
    xml_path: Path,
    seed: int,
    grasp_mode: str,
    block_position: np.ndarray | None = None,
    target_position: np.ndarray | None = None,
) -> dict[str, object]:
    world = SimWorld(xml_path, hide_robot_visuals=False)
    camera = SimCamera(world)
    arm = SimRobotArm(
        world,
        camera,
        "right",
        ik_position_tolerance=0.05,
        ik_rotation_tolerance_deg=10.0,
        interpolation_steps=12,
    )
    try:
        if target_position is not None:
            set_target_pose(world.model, world.data, target_position)
        if block_position is None:
            randomize_block(world.model, world.data, seed)
        else:
            set_block_pose(world.model, world.data, block_position)
        block_position, block_size = geom_center_and_size(world.model, world.data, BLOCK_GEOM)
        arm.go_home()
        arm.set_gripper(0.0, blocking=True)
        current_rotation = world.data.site_xmat[arm.site_id].reshape(3, 3).copy()
        # The vendor jaw slide axis is tilted in the home wrist frame; this roll
        # makes the two contact pads horizontal before the vertical approach.
        wrist_roll = Rotation.from_rotvec(np.array([0.52, 0.0, 0.0])).as_matrix()
        current_rotation = current_rotation @ wrist_roll
        collision_ids = [
            mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("JawBlock01_Link_collision", "JawBlock02_Link_collision")
        ]
        midpoint = np.mean([world.data.geom_xpos[index] for index in collision_ids], axis=0)
        site_offset = midpoint - world.data.site_xpos[arm.site_id]
        grasp_site_position = block_position - site_offset
        print(f"[oracle] block={np.round(block_position, 4).tolist()} half_size={block_size.tolist()}")
        print(f"[oracle] site_offset_to_finger_midpoint={np.round(site_offset, 4).tolist()}")

        if not _move(arm, world, camera, current_rotation, grasp_site_position + [0.0, 0.0, 0.07], "approach", 1.2):
            return {"success": False, "reason": "approach_ik", "ik_failures": arm.ik_failure_count}
        if not _move(arm, world, camera, current_rotation, grasp_site_position + [0.0, 0.0, 0.012], "descend", 1.2):
            return {"success": False, "reason": "descend_ik", "ik_failures": arm.ik_failure_count}
        _print_grasp_geometry(world, "before_close")
        # Leave a small preload so both 14 mm-thick finger pads contact the 30 mm block.
        arm.set_gripper(0.026 / arm.gripper_max, blocking=True)
        print(f"[oracle] close: gripper={arm.get_gripper():.3f}")
        _print_grasp_geometry(world, "after_close")
        if grasp_mode == "weld":
            _activate_grasp_weld(world)
        world.step(500)
        held_position = world.data.geom_xpos[
            mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, BLOCK_GEOM)
        ].copy()
        if not _move(arm, world, camera, current_rotation, grasp_site_position + [0.0, 0.0, 0.18], "lift", 2.0):
            return {"success": False, "reason": "lift_ik", "ik_failures": arm.ik_failure_count}
        lifted_position = world.data.geom_xpos[
            mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, BLOCK_GEOM)
        ].copy()
        target_floor_center, target_floor_size = geom_center_and_size(
            world.model, world.data, "target_container_floor"
        )
        target = np.array(
            [target_floor_center[0], target_floor_center[1], target_floor_center[2] + target_floor_size[2] + 0.045],
            dtype=float,
        )
        _move(arm, world, camera, current_rotation, target + [0.0, 0.0, 0.12], "place_approach", 2.0)
        _move(arm, world, camera, current_rotation, target + [0.0, 0.0, 0.045], "place", 2.0)
        arm.set_gripper(0.0, blocking=True)
        if grasp_mode == "weld":
            _deactivate_grasp_weld(world)
        for _ in range(50):
            world.step(10)
        status = scene_status(world.model, world.data)
        return {
            "success": bool(status["success"]),
            "reason": "placed" if status["success"] else "not_placed",
            "held_position": held_position.tolist(),
            "lifted_position": lifted_position.tolist(),
            "lift_delta_z": float(lifted_position[2] - held_position[2]),
            "final_scene": {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in status.items()},
            "ik_failures": arm.ik_failure_count,
        }
    finally:
        arm.close()
        camera.close()
        world.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--grasp-mode",
        choices=("weld", "physical"),
        default="physical",
        help="Use real contact dynamics or the explicit simulation-only weld fallback.",
    )
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
    result = run(
        args.xml.expanduser().resolve(),
        args.seed,
        args.grasp_mode,
        block_position=np.asarray(args.block_position) if args.block_position is not None else None,
        target_position=np.asarray(args.target_position) if args.target_position is not None else None,
    )
    print(f"[oracle] result={result}")


if __name__ == "__main__":
    main()
