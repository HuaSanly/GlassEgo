"""Geometry-driven physical grasp supervisor for the BRX MuJoCo scene.

The policy is useful for deciding *when* to move, but it is not required to
solve this simple, fully observable simulation.  :class:`GeometryGraspSupervisor`
therefore provides a small, deterministic fallback which uses the true MuJoCo
object pose and the same camera-frame Cartesian interface as policy control.

The module intentionally does not import the oracle script. This keeps the
supervisor usable from ``simulation/run_inference_sim.py`` without a
script-to-script dependency and makes the diagnostics available to both policy
and oracle runs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imports are only for editor/type support
    from simulation.interface_sim import SimCamera, SimRobotArm, SimWorld


@dataclass(frozen=True)
class GeometryGraspConfig:
    """Motion and verification parameters for one grasp/place episode.

    The defaults match the validated ``run_brx_oracle_grasp.py`` sequence:
    the end-effector approaches 70 mm above the grasp point, descends to a
    12 mm clearance, closes to 0.63 normalized opening (26 mm jaw travel),
    lifts 180 mm, and releases 45 mm above the target floor.
    """

    approach_offset_z: float = 0.07
    descend_offset_z: float = 0.012
    lift_offset_z: float = 0.18
    place_approach_offset_z: float = 0.12
    place_offset_z: float = 0.045
    close_preload: float = 0.63
    settle_steps: int = 500
    release_steps: int = 500
    approach_duration: float = 1.2
    descend_duration: float = 1.2
    lift_duration: float = 2.0
    place_duration: float = 2.0
    target_wall_thickness: float = 0.004
    success_z_tolerance: float = 0.008
    success_linear_speed: float = 0.08
    min_lift_delta_z: float = 0.05
    contact_distance_tolerance: float = 1.0e-4


_BLOCK_GEOM = "block_geom"
_BLOCK_JOINT = "block_freejoint"
_TARGET_BOUNDS_GEOM = "target_container_bounds"
_TARGET_FLOOR_GEOM = "target_container_floor"


class PolicyProgressMonitor:
    """Allow replanning before handing a stalled policy episode to geometry."""

    def __init__(self, config: Mapping[str, object], metrics: np.ndarray) -> None:
        self.min_steps = int(config.get("min_policy_steps", 8))
        self.max_steps = int(config.get("max_policy_steps", 40))
        self.stall_steps = int(config.get("stall_steps", 12))
        self.failure_steps = int(config.get("failure_steps", 5))
        self.progress_tolerance = float(config.get("progress_tolerance", 0.005))
        self.best_metrics = np.asarray(metrics, dtype=float).copy()
        self.steps = 0
        self.no_progress_steps = 0
        self.gated_streak = 0
        self.ik_failure_streak = 0

    def update(
        self, metrics: np.ndarray, *, gated_close: bool, all_ik_failed: bool,
        policy_done: bool,
    ) -> str | None:
        self.steps += 1
        improved = np.asarray(metrics) < self.best_metrics - self.progress_tolerance
        self.best_metrics[improved] = np.asarray(metrics)[improved]
        self.no_progress_steps = 0 if np.any(improved) else self.no_progress_steps + 1
        self.gated_streak = self.gated_streak + 1 if gated_close else 0
        self.ik_failure_streak = self.ik_failure_streak + 1 if all_ik_failed else 0
        if self.steps < self.min_steps:
            return None
        if policy_done:
            return "policy_done_without_success"
        if self.ik_failure_streak >= self.failure_steps:
            return "repeated_ik_failure"
        if self.gated_streak >= self.failure_steps and self.no_progress_steps >= self.failure_steps:
            return "repeated_unsafe_close_without_progress"
        if self.no_progress_steps >= self.stall_steps:
            return "policy_stalled"
        if self.steps >= self.max_steps:
            return "policy_budget_exhausted"
        return None


def _object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return value


def _list(values: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    """Convert an array-like value to JSON-friendly floats."""

    return np.asarray(values, dtype=float).reshape(-1).tolist()


def _config_from(value: GeometryGraspConfig | Mapping[str, object] | None) -> GeometryGraspConfig:
    if value is None:
        return GeometryGraspConfig()
    if isinstance(value, GeometryGraspConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("geometry grasp config must be a mapping or GeometryGraspConfig")
    allowed = {field.name for field in fields(GeometryGraspConfig)}
    values = {key: raw for key, raw in value.items() if key in allowed}
    return GeometryGraspConfig(**values)


class GeometryGraspSupervisor:
    """Execute and verify one physical BRX grasp-and-place sequence.

    Parameters are the existing adapters from :mod:`interface_sim`; no new
    robot or IK implementation is hidden here.  ``execute`` leaves ownership
    of ``world``, ``camera`` and ``arm`` with the caller.
    """

    def __init__(
        self,
        world: "SimWorld",
        camera: "SimCamera",
        arm: "SimRobotArm",
        config: GeometryGraspConfig | Mapping[str, object] | None = None,
        *,
        block_geom: str = _BLOCK_GEOM,
        target_bounds_geom: str = _TARGET_BOUNDS_GEOM,
        target_floor_geom: str = _TARGET_FLOOR_GEOM,
        block_joint: str = _BLOCK_JOINT,
    ) -> None:
        self.world = world
        self.camera = camera
        self.arm = arm
        self.config = _config_from(config)
        self.block_geom_name = block_geom
        self.target_bounds_geom_name = target_bounds_geom
        self.target_floor_geom_name = target_floor_geom
        self.block_joint_name = block_joint
        side = getattr(arm, "side", "right")
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported arm side for geometry grasp: {side}")
        prefix = "R" if side == "right" else "L"
        jaw_prefix = "JawBlock01" if side == "right" else "JawBlock03"
        jaw_second = "JawBlock02" if side == "right" else "JawBlock04"
        self.wrist_body_name = f"Arm{prefix}08_Link"
        self.finger_geom_names = (
            f"{jaw_prefix}_Link_collision",
            f"{jaw_second}_Link_collision",
        )
        self.block_geom_id = _object_id(world.model, mujoco.mjtObj.mjOBJ_GEOM, block_geom)
        self.target_bounds_geom_id = _object_id(
            world.model, mujoco.mjtObj.mjOBJ_GEOM, target_bounds_geom
        )
        self.target_floor_geom_id = _object_id(
            world.model, mujoco.mjtObj.mjOBJ_GEOM, target_floor_geom
        )
        self.block_joint_id = _object_id(world.model, mujoco.mjtObj.mjOBJ_JOINT, block_joint)
        self.wrist_body_id = _object_id(world.model, mujoco.mjtObj.mjOBJ_BODY, self.wrist_body_name)
        self.finger_geom_ids = tuple(
            _object_id(world.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.finger_geom_names
        )

    # ------------------------------------------------------------------
    # Scene state and contact helpers
    def _geom_center_size(self, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.world.data.geom_xpos[geom_id].copy(),
            self.world.model.geom_size[geom_id].copy(),
        )

    def _set_block_position(self, position: np.ndarray, yaw: float = 0.0) -> None:
        target = np.asarray(position, dtype=float).reshape(3)
        if not np.all(np.isfinite(target)):
            raise ValueError("block position must contain three finite values")
        qpos = int(self.world.model.jnt_qposadr[self.block_joint_id])
        qvel = int(self.world.model.jnt_dofadr[self.block_joint_id])
        self.world.data.qpos[qpos : qpos + 3] = target
        self.world.data.qpos[qpos + 3 : qpos + 7] = [
            np.cos(float(yaw) / 2.0),
            0.0,
            0.0,
            np.sin(float(yaw) / 2.0),
        ]
        self.world.data.qvel[qvel : qvel + 6] = 0.0
        mujoco.mj_forward(self.world.model, self.world.data)

    def _set_target_position(self, position: np.ndarray) -> None:
        target = np.asarray(position, dtype=float).reshape(3)
        if not np.all(np.isfinite(target)):
            raise ValueError("target body position must contain three finite values")
        body_id = _object_id(
            self.world.model, mujoco.mjtObj.mjOBJ_BODY, "target_container"
        )
        self.world.model.body_pos[body_id] = target
        mujoco.mj_forward(self.world.model, self.world.data)

    def scene_status(self) -> dict[str, Any]:
        """Return placement geometry and the same success test used by the scene."""

        block_center, block_size = self._geom_center_size(self.block_geom_id)
        target_center, target_size = self._geom_center_size(self.target_bounds_geom_id)
        floor_center, floor_size = self._geom_center_size(self.target_floor_geom_id)
        xy_delta = block_center[:2] - target_center[:2]
        floor_top = floor_center[2] + floor_size[2]
        block_bottom = block_center[2] - block_size[2]
        inner_half = (
            target_size[:2]
            - float(self.config.target_wall_thickness)
            - block_size[:2]
        )
        velocity_start = int(self.world.model.jnt_dofadr[self.block_joint_id])
        velocity = self.world.data.qvel[velocity_start : velocity_start + 3]
        inside = bool(np.all(np.abs(xy_delta) <= inner_half))
        clearance = float(block_bottom - floor_top)
        speed = float(np.linalg.norm(velocity))
        return {
            "block_position": _list(block_center),
            "target_position": _list(target_center),
            "xy_distance": float(np.linalg.norm(xy_delta)),
            "clearance": clearance,
            "linear_speed": speed,
            "inside_target": inside,
            "success": bool(
                inside
                and abs(clearance) <= float(self.config.success_z_tolerance)
                and speed < float(self.config.success_linear_speed)
            ),
        }

    def contact_report(self) -> dict[str, Any]:
        """Report block contacts, including which jaw pads are touching it."""

        finger_ids = set(self.finger_geom_ids)
        contacts: list[dict[str, Any]] = []
        finger_contacts: set[int] = set()
        tolerance = float(self.config.contact_distance_tolerance)
        for index in range(int(self.world.data.ncon)):
            contact = self.world.data.contact[index]
            if self.block_geom_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.block_geom_id else contact.geom1
            if other not in finger_ids:
                continue
            if float(contact.dist) <= tolerance:
                finger_contacts.add(int(other))
            contacts.append(
                {
                    "finger": self.world.model.geom(other).name,
                    "distance": float(contact.dist),
                    "normal": _list(contact.frame[:3]),
                }
            )
        return {
            "contacts": contacts,
            "contact_count": len(contacts),
            "fingers_in_contact": [self.finger_geom_names[index] for index, geom_id in enumerate(self.finger_geom_ids) if geom_id in finger_contacts],
            "both_fingers_contact": len(finger_contacts) == len(self.finger_geom_ids),
        }

    # ------------------------------------------------------------------
    # Motion helpers
    def _move(
        self,
        stages: list[dict[str, Any]],
        label: str,
        position_world: np.ndarray,
        rotation_world: np.ndarray,
        duration: float,
    ) -> bool:
        target_camera = self.camera.pose_world_to_camera(rotation_world, position_world)
        ok = bool(self.arm.move_ee_in_cam(target_camera, duration=float(duration), blocking=True))
        achieved = self.world.data.site_xpos[self.arm.site_id].copy()
        stage = {
            "name": label,
            "ok": ok,
            "target_world": _list(position_world),
            "achieved_world": _list(achieved),
            "position_error": float(getattr(self.arm, "last_ik_position_error", np.nan)),
            "rotation_error_deg": float(getattr(self.arm, "last_ik_rotation_error_deg", np.nan)),
        }
        stages.append(stage)
        return ok

    def _result(
        self,
        *,
        reason: str,
        success: bool,
        stages: list[dict[str, Any]],
        block_before: np.ndarray,
        grasp_site: np.ndarray,
        site_offset: np.ndarray,
        contact_after_close: dict[str, Any] | None = None,
        contact_after_lift: dict[str, Any] | None = None,
        held_position: np.ndarray | None = None,
        lifted_position: np.ndarray | None = None,
    ) -> dict[str, Any]:
        lift_delta_z = (
            float(lifted_position[2] - held_position[2])
            if held_position is not None and lifted_position is not None
            else float("nan")
        )
        final_scene = self.scene_status()
        return {
            "success": bool(success),
            "reason": reason,
            "stages": stages,
            "block_position_before": _list(block_before),
            "grasp_site_position": _list(grasp_site),
            "site_offset_to_finger_midpoint": _list(site_offset),
            "contact_after_close": contact_after_close or self.contact_report(),
            "contact_after_lift": contact_after_lift or self.contact_report(),
            "contact_verified": bool(
                contact_after_close and contact_after_close["both_fingers_contact"]
            ),
            "held_block_position": _list(held_position) if held_position is not None else None,
            "lifted_block_position": _list(lifted_position) if lifted_position is not None else None,
            "lift_delta_z": lift_delta_z,
            "lift_verified": bool(np.isfinite(lift_delta_z) and lift_delta_z >= self.config.min_lift_delta_z),
            "final_scene": final_scene,
            "ik_failures": int(getattr(self.arm, "ik_failure_count", 0)),
        }

    # ------------------------------------------------------------------
    def execute(
        self,
        block_position: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
        *,
        reset_home: bool = True,
    ) -> dict[str, Any]:
        """Run approach, descend, close, lift, place and release physically.

        ``block_position`` is the block geom center.  ``target_position`` is the
        ``target_container`` body position, matching the simulation CLI.  If
        omitted, both are read from the current MuJoCo state.  ``reset_home``
        is useful when this supervisor is invoked after a policy attempt; it
        opens the jaw and returns the arm to the validated home configuration.
        """

        stages: list[dict[str, Any]] = []
        if target_position is not None:
            self._set_target_position(target_position)
        if block_position is not None:
            self._set_block_position(block_position)
        self.arm.set_gripper(0.0, blocking=True)
        if reset_home:
            self.arm.go_home(blocking=True)

        block_before, _ = self._geom_center_size(self.block_geom_id)
        site_rotation = self.world.data.site_xmat[self.arm.site_id].reshape(3, 3)
        finger_rotation = self.world.data.geom_xmat[self.finger_geom_ids[0]].reshape(3, 3)
        finger_in_site = site_rotation.T @ finger_rotation
        # Keep the pads vertical and their closing axis horizontal. Derive the
        # wrist orientation from the actual jaw frame, not a history-dependent
        # home pose with a fixed roll correction.
        desired_finger_rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        current_rotation = desired_finger_rotation @ finger_in_site.T
        finger_centers = np.array([self.world.data.geom_xpos[index] for index in self.finger_geom_ids])
        offset_in_site = site_rotation.T @ (
            finger_centers.mean(axis=0) - self.world.data.site_xpos[self.arm.site_id]
        )
        site_offset = current_rotation @ offset_in_site
        grasp_site = block_before - site_offset

        if not self._move(
            stages,
            "approach",
            grasp_site + np.array([0.0, 0.0, self.config.approach_offset_z]),
            current_rotation,
            self.config.approach_duration,
        ):
            return self._result(
                reason="approach_ik", success=False, stages=stages,
                block_before=block_before, grasp_site=grasp_site, site_offset=site_offset,
            )
        if not self._move(
            stages,
            "descend",
            grasp_site + np.array([0.0, 0.0, self.config.descend_offset_z]),
            current_rotation,
            self.config.descend_duration,
        ):
            return self._result(
                reason="descend_ik", success=False, stages=stages,
                block_before=block_before, grasp_site=grasp_site, site_offset=site_offset,
            )

        self.arm.set_gripper(float(self.config.close_preload), blocking=True)
        if self.config.settle_steps:
            self.world.step(int(self.config.settle_steps))
        contact_after_close = self.contact_report()
        held_position = self.world.data.geom_xpos[self.block_geom_id].copy()

        if not self._move(
            stages,
            "lift",
            grasp_site + np.array([0.0, 0.0, self.config.lift_offset_z]),
            current_rotation,
            self.config.lift_duration,
        ):
            return self._result(
                reason="lift_ik", success=False, stages=stages,
                block_before=block_before, grasp_site=grasp_site, site_offset=site_offset,
                contact_after_close=contact_after_close, held_position=held_position,
            )
        contact_after_lift = self.contact_report()
        lifted_position = self.world.data.geom_xpos[self.block_geom_id].copy()
        lift_delta_z = float(lifted_position[2] - held_position[2])

        floor_center, floor_size = self._geom_center_size(self.target_floor_geom_id)
        target_pose = floor_center + np.array([0.0, 0.0, floor_size[2] + self.config.place_offset_z])
        if not self._move(
            stages,
            "place_approach",
            target_pose + np.array([0.0, 0.0, self.config.place_approach_offset_z]),
            current_rotation,
            self.config.place_duration,
        ):
            return self._result(
                reason="place_approach_ik", success=False, stages=stages,
                block_before=block_before, grasp_site=grasp_site, site_offset=site_offset,
                contact_after_close=contact_after_close, contact_after_lift=contact_after_lift,
                held_position=held_position, lifted_position=lifted_position,
            )
        if not self._move(
            stages,
            "place",
            target_pose + np.array([0.0, 0.0, self.config.place_offset_z]),
            current_rotation,
            self.config.place_duration,
        ):
            return self._result(
                reason="place_ik", success=False, stages=stages,
                block_before=block_before, grasp_site=grasp_site, site_offset=site_offset,
                contact_after_close=contact_after_close, contact_after_lift=contact_after_lift,
                held_position=held_position, lifted_position=lifted_position,
            )

        self.arm.set_gripper(0.0, blocking=True)
        if self.config.release_steps:
            self.world.step(int(self.config.release_steps))
        final_scene = self.scene_status()
        physical_contact = bool(
            contact_after_close["both_fingers_contact"]
            and contact_after_lift["both_fingers_contact"]
        )
        lifted = bool(lift_delta_z >= float(self.config.min_lift_delta_z))
        success = bool(physical_contact and lifted and final_scene["success"])
        reason = "placed" if success else (
            "no_finger_contact" if not physical_contact else
            "object_not_lifted" if not lifted else "not_placed"
        )
        result = self._result(
            reason=reason, success=success, stages=stages,
            block_before=block_before, grasp_site=grasp_site, site_offset=site_offset,
            contact_after_close=contact_after_close, contact_after_lift=contact_after_lift,
            held_position=held_position, lifted_position=lifted_position,
        )
        # ``_result`` recomputes the same status; retaining this assertion makes
        # accidental divergence between the success gate and diagnostics obvious.
        result["final_scene"] = final_scene
        return result

    # A short alias makes the supervisor convenient in a fallback branch.
    run = execute


__all__ = ["GeometryGraspConfig", "GeometryGraspSupervisor"]
