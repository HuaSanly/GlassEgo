"""MuJoCo adapters for the hardware-agnostic inference interfaces.

The simulation adapter deliberately keeps perception deterministic: object poses
come from MuJoCo state while the rendered image follows the training-time
``rgb_WoArm_WArmObjKpts`` convention.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from simulation.interfaces import Camera, Frame, ObjectState, Perception, RobotArm


CV_FROM_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
VIEWER_SHUTDOWN_DELAY = 0.5
VIEWER_FRAME_INTERVAL = 1.0 / 60.0


class ViewerClosedError(Exception):
    """Stop an episode when the user closes the interactive viewer."""


def _nvidia_glx_available() -> bool:
    """Return whether the NVIDIA GLX client library is loadable."""
    library = ctypes.util.find_library("GLX_nvidia")
    if library is None:
        library = "libGLX_nvidia.so.0"
    try:
        ctypes.CDLL(library)
    except OSError:
        return False
    return True


def _configure_viewer_glx() -> None:
    """Select the NVIDIA GLX vendor when a desktop NVIDIA stack is present."""
    if sys.platform == "darwin" or not os.environ.get("DISPLAY"):
        return
    if not os.environ.get("__GLX_VENDOR_LIBRARY_NAME") and _nvidia_glx_available():
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"


def _named(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(index)


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return result


def _interpolate_pose(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate a pose with linear translation and spherical rotation."""
    fraction = float(fraction)
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]])),
    )([fraction]).as_matrix()[0]
    result[:3, 3] = (
        (1.0 - fraction) * start[:3, 3] + fraction * target[:3, 3]
    )
    return result


class SimWorld:
    """Own the MuJoCo model/data and the simulation clock."""

    def __init__(
        self,
        xml_path: str | Path,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        hide_robot_visuals: bool = True,
    ) -> None:
        self.xml_path = Path(xml_path).expanduser().resolve()
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"Simulation XML not found: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.width, self.height = int(width), int(height)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.scene_option = mujoco.MjvOption()
        if hide_robot_visuals:
            # The task XML puts arm meshes in geom group 2; task geometry remains visible.
            self.scene_option.geomgroup[2] = 0
            self.scene_option.geomgroup[3] = 0
        self.viewer = None
        self.step_count = 0
        self.reset()

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

    def step(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        started = time.perf_counter()
        frame_steps = max(1, round(VIEWER_FRAME_INTERVAL / self.model.opt.timestep))
        for index in range(int(steps)):
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            if self.viewer is not None and (
                (index + 1) % frame_steps == 0 or index + 1 == int(steps)
            ):
                if not self.sync_viewer():
                    raise ViewerClosedError()
                remaining = (index + 1) * self.model.opt.timestep - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

    def advance(self, duration: float) -> None:
        """Advance all scheduled arm and gripper controls on the same clock."""
        self.step(max(1, int(round(duration / self.model.opt.timestep))))

    def render(self, camera_name: str, depth: bool = False) -> np.ndarray:
        self.renderer.enable_depth_rendering() if depth else self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera_name, scene_option=self.scene_option)
        return np.asarray(self.renderer.render()).copy()

    def launch_viewer(self) -> None:
        _configure_viewer_glx()
        import mujoco.viewer

        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo viewer could not start. Use --headless for EGL/offscreen "
                "runs, or fix the X11/GLX driver for interactive viewing."
            ) from exc
        self.viewer.cam.lookat[:] = [0.45, -0.08, 0.65]
        self.viewer.cam.distance = 1.7
        self.viewer.cam.azimuth = 130
        self.viewer.cam.elevation = -28
        self.viewer.sync()

    def sync_viewer(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        self.viewer.sync()
        return True

    def close(self) -> None:
        if self.viewer is not None:
            viewer = self.viewer
            viewer.close()
            # launch_passive owns a daemon UI thread; give its native renderer
            # time to leave GLFW before destroying the offscreen context.
            time.sleep(VIEWER_SHUTDOWN_DELAY)
            self.viewer = None
        self.renderer.close()


class SimCamera(Camera):
    """Ego RGB-D camera with MuJoCo-to-OpenCV optical-frame conversion."""

    def __init__(self, world: SimWorld, name: str = "ego_rgbd") -> None:
        self.world = world
        self.name = name
        self.camera_id = _named(world.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        fy = 0.5 * world.height / math.tan(math.radians(float(world.model.cam_fovy[self.camera_id])) / 2.0)
        # MuJoCo fovy uses square pixels: aspect changes horizontal FOV, not fx.
        fx = fy
        self.K = np.array(
            [[fx, 0.0, (world.width - 1) / 2.0], [0.0, fy, (world.height - 1) / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def get_frame(self) -> Frame:
        rgb_rgb = self.world.render(self.name)
        depth = self.world.render(self.name, depth=True).astype(np.float32)
        return Frame(rgb=cv2.cvtColor(rgb_rgb, cv2.COLOR_RGB2BGR), depth_m=depth, K=self.K.copy())

    def world_to_camera(self, point_world: np.ndarray) -> np.ndarray:
        point_world = np.asarray(point_world, dtype=np.float32).reshape(-1, 3)
        origin = self.world.data.cam_xpos[self.camera_id]
        world_to_mj = self.world.data.cam_xmat[self.camera_id].reshape(3, 3).T
        point_mj = (world_to_mj @ (point_world - origin).T).T
        return (CV_FROM_MUJOCO_CAMERA @ point_mj.T).T

    def pose_world_to_camera(self, rotation_world: np.ndarray, position_world: np.ndarray) -> np.ndarray:
        cam_rotation = CV_FROM_MUJOCO_CAMERA @ self.world.data.cam_xmat[self.camera_id].reshape(3, 3).T
        return _pose(cam_rotation @ np.asarray(rotation_world), self.world_to_camera(position_world)[0])

    def close(self) -> None:
        return None


class SimRobotArm(RobotArm):
    """Position-controlled MuJoCo arm using damped least-squares site IK."""

    def __init__(
        self,
        world: SimWorld,
        camera: SimCamera,
        side: str,
        ik_damping: float = 0.04,
        ik_iterations: int = 40,
        ik_rotation_weight: float = 0.35,
        ik_position_tolerance: float = 0.015,
        ik_rotation_tolerance_deg: float = 8.0,
        interpolation_steps: int = 4,
        gripper_max: float = 0.041,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported arm side: {side}")
        self.world, self.camera, self.side = world, camera, side
        self.ik_damping, self.ik_iterations = float(ik_damping), int(ik_iterations)
        self.ik_rotation_weight = float(ik_rotation_weight)
        self.ik_position_tolerance = float(ik_position_tolerance)
        self.ik_rotation_tolerance_deg = float(ik_rotation_tolerance_deg)
        self.interpolation_steps = max(1, int(interpolation_steps))
        self.gripper_max = float(gripper_max)
        prefix = f"{side}_"
        self.site_id = _named(world.model, mujoco.mjtObj.mjOBJ_SITE, prefix + "ee_site")
        arm_prefix = "R" if side == "right" else "L"
        arm_joint_names = [f"Arm{arm_prefix}0{i}_Joint" for i in range(2, 9)]
        self.joint_ids = [
            _named(world.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in arm_joint_names
        ]
        self.dof_ids = np.array([world.model.jnt_dofadr[i] for i in self.joint_ids], dtype=np.int32)
        self.qpos_ids = np.array([world.model.jnt_qposadr[i] for i in self.joint_ids], dtype=np.int32)
        self.actuator_ids = np.array([
            _named(world.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in arm_joint_names
        ], dtype=np.int32)
        self.gripper_joint_name = "JawBlock01_Joint" if side == "right" else "JawBlock03_Joint"
        self.gripper_actuator = _named(world.model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + "gripper")
        finger_names = (
            ("JawBlock01_Link_collision", "JawBlock02_Link_collision")
            if side == "right"
            else ("JawBlock03_Link_collision", "JawBlock04_Link_collision")
        )
        self.finger_geom_ids = np.array(
            [_named(world.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in finger_names],
            dtype=np.int32,
        )
        # The generated BRX MJCF fuses the fixed base link into MuJoCo's world body.
        self.base_body_id = 0
        self._home = np.zeros(len(self.joint_ids), dtype=np.float32)
        self._ik_data = mujoco.MjData(world.model)
        self.ik_failure_count = 0
        self.last_ik_position_error = float("nan")
        self.last_ik_rotation_error_deg = float("nan")
        self.last_ik_target_world = np.full(3, np.nan, dtype=np.float32)
        self.last_ik_achieved_world = np.full(3, np.nan, dtype=np.float32)

    @property
    def T_base_in_cam(self) -> np.ndarray:
        return self.camera.pose_world_to_camera(
            self.world.data.xmat[self.base_body_id].reshape(3, 3),
            self.world.data.xpos[self.base_body_id],
        )

    def get_T_ee_in_cam(self) -> np.ndarray:
        return self.camera.pose_world_to_camera(self.world.data.site_xmat[self.site_id].reshape(3, 3), self.world.data.site_xpos[self.site_id])

    def _ik(self, target: np.ndarray) -> Optional[np.ndarray]:
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            return None
        self._ik_data.qpos[:] = self.world.data.qpos
        self._ik_data.qvel[:] = self.world.data.qvel
        q = self._ik_data.qpos[self.qpos_ids].copy()
        for _ in range(self.ik_iterations):
            self._ik_data.qpos[self.qpos_ids] = q
            mujoco.mj_forward(self.world.model, self._ik_data)
            current_position = self._ik_data.site_xpos[self.site_id].copy()
            current_rotation = self._ik_data.site_xmat[self.site_id].reshape(3, 3).copy()
            position_error = target[:3, 3] - current_position
            rotation_error = Rotation.from_matrix(target[:3, :3] @ current_rotation.T).as_rotvec()
            error = np.concatenate([
                position_error,
                self.ik_rotation_weight * rotation_error,
            ])
            jac_position = np.zeros((3, self.world.model.nv))
            jac_rotation = np.zeros((3, self.world.model.nv))
            mujoco.mj_jacSite(self.world.model, self._ik_data, jac_position, jac_rotation, self.site_id)
            jacobian = np.vstack([
                jac_position[:, self.dof_ids],
                self.ik_rotation_weight * jac_rotation[:, self.dof_ids],
            ])
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + self.ik_damping**2 * np.eye(6), error)
            q += np.clip(dq, -0.12, 0.12)
            q = np.clip(q, self.world.model.jnt_range[self.joint_ids, 0], self.world.model.jnt_range[self.joint_ids, 1])
        self._ik_data.qpos[self.qpos_ids] = q
        mujoco.mj_forward(self.world.model, self._ik_data)
        achieved_position = self._ik_data.site_xpos[self.site_id].copy()
        achieved_rotation = self._ik_data.site_xmat[self.site_id].reshape(3, 3)
        self.last_ik_position_error = float(
            np.linalg.norm(achieved_position - target[:3, 3])
        )
        self.last_ik_rotation_error_deg = float(np.degrees(
            Rotation.from_matrix(
                target[:3, :3] @ achieved_rotation.T
            ).magnitude()
        ))
        self.last_ik_target_world = target[:3, 3].copy()
        self.last_ik_achieved_world = achieved_position
        if self.last_ik_position_error > self.ik_position_tolerance:
            return None
        if self.last_ik_rotation_error_deg > self.ik_rotation_tolerance_deg:
            return None
        return q

    def _world_target_position(self, target_camera: np.ndarray) -> np.ndarray:
        camera_origin = self.world.data.cam_xpos[self.camera.camera_id]
        world_to_mj = self.world.data.cam_xmat[self.camera.camera_id].reshape(3, 3).T
        return camera_origin + world_to_mj.T @ CV_FROM_MUJOCO_CAMERA @ target_camera[:3, 3]

    def world_position_from_camera(self, target_camera: np.ndarray) -> np.ndarray:
        """Convert a camera-frame Cartesian target to a world-space position."""
        return self._world_target_position(target_camera)

    def _world_target_pose(self, target_camera: np.ndarray) -> np.ndarray:
        cam_rotation = CV_FROM_MUJOCO_CAMERA @ self.world.data.cam_xmat[self.camera.camera_id].reshape(3, 3).T
        return _pose(
            cam_rotation.T @ target_camera[:3, :3],
            self._world_target_position(target_camera),
        )

    def _set_ee_target(self, T_ee_in_cam: np.ndarray) -> bool:
        q_target = self._ik(self._world_target_pose(T_ee_in_cam))
        if q_target is None:
            self.ik_failure_count += 1
            return False
        self.world.data.ctrl[self.actuator_ids] = q_target
        return True

    def move_ee_in_cam(self, T_ee_in_cam: np.ndarray, duration: float, blocking: bool = False) -> bool:
        """Send a servo target, or synchronously interpolate a blocking move.

        Nonblocking policy commands only set actuator controls. The controller
        sends the gripper command next and calls ``world.advance(duration)``
        once, so both actuators evolve together. No queued motion survives reset.
        """
        if not blocking:
            return self._set_ee_target(T_ee_in_cam)
        start_camera = self.get_T_ee_in_cam()
        sub_duration = float(duration) / self.interpolation_steps
        for index in range(1, self.interpolation_steps + 1):
            subtarget = _interpolate_pose(
                start_camera,
                T_ee_in_cam,
                index / self.interpolation_steps,
            )
            if not self._set_ee_target(subtarget):
                return False
            steps = max(1, int(round(sub_duration / self.world.model.opt.timestep)))
            self.world.step(steps)
        return True

    def get_gripper(self) -> float:
        q = float(self.world.data.qpos[
            self.world.model.jnt_qposadr[
                self.world.model.actuator_trnid[self.gripper_actuator, 0]
            ]
        ])
        return float(np.clip(q / self.gripper_max, 0.0, 1.0))

    def get_gripper_midpoint_world(self) -> np.ndarray:
        """Return the midpoint between the two collision pads in world coordinates."""
        return np.mean(self.world.data.geom_xpos[self.finger_geom_ids], axis=0).copy()

    def set_gripper(self, value: float, blocking: bool = False) -> None:
        self.world.data.ctrl[self.gripper_actuator] = float(
            np.clip(value, 0.0, 1.0) * self.gripper_max
        )
        if blocking:
            self.world.advance(0.5)

    def go_home(self, blocking: bool = True) -> None:
        self.world.data.ctrl[self.actuator_ids] = self._home
        self.world.step(max(1, int(round(1.0 / self.world.model.opt.timestep))))

    def close(self) -> None:
        return None


class OracleSimPerception(Perception):
    """Ground-truth poses plus training-style gripper/keypoint overlays."""

    def __init__(self, world: SimWorld, camera: SimCamera, anchor_key: str = "obj_1") -> None:
        self.world, self.camera, self.anchor_key = world, camera, anchor_key
        self.geom_names = {"obj_1": "target_container_bounds", "obj_2": "block_geom"}
        self._body_ids = {
            key: _named(world.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for key, name in {"obj_1": "target_container", "obj_2": "block"}.items()
        }

    def estimate_objects(self, frames: List[Frame]) -> Dict[str, ObjectState]:
        del frames
        result: Dict[str, ObjectState] = {}
        for key, body_id in self._body_ids.items():
            rotation = self.world.data.xmat[body_id].reshape(3, 3)
            position = self.world.data.xpos[body_id]
            size = self.world.model.geom_size[_named(self.world.model, mujoco.mjtObj.mjOBJ_GEOM, self.geom_names[key])]
            corners = np.array([[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32)
            result[key] = ObjectState(self.camera.pose_world_to_camera(rotation, position), corners)
        return result

    def make_clean_image(self, frame: Frame, ee_poses_in_cam: Dict[str, np.ndarray], grippers: Dict[str, float]) -> np.ndarray:
        canvas = frame.rgb.copy()
        for side, pose in ee_poses_in_cam.items():
            self._draw_gripper(canvas, pose, grippers.get(side, 0.0))
        for key, state in self.estimate_objects([]).items():
            points = (state.T_in_cam[:3, :3] @ state.kpts_local.T).T + state.T_in_cam[:3, 3]
            for point in points:
                if point[2] <= 0.01:
                    continue
                uv = self._project(point, frame.K)
                if uv is not None:
                    color = (255, 80, 30) if key == self.anchor_key else (30, 30, 220)
                    cv2.circle(canvas, uv, 4, color, -1)
        return canvas

    def _draw_gripper(self, image: np.ndarray, pose: np.ndarray, grasp: float) -> None:
        width = 0.05 if grasp > 0.5 else 0.18
        local = np.array([[0, -(0.08 + 0.08), 0], [-width / 2, 0, 0], [width / 2, 0, 0], [-width / 2, -0.08, 0], [width / 2, -0.08, 0]], dtype=np.float32)
        points = (pose[:3, :3] @ local.T).T + pose[:3, 3]
        segments = ((0, 1), (0, 2), (1, 3), (2, 4))
        for start, end in segments:
            uv0, uv1 = self._project(points[start], self._last_K), self._project(points[end], self._last_K)
            if uv0 is not None and uv1 is not None:
                cv2.line(image, uv0, uv1, (0, 255, 255), 2, cv2.LINE_AA)

    def _project(self, point: np.ndarray, K: np.ndarray | None = None) -> Optional[tuple[int, int]]:
        if K is None:
            return None
        if point[2] <= 0.01:
            return None
        uv = K @ point
        return int(round(float(uv[0] / point[2]))), int(round(float(uv[1] / point[2])))

    @property
    def _last_K(self) -> np.ndarray:
        return self.camera.K
