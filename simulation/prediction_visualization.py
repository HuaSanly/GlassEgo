"""Display decoded model predictions without changing MuJoCo physics or policy RGB."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


class PredictionOverlay:
    """A frozen world-frame prediction, independent of IK and subsequent camera motion."""

    def __init__(
        self, poses_world: np.ndarray, grasps: np.ndarray,
        grasp_threshold: float, exec_horizon: int,
    ) -> None:
        self.poses = np.asarray(poses_world, dtype=float).copy()
        self.grasps = np.asarray(grasps, dtype=float).reshape(-1).copy()
        if self.poses.shape != (len(self.grasps), 4, 4) or not len(self.grasps):
            raise ValueError("Prediction display requires a nonempty (H, 4, 4) pose sequence and H grasps")
        if not np.isfinite(self.poses).all() or not np.isfinite(self.grasps).all():
            raise ValueError("Cannot display non-finite model predictions")
        self.grasp_threshold = grasp_threshold
        self.exec_horizon = exec_horizon

    def draw(self, scene: mujoco.MjvScene, active_index: int | None = None) -> None:
        """Append decorative geometry only; never add bodies or collision geometry."""
        count = len(self.poses)
        axes_indices = set(range(0, count, max(1, count // 8))) | {count - 1}
        if active_index is not None:
            axes_indices.add(active_index)
        required = count + count - 1 + 3 * len(axes_indices)
        if scene.ngeom + required > scene.maxgeom:
            raise ValueError(f"Prediction needs {required} display geoms; only {scene.maxgeom - scene.ngeom} available")

        def geom(kind, position, size, color):
            item = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(item, kind, np.full(3, size), position, np.eye(3).ravel(), color)
            item.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1
            return item

        def connector(start, end, color, width=0.002, kind=mujoco.mjtGeom.mjGEOM_CAPSULE):
            item = geom(kind, start, width, color)
            mujoco.mjv_connector(item, kind, width, start, end)

        for index, (pose, grasp) in enumerate(zip(self.poses, self.grasps)):
            closed = grasp > self.grasp_threshold
            alpha = 1.0 if index < self.exec_horizon else 0.55
            color = (1.0, 0.45, 0.05, alpha) if closed else (0.0, 0.85, 0.9, alpha)
            position = pose[:3, 3]
            radius = 0.009 if index == active_index else 0.004
            point = geom(mujoco.mjtGeom.mjGEOM_SPHERE, position, radius, color)
            if index == active_index:
                point.label = f"pred[{index}] p={grasp:.2f}"
            if index:
                connector(self.poses[index - 1, :3, 3], position, color)
            if index in axes_indices:
                length = 0.055 if index == active_index else 0.025
                for axis, axis_color in enumerate(((1, 0.1, 0.1, 1), (0.1, 1, 0.1, 1), (0.2, 0.3, 1, 1))):
                    connector(position, position + length * pose[:3, axis], axis_color,
                              width=0.0015, kind=mujoco.mjtGeom.mjGEOM_ARROW)

    def show(self, viewer, active_index: int | None = None) -> None:
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            self.draw(viewer.user_scn, active_index)
        viewer.sync()

    def frame_camera(self, camera: mujoco.MjvCamera, world) -> None:
        """Frame the task and prediction, including targets outside robot reach."""
        points = np.vstack((self.poses[:, :3, 3], world.data.geom("block_geom").xpos,
                            world.data.geom("target_container_bounds").xpos))
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (points.min(axis=0) + points.max(axis=0)) / 2
        camera.distance = max(0.65, 2.5 * float(np.linalg.norm(np.ptp(points, axis=0))))
        camera.azimuth, camera.elevation = 130, -55

    def save(self, world, output: Path, metadata: dict) -> None:
        """Export the latest prediction and an observer-only overview PNG."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **metadata,
            "coordinate_frame": "mujoco_world_z_up",
            "pose_convention": "decoded_robot_ee_before_smoothing_limits_and_ik",
            "position_units": "meters",
            "poses_world": self.poses.tolist(),
            "grasp_probabilities": self.grasps.tolist(),
            "grasp_threshold": self.grasp_threshold,
            "exec_horizon": self.exec_horizon,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        camera = mujoco.MjvCamera()
        self.frame_camera(camera, world)
        # This separate scene update is overwritten by the next policy camera render.
        world.renderer.disable_depth_rendering()
        world.renderer.update_scene(world.data, camera=camera)
        self.draw(world.renderer.scene)
        bgr = cv2.cvtColor(world.renderer.render(), cv2.COLOR_RGB2BGR)
        legend = (
            f"Raw model EE prediction: {len(self.poses)} steps (world, meters)",
            "Cyan: open | Orange: close | RGB axes: X/Y/Z",
            f"Bright: first {self.exec_horizon} commands | Faint: remaining horizon",
        )
        for index, text in enumerate(legend):
            position = (12, 22 + index * 20)
            cv2.putText(bgr, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(output.with_suffix(".png")), bgr):
            raise OSError(f"Could not write prediction image: {output.with_suffix('.png')}")
