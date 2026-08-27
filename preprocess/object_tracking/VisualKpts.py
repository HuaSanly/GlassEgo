"""Render hand and object keypoints on raw and arm-inpainted RGB frames."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from utils.utils_media import create_video_from_frames
from utils.utils_artifact_store import FrameArtifactStore


class VisualKptsGenerator:
    """Project the existing world-space hand and tracked object state to RGB."""

    def __init__(
        self,
        unit_dir: str | Path,
        cfg,
        store: FrameArtifactStore | None = None,
    ):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg
        self.store = store or FrameArtifactStore(self.unit_dir)

    def run(
        self,
        frame_images: dict[int, np.ndarray],
        frame_indices: list[int],
        training_frames: set[int],
        vio_result,
        hands,
        tracks_document: dict,
        fps: float,
    ) -> dict:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {"status": "disabled", "frames": 0, "path": None}
        if not frame_indices:
            raise ValueError("VisualKpts requires at least one frame")

        self.calibration = vio_result.calibration
        vio_by_idx = {int(frame.frame_idx): frame for frame in vio_result.trajectory.frames}
        tracks_by_frame = self._index_tracks(tracks_document)
        rendered = []
        outputs = []
        for frame_idx in frame_indices:
            frame_idx = int(frame_idx)
            image = frame_images.get(frame_idx)
            if image is None:
                raise FileNotFoundError(f"Missing RGB frame for VisualKpts: {frame_idx}")
            vio_frame = vio_by_idx.get(frame_idx)
            if vio_frame is None:
                raise ValueError(f"Missing VIO pose for VisualKpts frame: {frame_idx}")
            is_training = frame_idx in training_frames
            frame_dir = self.store.frame_dir(frame_idx, is_training)
            rgb_path = frame_dir / "rgb.png"
            if not rgb_path.is_file():
                self.store.write_image(frame_idx, "rgb.png", image, is_training)

            raw = cv2.imread(str(rgb_path))
            if raw is None:
                raise IOError(f"Unable to read RGB frame for VisualKpts: {rgb_path}")
            overlay = self._render_frame(
                raw,
                frame_idx,
                vio_frame,
                self._hand_frame(hands, frame_idx),
                tracks_by_frame.get(frame_idx, {}),
            )
            raw_overlay_path = self.store.write_image(
                frame_idx, "rgb_WArmObjKpts.png", overlay, is_training
            )

            inpainted_path = frame_dir / "rgb_WoArm.png"
            inpainted = cv2.imread(str(inpainted_path))
            if inpainted is None:
                raise FileNotFoundError(
                    f"Missing LaMa output for VisualKpts: {inpainted_path}"
                )
            inpainted_overlay = self._render_frame(
                inpainted,
                frame_idx,
                vio_frame,
                self._hand_frame(hands, frame_idx),
                tracks_by_frame.get(frame_idx, {}),
            )
            clean_overlay_path = self.store.write_image(
                frame_idx,
                "rgb_WoArm_WArmObjKpts.png",
                inpainted_overlay,
                is_training,
            )
            outputs.extend(
                [
                    self.store.relative_path(raw_overlay_path),
                    self.store.relative_path(clean_overlay_path),
                ]
            )
            rendered.append(inpainted_overlay)

        video_path = self.store.vis_dir / "visualkpts_vis.mp4"
        output_cfg = getattr(self.cfg, "output", None)
        if bool(getattr(output_cfg, "export_video", True)):
            create_video_from_frames(
                rendered,
                video_path,
                fps=float(fps),
                export_gif=bool(getattr(output_cfg, "export_gif", False)),
                ratio=int(getattr(output_cfg, "gif_frame_ratio", 10)),
                export_video=True,
            )
        report = {
            "status": "completed",
            "frames": len(frame_indices),
            "outputs": outputs,
            "video": self.store.relative_path(video_path)
            if video_path.is_file()
            else None,
        }
        report_path = self.store.module_vis_dir("visualkpts") / "report.json"
        self._atomic_write_json(report_path, report)
        return {**report, "path": self.store.relative_path(report_path)}

    def _render_frame(
        self,
        image_bgr: np.ndarray,
        frame_idx: int,
        vio_frame,
        hand_frame,
        object_tracks: dict,
    ) -> np.ndarray:
        canvas = image_bgr.copy()
        k = self._camera_matrix(vio_frame, canvas.shape)
        world_to_camera = np.linalg.inv(np.asarray(vio_frame.c2w, dtype=np.float64))

        for hand in self._hands(hand_frame):
            pose = hand.midpoint_pose_opt_world
            if pose is None:
                continue
            self._draw_gripper(
                canvas,
                np.asarray(pose, dtype=np.float64),
                world_to_camera,
                k,
                float(hand.grasp_score),
            )

        colors = self._object_colors()
        for object_index, (object_key, data) in enumerate(sorted(object_tracks.items())):
            points = np.asarray(data, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] < 2:
                continue
            color = colors[object_index % len(colors)]
            for point in points:
                if not np.all(np.isfinite(point[:2])):
                    continue
                self._draw_point(canvas, point[0], point[1], color)
        return canvas

    def _draw_gripper(self, canvas, pose_world, world_to_camera, k, grasp_score):
        width = self._value("gripper_width_min", 0.05) if grasp_score > 0.5 else self._value("gripper_width_max", 0.18)
        finger_len = self._value("gripper_finger_len", 0.08)
        root_depth = self._value("gripper_root_depth", 0.08)
        half_width = width / 2.0
        points_local = np.asarray(
            [
                [0.0, 0.0, -(finger_len + root_depth), 1.0],
                [-half_width, 0.0, -finger_len, 1.0],
                [half_width, 0.0, -finger_len, 1.0],
                [-half_width, 0.0, 0.0, 1.0],
                [half_width, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        points_world = (pose_world @ points_local.T).T
        points_camera = (world_to_camera @ points_world.T).T
        projected = self._project(points_camera[:, :3], k)
        if projected is None:
            return
        segments = ((0, 1), (0, 2), (1, 3), (2, 4))
        color = (255, 255, 255) if grasp_score > 0.5 else (0, 80, 255)
        for start, end in segments:
            if points_camera[start, 2] <= 0.01 or points_camera[end, 2] <= 0.01:
                continue
            cv2.line(canvas, projected[start], projected[end], color, 2, cv2.LINE_AA)
            self._draw_point(canvas, *projected[start], color)
            self._draw_point(canvas, *projected[end], color)

    @staticmethod
    def _project(points_camera, k):
        depth = points_camera[:, 2]
        if np.any(depth <= 0.01):
            return None
        homogeneous = (k @ points_camera.T).T
        return [
            (int(round(point[0] / point[2])), int(round(point[1] / point[2])))
            for point in homogeneous
        ]

    def _draw_point(self, canvas, u, v, color):
        u, v = int(round(u)), int(round(v))
        if not (0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]):
            return
        radius = int(round(self._value("radius_current", 4)))
        cv2.circle(canvas, (u, v), radius, color, -1, cv2.LINE_AA)
        core_radius = int(round(self._value("white_core_radius", 1)))
        cv2.circle(canvas, (u, v), core_radius, (255, 255, 255), -1, cv2.LINE_AA)

    def _camera_matrix(self, vio_frame, shape):
        calibration = getattr(vio_frame, "calibration", None)
        if calibration is not None:
            fx, fy, cx, cy = calibration.intrinsics
        else:
            calibration = getattr(self, "calibration", None)
            if calibration is None:
                raise ValueError("VisualKpts requires camera intrinsics")
            fx, fy, cx, cy = calibration.intrinsics
        return np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    @staticmethod
    def _hand_frame(hands, frame_idx):
        if hands is None:
            return None
        for frame in hands.hands:
            if int(frame.idx) == int(frame_idx):
                return frame
        return None

    @staticmethod
    def _hands(frame):
        if frame is None:
            return ()
        return tuple(hand for hand in (frame.hand_l, frame.hand_r) if hand is not None)

    @staticmethod
    def _index_tracks(document):
        frame_indices = [int(value) for value in document.get("frames", [])]
        indexed = {frame_idx: {} for frame_idx in frame_indices}
        for object_key, data in document.get("objects", {}).items():
            tracks = data.get("tracks", [])
            for position, frame_idx in enumerate(frame_indices):
                if position < len(tracks):
                    indexed[frame_idx][object_key] = tracks[position]
        return indexed

    def _value(self, name, default):
        return float(getattr(self.cfg, f"visualkpts_{name}", default))

    def _object_colors(self):
        values = getattr(self.cfg, "obj_colors", None)
        return [tuple(int(channel) for channel in color) for color in (values or [(255, 255, 0), (0, 255, 0)])]

    @staticmethod
    def _atomic_write_json(path: Path, document: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        )
        temporary_path = Path(handle.name)
        try:
            with handle:
                json.dump(document, handle, indent=2, ensure_ascii=True)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
