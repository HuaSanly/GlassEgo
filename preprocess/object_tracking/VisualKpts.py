"""Render HumanEgo-style hand and object keypoints on RGB frames."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from utils.utils_artifact_store import FrameArtifactStore
from utils.utils_media import create_video_from_frames


class VisualKptsGenerator:
    """Project world-space hand geometry and tracked object points to RGB."""

    _OBJECT_CLUSTER_SCALE = 3.0
    _OBJECT_CLUSTER_MIN_RADIUS_PX = 8.0
    _TRACK_RANSAC_THRESHOLD_PX = 8.0
    _TRACK_RANSAC_MIN_POINTS = 4
    _HAND_SIDES = ("hand_l", "hand_r")
    _HAND_OPEN_COLORS = (
        "color_hand_1_line",
        "color_hand_2_line",
        "color_hand_3_line",
        "color_hand_4_line",
    )
    _HAND_CLOSED_COLORS = (
        "color_hand_closed_1",
        "color_hand_closed_2",
        "color_hand_closed_3",
        "color_hand_closed_4",
    )

    def __init__(
        self,
        unit_dir: str | Path,
        cfg,
        store: FrameArtifactStore | None = None,
    ):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg
        self.store = store or FrameArtifactStore(self.unit_dir)
        self.calibration = None
        self._frame_indices: list[int] = []
        self._frame_positions: dict[int, int] = {}
        self._hands_by_frame: dict[int, object] = {}
        self._tracks_by_frame: dict[int, dict[str, object]] = {}
        self._visibility_by_frame: dict[int, dict[str, object]] = {}
        self._tracks_document: dict = {}

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
        if not bool(self._setting("enabled", True)):
            return {"status": "disabled", "frames": 0, "path": None}
        if not frame_indices:
            raise ValueError("VisualKpts requires at least one frame")

        self.calibration = vio_result.calibration
        self._frame_indices = [int(value) for value in frame_indices]
        self._frame_positions = {
            frame_idx: position
            for position, frame_idx in enumerate(self._frame_indices)
        }
        self._hands_by_frame = self._index_hand_frames(hands)
        self._tracks_document = tracks_document or {}
        self._tracks_by_frame = self._index_tracks(self._tracks_document)
        self._visibility_by_frame = self._index_visibility(self._tracks_document)
        self._filter_object_track_outliers()

        vio_by_idx = {
            int(frame.frame_idx): frame for frame in vio_result.trajectory.frames
        }
        rendered = []
        outputs = []
        for position, frame_idx in enumerate(self._frame_indices):
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

            frame_hands = self._hands_by_frame.get(frame_idx)
            frame_objects = self._tracks_by_frame.get(frame_idx, {})
            overlay = self._render_frame(
                raw,
                frame_idx,
                vio_frame,
                frame_hands,
                frame_objects,
                position,
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
                frame_hands,
                frame_objects,
                position,
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
        output_cfg = self._setting("output", None)
        if bool(self._setting_from(output_cfg, "export_video", True)):
            create_video_from_frames(
                rendered,
                video_path,
                fps=float(fps),
                export_gif=bool(self._setting_from(output_cfg, "export_gif", False)),
                ratio=int(self._setting_from(output_cfg, "gif_frame_ratio", 10)),
                export_video=True,
            )

        report = {
            "status": "completed",
            "frames": len(self._frame_indices),
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
        frame_position: int | None = None,
    ) -> np.ndarray:
        canvas = image_bgr.copy()
        k = self._camera_matrix(vio_frame, canvas.shape)
        world_to_camera = np.linalg.inv(np.asarray(vio_frame.c2w, dtype=np.float64))
        if frame_position is None:
            frame_position = self._frame_positions.get(int(frame_idx), 0)

        self._draw_advanced_trails(canvas, frame_position, world_to_camera, k)

        for hand in self._hands(hand_frame):
            pose = getattr(hand, "midpoint_pose_opt_world", None)
            if pose is None:
                continue
            pose = np.asarray(pose, dtype=np.float64)
            if pose.size != 16 or not np.all(np.isfinite(pose)):
                continue
            pose = pose.reshape(4, 4)
            grasp_score = self._hand_grasp_score(hand)
            if bool(self._setting("draw_axes", False)):
                self._draw_xyz_axes(
                    canvas,
                    pose,
                    world_to_camera,
                    k,
                    self._value("axes_len", 0.06),
                )
            self._draw_gripper(
                canvas,
                pose,
                world_to_camera,
                k,
                grasp_score,
            )

        colors = self._object_colors()
        object_points = self._tracked_object_points(frame_idx, object_tracks)
        object_keys = sorted(object_points)

        for object_index, object_key in enumerate(object_keys):
            if not self._is_object_key(object_key):
                continue
            points = np.asarray(object_points.get(object_key, []), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] < 2:
                continue
            color = colors[object_index % len(colors)]
            for point in points:
                if not np.all(np.isfinite(point[:2])):
                    continue
                self._draw_kpt_with_core(
                    canvas,
                    point[0],
                    point[1],
                    color,
                    self._value("radius_current", 4) + 1,
                )
        return canvas

    def _tracked_object_points(self, frame_idx, object_tracks):
        tracked = {}
        for object_key, points in object_tracks.items():
            if not self._is_object_key(object_key):
                continue
            if isinstance(points, Mapping):
                points = points.get("tracks", [])
            points = np.asarray(points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] < 2:
                continue
            point_mask = self._object_render_mask(frame_idx, object_key, points)
            points = points[point_mask]
            tracked[object_key] = self._largest_object_cluster(points)
        return tracked

    def _object_render_mask(self, frame_idx, object_key, points):
        point_mask = np.isfinite(points[:, :2]).all(axis=1)
        render_validity = self._render_validity_by_frame.get(int(frame_idx), {}).get(
            object_key
        )
        if render_validity is not None and len(render_validity) == len(points):
            point_mask &= render_validity
        visibility = np.asarray(
            self._visibility_by_frame.get(int(frame_idx), {}).get(object_key, []),
            dtype=np.float64,
        )
        if len(visibility) == len(points):
            point_mask &= visibility > 0
        valid_indices = np.flatnonzero(point_mask)
        if len(valid_indices) < 3:
            return point_mask
        cluster_mask = self._largest_object_cluster_mask(points[valid_indices, :2])
        point_mask[valid_indices] = cluster_mask
        return point_mask

    def _filter_object_track_outliers(self) -> None:
        self._render_validity_by_frame = {
            int(frame_idx): {}
            for frame_idx in self._tracks_document.get("frames", [])
        }
        for object_key, data in self._tracks_document.get("objects", {}).items():
            if not self._is_object_key(object_key) or not isinstance(data, Mapping):
                continue
            tracks = np.asarray(data.get("tracks", []), dtype=np.float64)
            visibility = np.asarray(data.get("visibility", []), dtype=np.float64)
            if (
                tracks.ndim != 3
                or tracks.shape[2] < 2
                or visibility.shape != tracks.shape[:2]
            ):
                continue

            finite = np.isfinite(tracks[:, :, :2]).all(axis=2)
            base_valid = (visibility > 0) & finite
            render_valid = base_valid.copy()
            for frame_position in range(1, len(tracks)):
                common = np.flatnonzero(
                    base_valid[frame_position - 1] & base_valid[frame_position]
                )
                if len(common) < self._TRACK_RANSAC_MIN_POINTS:
                    continue

                previous_points = tracks[frame_position - 1, common, :2]
                current_points = tracks[frame_position, common, :2]
                _, inliers = cv2.estimateAffine2D(
                    previous_points,
                    current_points,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self._TRACK_RANSAC_THRESHOLD_PX,
                    maxIters=500,
                    confidence=0.99,
                    refineIters=10,
                )
                if inliers is None:
                    continue
                render_valid[frame_position, common[inliers.ravel() == 0]] = False

            frame_indices = [int(frame_idx) for frame_idx in self._tracks_document["frames"]]
            for frame_position, frame_idx in enumerate(frame_indices):
                if frame_position >= len(render_valid):
                    break
                self._render_validity_by_frame.setdefault(frame_idx, {})[
                    object_key
                ] = render_valid[frame_position]

    @classmethod
    def _largest_object_cluster(cls, points: np.ndarray) -> np.ndarray:
        return points[cls._largest_object_cluster_mask(points)]

    @classmethod
    def _largest_object_cluster_mask(cls, points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            return np.ones(len(points), dtype=bool)

        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = distances.min(axis=1)
        finite_nearest = nearest_distances[np.isfinite(nearest_distances)]
        if len(finite_nearest) == 0:
            return np.ones(len(points), dtype=bool)

        radius = max(
            cls._OBJECT_CLUSTER_MIN_RADIUS_PX,
            float(np.median(finite_nearest)) * cls._OBJECT_CLUSTER_SCALE,
        )
        adjacency = distances <= radius
        remaining = set(range(len(points)))
        clusters = []
        while remaining:
            pending = [remaining.pop()]
            cluster = []
            while pending:
                point_index = pending.pop()
                cluster.append(point_index)
                neighbors = {
                    int(index)
                    for index in np.flatnonzero(adjacency[point_index])
                    if int(index) in remaining
                }
                remaining.difference_update(neighbors)
                pending.extend(neighbors)
            clusters.append(cluster)

        largest = max(clusters, key=len)
        if len(largest) <= len(points) / 2:
            return np.ones(len(points), dtype=bool)
        mask = np.zeros(len(points), dtype=bool)
        mask[np.asarray(sorted(largest), dtype=np.int64)] = True
        return mask

    def _draw_gripper(
        self,
        canvas: np.ndarray,
        pose_world: np.ndarray,
        world_to_camera: np.ndarray,
        k: np.ndarray,
        grasp_score: float,
    ) -> None:
        segments, is_closed = self._get_segments_3d_oriented_binary(
            pose_world, grasp_score
        )
        c1, c2, c3, c4 = self._hand_colors(is_closed)
        points_per_line = max(2, int(round(self._value("pts_per_line", 6))))

        for segment_index, (start_world, end_world) in enumerate(segments):
            start_color, end_color = (c4, c3) if segment_index < 2 else (c2, c1)
            line_world = np.linspace(start_world, end_world, points_per_line)
            line_camera = self._world_to_camera(line_world, world_to_camera)
            for point_index, point_camera in enumerate(line_camera):
                if point_camera[2] <= 0.1:
                    continue
                projected = self._project_one(point_camera, k)
                if projected is None:
                    continue
                ratio = point_index / (points_per_line - 1)
                color = self._interpolate_bgr(start_color, end_color, ratio)
                self._draw_kpt_with_core(
                    canvas,
                    projected[0],
                    projected[1],
                    color,
                    self._value("radius_current", 4),
                )

    def _get_segments_3d_oriented_binary(
        self,
        pose_world: np.ndarray,
        grasp_score: float,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], bool]:
        """Build the HumanEgo gripper: X is spread and -Y points to the wrist."""
        pose = np.asarray(pose_world, dtype=np.float64)
        if pose.size != 16:
            raise ValueError("VisualKpts hand pose must contain 16 values")
        pose = pose.reshape(4, 4)
        is_closed = float(grasp_score) > 0.5
        width = self._value(
            "gripper_width_min" if is_closed else "gripper_width_max",
            0.05 if is_closed else 0.18,
        )
        finger_len = self._value("gripper_finger_len", 0.08)
        root_depth = self._value("gripper_root_depth", 0.08)
        half_width = width / 2.0

        # HumanEgo's midpoint pose uses local X for finger spread and local Y
        # for the approach direction. The root is behind the finger bases at -Y.
        points_local = np.asarray(
            [
                [-half_width, 0.0, 0.0, 1.0],
                [half_width, 0.0, 0.0, 1.0],
                [-half_width, -finger_len, 0.0, 1.0],
                [half_width, -finger_len, 0.0, 1.0],
                [0.0, -(finger_len + root_depth), 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        points_world = (pose @ points_local.T).T[:, :3]
        left_tip, right_tip, left_base, right_base, root = points_world
        return [
            (root, left_base),
            (root, right_base),
            (left_base, left_tip),
            (right_base, right_tip),
        ], is_closed

    @staticmethod
    def _world_to_camera(
        points_world: np.ndarray,
        world_to_camera: np.ndarray,
    ) -> np.ndarray:
        points_world = np.asarray(points_world, dtype=np.float64)
        transform = np.asarray(world_to_camera, dtype=np.float64)
        return (transform[:3, :3] @ points_world.T).T + transform[:3, 3]

    @staticmethod
    def _project_one(
        point_camera: np.ndarray,
        k: np.ndarray,
    ) -> tuple[float, float] | None:
        point_camera = np.asarray(point_camera, dtype=np.float64)
        depth = float(point_camera[2])
        if depth <= 0.01 or not np.all(np.isfinite(point_camera)):
            return None
        homogeneous = np.asarray(k, dtype=np.float64) @ point_camera
        return float(homogeneous[0] / depth), float(homogeneous[1] / depth)

    @classmethod
    def _project(cls, points_camera: np.ndarray, k: np.ndarray):
        """Project a batch of camera-space points, preserving the old helper API."""
        points_camera = np.asarray(points_camera, dtype=np.float64)
        if points_camera.ndim != 2 or points_camera.shape[1] != 3:
            return None
        projected = []
        for point in points_camera:
            uv = cls._project_one(point, k)
            if uv is None:
                return None
            projected.append((int(round(uv[0])), int(round(uv[1]))))
        return projected

    def _draw_advanced_trails(
        self,
        canvas: np.ndarray,
        frame_position: int,
        world_to_camera: np.ndarray,
        k: np.ndarray,
    ) -> None:
        trail_len = max(0, int(round(self._value("trail_len", 0))))
        if trail_len <= 0 or frame_position <= 0:
            return
        trail_step = max(1, int(round(self._value("trail_step", 1))))
        start_position = max(0, frame_position - trail_len)
        overlay = canvas.copy()
        open_colors = self._hand_colors(False)

        for position in range(start_position, frame_position, trail_step):
            next_position = min(position + trail_step, frame_position)
            current_idx = self._frame_indices[position]
            next_idx = self._frame_indices[next_position]
            current_hands = self._hands_by_frame.get(current_idx)
            next_hands = self._hands_by_frame.get(next_idx)
            time_ratio = (position - start_position) / max(
                frame_position - start_position,
                1,
            )
            thickness = int(round(
                self._value("trail_thick_min", 1)
                + (
                    self._value("trail_thick_max", 4)
                    - self._value("trail_thick_min", 1)
                )
                * time_ratio
            ))
            if current_hands is not None and next_hands is not None:
                for side in self._HAND_SIDES:
                    current_hand = self._side_hand(current_hands, side)
                    next_hand = self._side_hand(next_hands, side)
                    if current_hand is None or next_hand is None:
                        continue
                    current_pose = getattr(current_hand, "midpoint_pose_opt_world", None)
                    next_pose = getattr(next_hand, "midpoint_pose_opt_world", None)
                    if current_pose is None or next_pose is None:
                        continue
                    current_segments, _ = self._get_segments_3d_oriented_binary(
                        current_pose,
                        self._hand_grasp_score(current_hand),
                    )
                    next_segments, _ = self._get_segments_3d_oriented_binary(
                        next_pose,
                        self._hand_grasp_score(next_hand),
                    )
                    points_per_line = max(2, int(round(self._value("pts_per_line", 6))))
                    for segment_index, (current_segment, next_segment) in enumerate(
                        zip(current_segments, next_segments)
                    ):
                        start_color, end_color = (
                            (open_colors[3], open_colors[2])
                            if segment_index < 2
                            else (open_colors[1], open_colors[0])
                        )
                        current_line = np.linspace(
                            current_segment[0],
                            current_segment[1],
                            points_per_line,
                        )
                        next_line = np.linspace(
                            next_segment[0],
                            next_segment[1],
                            points_per_line,
                        )
                        current_camera = self._world_to_camera(
                            current_line,
                            world_to_camera,
                        )
                        next_camera = self._world_to_camera(
                            next_line,
                            world_to_camera,
                        )
                        for point_index, (point_c, point_n) in enumerate(
                            zip(current_camera, next_camera)
                        ):
                            if point_c[2] <= 0.1 or point_n[2] <= 0.1:
                                continue
                            uv_c = self._project_one(point_c, k)
                            uv_n = self._project_one(point_n, k)
                            if uv_c is None or uv_n is None:
                                continue
                            spatial_ratio = point_index / (points_per_line - 1)
                            color = self._interpolate_bgr(
                                start_color,
                                end_color,
                                spatial_ratio,
                            )
                            intensity = 0.2 + 0.8 * time_ratio
                            color = tuple(int(channel * intensity) for channel in color)
                            cv2.line(
                                overlay,
                                (int(round(uv_c[0])), int(round(uv_c[1]))),
                                (int(round(uv_n[0])), int(round(uv_n[1]))),
                                color,
                                max(1, thickness),
                                cv2.LINE_AA,
                            )

            self._draw_object_trails(
                overlay,
                start_position,
                position,
                next_position,
                time_ratio,
            )

        alpha = np.clip(self._value("trail_alpha_max", 0.9), 0.0, 1.0)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, canvas)

    def _draw_object_trails(
        self,
        canvas: np.ndarray,
        start_position: int,
        position: int,
        next_position: int,
        time_ratio: float,
    ) -> None:
        threshold = self._value("obj_motion_th", 10.0)
        thickness = max(1, int(round(
            self._value("trail_thick_min", 1)
            + (
                self._value("trail_thick_max", 4)
                - self._value("trail_thick_min", 1)
            )
            * time_ratio
        )))
        current_idx = self._frame_indices[position]
        next_idx = self._frame_indices[next_position]
        start_idx = self._frame_indices[start_position]
        current_objects = self._tracks_by_frame.get(current_idx, {})
        next_objects = self._tracks_by_frame.get(next_idx, {})
        start_objects = self._tracks_by_frame.get(start_idx, {})
        trail_color = self._color("color_object_trail", (200, 150, 0))
        intensity = 0.2 + 0.8 * time_ratio
        trail_color = tuple(int(channel * intensity) for channel in trail_color)

        for object_key, current_points in current_objects.items():
            if not self._is_object_key(object_key):
                continue
            next_points = np.asarray(next_objects.get(object_key, []), dtype=np.float64)
            start_points = np.asarray(start_objects.get(object_key, []), dtype=np.float64)
            current_points = np.asarray(current_points, dtype=np.float64)
            if (
                current_points.ndim != 2
                or next_points.ndim != 2
                or start_points.ndim != 2
                or current_points.shape[1] < 2
                or next_points.shape[1] < 2
                or start_points.shape[1] < 2
            ):
                continue
            current_valid = self._object_render_mask(
                current_idx, object_key, current_points
            )
            next_valid = self._object_render_mask(next_idx, object_key, next_points)
            start_valid = self._object_render_mask(start_idx, object_key, start_points)
            point_count = min(len(current_points), len(next_points), len(start_points))
            for point_index in range(point_count):
                current_point = current_points[point_index, :2]
                next_point = next_points[point_index, :2]
                start_point = start_points[point_index, :2]
                if not (
                    np.all(np.isfinite(current_point))
                    and np.all(np.isfinite(next_point))
                    and np.all(np.isfinite(start_point))
                ):
                    continue
                if np.linalg.norm(current_point - start_point) <= threshold:
                    continue
                if not (
                    current_valid[point_index]
                    and next_valid[point_index]
                    and start_valid[point_index]
                ):
                    continue
                cv2.line(
                    canvas,
                    tuple(int(round(value)) for value in current_point),
                    tuple(int(round(value)) for value in next_point),
                    trail_color,
                    thickness,
                    cv2.LINE_AA,
                )

    def _draw_xyz_axes(
        self,
        canvas: np.ndarray,
        pose_world: np.ndarray,
        world_to_camera: np.ndarray,
        k: np.ndarray,
        length: float = 0.06,
    ) -> None:
        pose_world = np.asarray(pose_world, dtype=np.float64).reshape(4, 4)
        transform_camera = world_to_camera @ pose_world
        origin = transform_camera[:3, 3]
        if origin[2] <= 0.01:
            return
        origin_uv = self._project_one(origin, k)
        if origin_uv is None:
            return
        origin_xy = (int(round(origin_uv[0])), int(round(origin_uv[1])))
        for axis, color, label in (
            (0, (0, 0, 255), "X"),
            (1, (0, 255, 0), "Y"),
            (2, (255, 0, 0), "Z"),
        ):
            tip = origin + transform_camera[:3, axis] * length
            tip_uv = self._project_one(tip, k)
            label_uv = self._project_one(
                origin + transform_camera[:3, axis] * length * 1.25,
                k,
            )
            if tip_uv is None or label_uv is None:
                continue
            tip_xy = (int(round(tip_uv[0])), int(round(tip_uv[1])))
            label_xy = (int(round(label_uv[0])), int(round(label_uv[1])))
            cv2.arrowedLine(
                canvas,
                origin_xy,
                tip_xy,
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )
            cv2.putText(
                canvas,
                label,
                (label_xy[0] + 1, label_xy[1] + 1),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                label_xy,
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.circle(canvas, origin_xy, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_kpt_with_core(
        self,
        canvas: np.ndarray,
        u: float,
        v: float,
        color,
        radius: float | None = None,
    ) -> None:
        u, v = int(round(u)), int(round(v))
        if not (0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]):
            return
        radius = max(1, int(round(
            self._value("radius_current", 4) if radius is None else radius
        )))
        core_radius = max(0, int(round(self._value("white_core_radius", 1))))
        cv2.circle(
            canvas,
            (u, v),
            radius,
            tuple(int(channel) for channel in color),
            -1,
            cv2.LINE_AA,
        )
        if core_radius:
            cv2.circle(canvas, (u, v), core_radius, (255, 255, 255), -1, cv2.LINE_AA)

    def _camera_matrix(self, vio_frame, shape):
        calibration = getattr(vio_frame, "calibration", None)
        if calibration is None:
            calibration = self.calibration
        if calibration is None:
            raise ValueError("VisualKpts requires camera intrinsics")
        fx, fy, cx, cy = calibration.intrinsics
        return np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def _index_hand_frames(hands) -> dict[int, object]:
        if hands is None:
            return {}
        return {
            int(frame.idx): frame
            for frame in getattr(hands, "hands", ())
        }

    @staticmethod
    def _hands(frame):
        if frame is None:
            return ()
        return tuple(
            hand
            for hand in (getattr(frame, "hand_l", None), getattr(frame, "hand_r", None))
            if hand is not None
        )

    @staticmethod
    def _side_hand(frame, side):
        return getattr(frame, side, None)

    @staticmethod
    def _index_tracks(document):
        document = document or {}
        frame_indices = [int(value) for value in document.get("frames", [])]
        indexed = {frame_idx: {} for frame_idx in frame_indices}
        for object_key, data in document.get("objects", {}).items():
            if not VisualKptsGenerator._is_object_key(object_key):
                continue
            tracks = data.get("tracks", []) if isinstance(data, Mapping) else data
            for position, frame_idx in enumerate(frame_indices):
                if position < len(tracks):
                    indexed[frame_idx][object_key] = tracks[position]
        return indexed

    @staticmethod
    def _index_visibility(document):
        document = document or {}
        frame_indices = [int(value) for value in document.get("frames", [])]
        indexed = {frame_idx: {} for frame_idx in frame_indices}
        for object_key, data in document.get("objects", {}).items():
            if not VisualKptsGenerator._is_object_key(object_key):
                continue
            visibility = data.get("visibility", []) if isinstance(data, Mapping) else []
            for position, frame_idx in enumerate(frame_indices):
                if position < len(visibility):
                    indexed[frame_idx][object_key] = visibility[position]
        return indexed

    @staticmethod
    def _is_object_key(object_key) -> bool:
        object_key = str(object_key)
        return object_key.startswith("obj") and object_key != "obj_and_arm"

    @staticmethod
    def _hand_grasp_score(hand) -> float:
        value = getattr(hand, "grasp_score", getattr(hand, "grasp_state", 0.0))
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    @classmethod
    def _interpolate_bgr(cls, color_start, color_end, ratio: float) -> tuple[int, int, int]:
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return tuple(
            int(start + (end - start) * ratio)
            for start, end in zip(color_start, color_end)
        )

    def _value(self, name, default):
        return float(self._setting(f"visualkpts_{name}", default))

    def _color(self, name, default):
        value = self._setting(name, default)
        if value is None:
            value = default
        return tuple(int(channel) for channel in value)

    def _hand_colors(self, is_closed: bool):
        names = self._HAND_CLOSED_COLORS if is_closed else self._HAND_OPEN_COLORS
        defaults = (
            (
                (255, 255, 255),
                (255, 255, 255),
                (255, 0, 255),
                (255, 0, 255),
            )
            if is_closed
            else (
                (0, 255, 255),
                (0, 160, 255),
                (0, 60, 255),
                (0, 0, 255),
            )
        )
        return [self._color(name, fallback) for name, fallback in zip(names, defaults)]

    def _object_colors(self):
        values = self._setting("obj_colors", None)
        if not values:
            values = [(255, 255, 0), (0, 255, 0)]
        return [tuple(int(channel) for channel in color) for color in values]

    def _setting(self, name, default):
        return self._setting_from(self.cfg, name, default)

    @staticmethod
    def _setting_from(config, name, default):
        if config is None:
            return default
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    @staticmethod
    def _atomic_write_json(path: Path, document: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        )
        temporary_path = Path(handle.name)
        try:
            with handle:
                json.dump(document, handle, indent=2, ensure_ascii=True)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
