"""Build the final per-frame training contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
    OPENCV_CAMERA_FRAME,
)
from utils.utils_artifact_store import FrameArtifactStore


class DatasetGenerator:
    """Consolidate object poses, hand poses, masks and image variants."""

    def __init__(self, unit_dir, cfg=None, store=None):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg
        self.store = store or FrameArtifactStore(self.unit_dir)

    def run(
        self,
        pose_document: dict,
        triangulation_document: dict,
        frame_data,
        frame_images: dict[int, np.ndarray],
        vio_result,
        fps: float,
        training_frames: set[int],
        finished_frames: set[int],
        video_path: str | Path | None = None,
    ) -> dict:
        frame_data_by_idx = {int(frame.frame_idx): frame for frame in frame_data}
        vio_by_idx = {
            int(frame.frame_idx): frame for frame in vio_result.trajectory.frames
        }
        fx, fy, cx, cy = np.asarray(
            vio_result.calibration.intrinsics, dtype=np.float64
        )
        camera_intrinsics = [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ]
        width, height = vio_result.calibration.resolution
        object_local_points = self._object_local_points(triangulation_document)
        written = []
        for pose_frame in pose_document.get("frames", []):
            frame_idx = int(pose_frame["frame_idx"])
            if frame_idx not in training_frames:
                continue
            if frame_idx not in frame_data_by_idx or frame_idx not in frame_images:
                raise ValueError(f"Missing object frame inputs for training frame {frame_idx}")
            vio_frame = vio_by_idx.get(frame_idx)
            if vio_frame is None:
                raise ValueError(f"Missing VIO frame for training frame {frame_idx}")
            frame_dir = self.store.frame_dir(frame_idx, True)
            rgb_path = frame_dir / "rgb.png"
            if not rgb_path.is_file():
                self.store.write_image(frame_idx, "rgb.png", frame_images[frame_idx], True)
            required_images = (
                "rgb_WoArm.png",
                "rgb_WArmObjKpts.png",
                "rgb_WoArm_WArmObjKpts.png",
            )
            required_masks = (
                "mask_arm.png",
                "mask_arm_and_obj.png",
            )
            missing_images = [
                name for name in required_images if not (frame_dir / name).is_file()
            ]
            missing_masks = [
                name for name in required_masks if not (frame_dir / name).is_file()
            ]
            if missing_images or missing_masks:
                raise FileNotFoundError(
                    f"Missing training artifacts for frame {frame_idx}: "
                    f"images={missing_images}, masks={missing_masks}"
                )

            detection = frame_data_by_idx[frame_idx]
            object_keypoints = self._transform_object_points(
                object_local_points,
                pose_frame["objects"],
            )
            output = {
                "schema_version": 1,
                "world_frame": ARIA_MPS_WORLD_FRAME,
                "world_origin": ARIA_MPS_WORLD_ORIGIN,
                "initial_heading": ARIA_MPS_INITIAL_HEADING,
                "metadata": {
                    "idx": frame_idx,
                    "ts": int(pose_frame["timestamp_ns"]),
                    "timestamp_ns": int(pose_frame["timestamp_ns"]),
                    "w": int(width),
                    "h": int(height),
                    "fps": float(fps),
                    "k": camera_intrinsics,
                    "camera_frame": OPENCV_CAMERA_FRAME,
                    "c2w": np.asarray(vio_frame.c2w).tolist(),
                    "camera_intrinsics": camera_intrinsics,
                    "anchor_key": pose_document["anchor_key"],
                    "is_finished": 1.0 if frame_idx in finished_frames else 0.0,
                    "world_transforms": {
                        "cam0": pose_document["cam0_c2w"],
                        "virtual_static_anchor": pose_document["anchor_to_world"],
                        "camera_to_world": np.asarray(vio_frame.c2w).tolist(),
                        "anchor_to_world": pose_document["anchor_to_world"],
                        "world_to_anchor": pose_document["world_to_anchor"],
                    },
                },
                "obs": {
                    "rgb_path": str(rgb_path),
                    "rgb_WoArm_path": str(frame_dir / "rgb_WoArm.png"),
                    "rgb_WArmObjKpts_path": str(frame_dir / "rgb_WArmObjKpts.png"),
                    "rgb_WoArm_WArmObjKpts_path": str(
                        frame_dir / "rgb_WoArm_WArmObjKpts.png"
                    ),
                    "mask_arm_path": str(
                        detection.combined_mask_path.parent / "mask_arm.png"
                    ),
                    "mask_obj_path": str(detection.combined_mask_path),
                    "source_video_path": str(video_path or self.unit_dir),
                    "source_frame_idx": frame_idx,
                    "object_masks": {
                        item.key: str(item.mask_path)
                        for item in detection.objects
                        if item.key.startswith("obj")
                    },
                    "objects_kpts": object_keypoints,
                },
                "entities": {
                    "hands": self._hands_for_frame(pose_frame),
                    "objects": pose_frame["objects"],
                },
            }
            path = frame_dir / "training_data.json"
            self._atomic_write_json(path, output)
            written.append(str(path))

        if len(written) != len(training_frames):
            missing = sorted(training_frames - {int(Path(path).parent.name) for path in written})
            raise ValueError(f"DatasetGen did not write training frames: {missing}")
        report = {
            "status": "completed",
            "frames": len(written),
            "finished_frames": sorted(int(frame) for frame in finished_frames),
            "path": str(self.store.all_data_dir),
        }
        report_path = self.store.module_vis_dir("datasetgen") / "report.json"
        self._atomic_write_json(report_path, report)
        report["report_path"] = str(report_path)
        return report

    @staticmethod
    def _hands_for_frame(pose_frame):
        return {
            side: {
                "T_hand_to_world": value["T_hand_to_world"],
                "grasp": value["grasp"],
            }
            for side, value in pose_frame.get("hands", {}).items()
            if value.get("T_hand_to_world") is not None
        }

    @staticmethod
    def _object_local_points(triangulation_document):
        local_points = {}
        for key, value in triangulation_document["objects"].items():
            points_world = np.asarray(value["points_3d_world"], dtype=np.float64)
            object_to_world = np.asarray(value["object_to_world_matrix"], dtype=np.float64)
            world_to_object = np.linalg.inv(object_to_world)
            local_points[key] = (
                world_to_object[:3, :3] @ points_world.T
            ).T + world_to_object[:3, 3]
        return local_points

    @staticmethod
    def _transform_object_points(local_points, frame_objects):
        result = {}
        for key, points in local_points.items():
            object_to_world = np.asarray(
                frame_objects[key]["T_obj_to_world"], dtype=np.float64
            )
            points_world = (
                object_to_world[:3, :3] @ points.T
            ).T + object_to_world[:3, 3]
            result[key] = {"world": points_world.tolist()}
        return result

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
