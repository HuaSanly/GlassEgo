"""Runtime perception orchestration for the stereo inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Mapping

import cv2
import numpy as np

from .DINOSAM import DinoSamSegmenter, MaskResult
from .LaMa import LaMaInpaint
from .StereoDepth import StereoDepthEstimator, StereoDepthResult
from .stereo_types import StereoFrame


@dataclass(frozen=True)
class ObjectState:
    """Object pose expressed in the left rectified camera frame, in meters."""

    key: str
    T_in_cam: np.ndarray
    mask: np.ndarray
    confidence: float
    valid_depth_ratio: float


class Perception:
    """Compose segmentation, stereo depth, pose estimation and inpainting."""

    def __init__(
        self,
        cfg: Mapping | None = None,
        segmenter=None,
        depth_estimator=None,
        inpainter=None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        values = dict(cfg or {})
        if "perception" in values:
            values = dict(values["perception"])

        self.anchor_key = str(values.get("anchor_key", "obj1"))
        self.object_prompts = {
            str(key): str(prompt)
            for key, prompt in dict(values.get("object_prompts", {})).items()
        }
        self.arm_prompt = str(
            values.get("arm_prompt", values.get("erase_prompt", "robot arm"))
        )
        self.mask_erode_px = int(values.get("mask_erode_px", 3))
        self.min_points = int(values.get("min_points", 100))
        if self.mask_erode_px < 0:
            raise ValueError("perception.mask_erode_px must be non-negative")
        if self.min_points < 3:
            raise ValueError("perception.min_points must be at least 3")

        dino_cfg = values.get("dino", {})
        stereo_cfg = values.get("stereo", {})
        lama_cfg = values.get("lama", {})
        self.segmenter = (
            segmenter
            if segmenter is not None
            else DinoSamSegmenter(dino_cfg, logger=self.logger)
        )
        self.depth_estimator = (
            depth_estimator
            if depth_estimator is not None
            else StereoDepthEstimator(stereo_cfg, logger=self.logger)
        )
        self.inpainter = (
            inpainter
            if inpainter is not None
            else LaMaInpaint(lama_cfg, logger=self.logger)
        )
        self.last_diagnostics: dict = {}

    @staticmethod
    def _validate_frame(frame: StereoFrame) -> None:
        if not isinstance(frame, StereoFrame):
            raise TypeError("frame must be a StereoFrame")
        if frame.left.ndim != 3 or frame.left.shape[2] != 3:
            raise ValueError("StereoFrame.left must have shape HxWx3")
        if frame.right.ndim != 3 or frame.right.shape[2] != 3:
            raise ValueError("StereoFrame.right must have shape HxWx3")
        if frame.left.shape[:2] != frame.right.shape[:2]:
            raise ValueError("left and right stereo images must have equal shape")

    @staticmethod
    def _as_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        values = np.asarray(mask).astype(bool, copy=False)
        if values.shape != shape:
            raise ValueError(f"mask shape {values.shape} does not match frame {shape}")
        return values

    def _erode_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.mask_erode_px == 0:
            return mask
        size = self.mask_erode_px * 2 + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    @staticmethod
    def _filter_points(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape Nx3")
        values = values[np.all(np.isfinite(values), axis=1)]
        values = values[values[:, 2] > 0]
        if len(values) < 3:
            return values

        z_low, z_high = np.percentile(values[:, 2], [5.0, 95.0])
        values = values[(values[:, 2] >= z_low) & (values[:, 2] <= z_high)]
        if len(values) < 3:
            return values

        center = np.median(values, axis=0)
        distances = np.linalg.norm(values - center, axis=1)
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        if mad > 1e-6:
            limit = median_distance + 3.5 * 1.4826 * mad
            values = values[distances <= limit]
        return values

    @staticmethod
    def _pca2_pose(
        points: np.ndarray,
        is_anchor: bool,
        anchor_center: np.ndarray | None,
    ) -> np.ndarray:
        center = points.mean(axis=0)
        centered = points - center[None, :]
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        v_long, v_mid, v_short = eigenvectors[:, order].T

        cam_down = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        is_vertical = abs(np.dot(v_long, cam_down)) > abs(np.dot(v_long, cam_right))

        if is_vertical:
            y_axis = v_long.copy()
            if np.dot(y_axis, cam_down) < 0:
                y_axis = -y_axis
            if is_anchor or anchor_center is None:
                x_axis = v_mid if abs(np.dot(v_mid, cam_right)) > abs(
                    np.dot(v_short, cam_right)
                ) else v_short
                if np.dot(x_axis, cam_right) < 0:
                    x_axis = -x_axis
            else:
                toward_anchor = anchor_center - center
                x_axis = v_mid if abs(np.dot(v_mid, toward_anchor)) > abs(
                    np.dot(v_short, toward_anchor)
                ) else v_short
                if np.dot(x_axis, toward_anchor) < 0:
                    x_axis = -x_axis
        else:
            x_axis = v_long.copy()
            if is_anchor or anchor_center is None:
                if np.dot(x_axis, cam_right) < 0:
                    x_axis = -x_axis
            else:
                toward_anchor = anchor_center - center
                if np.dot(x_axis, toward_anchor) < 0:
                    x_axis = -x_axis
            y_axis = v_mid if abs(np.dot(v_mid, cam_down)) > abs(
                np.dot(v_short, cam_down)
            ) else v_short
            if np.dot(y_axis, cam_down) < 0:
                y_axis = -y_axis

        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
        x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1.0

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation.astype(np.float32)
        transform[:3, 3] = center.astype(np.float32)
        return transform

    def estimate_objects(self, frame: StereoFrame) -> Dict[str, ObjectState]:
        self._validate_frame(frame)
        self.last_diagnostics = {"objects": {}, "depth": {}}
        if self.anchor_key not in self.object_prompts:
            raise ValueError(
                f"anchor_key {self.anchor_key!r} is missing from object_prompts"
            )

        masks: Dict[str, MaskResult] = self.segmenter.segment(
            frame,
            self.object_prompts,
        )
        depth: StereoDepthResult = self.depth_estimator.estimate(frame)
        self.last_diagnostics["depth"] = {
            "valid_ratio": float(np.mean(depth.valid_mask)),
        }

        ordered_keys = [self.anchor_key] + [
            key for key in self.object_prompts if key != self.anchor_key
        ]
        states: Dict[str, ObjectState] = {}
        anchor_center = None
        for key in ordered_keys:
            detection = masks.get(key)
            object_diag = {"confidence": 0.0, "valid_depth_ratio": 0.0}
            if detection is None:
                object_diag["error"] = "missing_detection"
                self.last_diagnostics["objects"][key] = object_diag
                if key == self.anchor_key:
                    raise ValueError(f"anchor object {key!r} was not detected")
                continue

            mask = self._as_mask(detection.mask, frame.left.shape[:2])
            object_diag["confidence"] = float(detection.confidence)
            pixel_count = int(np.count_nonzero(mask))
            valid_depth_ratio = float(
                np.count_nonzero(mask & depth.valid_mask) / max(pixel_count, 1)
            )
            object_diag["valid_depth_ratio"] = valid_depth_ratio
            points_mask = self._erode_mask(mask)
            points = self.depth_estimator.points_from_mask(
                frame,
                points_mask,
                depth,
            )
            points = self._filter_points(points)
            object_diag["valid_points"] = int(len(points))
            if len(points) < self.min_points:
                object_diag["error"] = "insufficient_3d_points"
                self.last_diagnostics["objects"][key] = object_diag
                if key == self.anchor_key:
                    raise ValueError(
                        f"anchor object {key!r} has fewer than "
                        f"{self.min_points} valid 3D points"
                    )
                continue

            transform = self._pca2_pose(
                points,
                is_anchor=key == self.anchor_key,
                anchor_center=anchor_center,
            )
            if key == self.anchor_key:
                anchor_center = transform[:3, 3].astype(np.float64)
            states[key] = ObjectState(
                key=key,
                T_in_cam=transform,
                mask=mask,
                confidence=float(detection.confidence),
                valid_depth_ratio=valid_depth_ratio,
            )
            self.last_diagnostics["objects"][key] = object_diag

        if self.anchor_key not in states:
            raise ValueError(f"anchor object {self.anchor_key!r} has no valid pose")
        return states

    def make_clean_image(
        self,
        frame: StereoFrame,
        ee_poses_in_cam: Dict[str, np.ndarray] | None = None,
        grippers: Dict[str, float] | None = None,
    ) -> np.ndarray:
        del ee_poses_in_cam, grippers
        self._validate_frame(frame)
        if not self.arm_prompt.strip():
            return frame.left.copy()

        result = self.segmenter.segment(frame, {"arm": self.arm_prompt}).get("arm")
        if result is None or not np.any(result.mask):
            self.last_diagnostics.setdefault("clean_image", {})[
                "arm_mask"
            ] = "missing"
            return frame.left.copy()

        self.last_diagnostics.setdefault("clean_image", {})["arm_confidence"] = float(
            result.confidence
        )
        return self.inpainter.inpaint(frame, result.mask)

    def close(self) -> None:
        for module in (self.segmenter, self.depth_estimator, self.inpainter):
            close = getattr(module, "close", None)
            if close is not None:
                close()


__all__ = ["ObjectState", "Perception"]
